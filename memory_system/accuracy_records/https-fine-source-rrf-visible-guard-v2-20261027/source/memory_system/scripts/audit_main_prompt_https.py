"""Replay actual main admission and a fixed-upstream 24 KB counterfactual.

Uses the run's frozen implementation and recorded post-CE token headroom.
This isolates admission only, not a counterfactual listwise judgment, final
packet or answer. References are scored after all input selections are fixed.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def progress_inputs(run, plan, fixture):
    """Read only completed lines; never fabricate terminal manifests for replay."""
    inputs={}
    def read(path):
        data=path.read_bytes()
        if data and not data.endswith(b'\n'):
            data=data[:data.rfind(b'\n')+1]
        inputs[str(path)]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        return [json.loads(line) for line in data.splitlines() if line.strip()]
    manifest=json.loads((run/'isolated-manifest.json').read_text())
    prepared=Path(manifest['data'])
    assert sha(prepared)==plan['data_sha256']
    conversations=read(prepared)
    questions={(c['conversation_id'],q['id']):q for c in conversations for q in c['qa']}
    results=read(run/'completed/data/results/personamem-v2-32k.jsonl')
    traces=read(run/'server-search-debug.jsonl')
    by_search={t['search_id']:t for t in traces}
    assert len(by_search)==len(traces)
    assert len({r['qa_id'] for r in results})==len(results)
    expected={(r['conversation_id'],r['qa_id']):r for r in fixture['requests']}
    jobs=[]
    for result in results:
        assert not result.get('error_stage')
        t=by_search[result['search_id']]
        key=result['conversation_id'],result['qa_id']
        q=questions[key];requested=expected[key]
        assert result['user_id']==t['user_id'] and result['packet_hash']==t['packet_hash']
        assert t['query']==q['question'] and t['options']==q['options']
        assert t['plan']==requested['plan'] and t['anchor_time']==requested['anchor_time']
        assert all((t.get('requested_include_sensitive') if k=='include_sensitive' else t[k])==v
                   for k,v in requested['request'].items())
        assert result['fake'] is False and t['fake'] is False and result['transport']=='https'
        assert t['cascade']['cross_encoder']['status']=='ok'
        assert all(plan['source_files'][name]==value for name,value in t['versions']['file_hashes'].items())
        pool={v['id']:v for v in t['ranked']}
        jobs.append(dict(qa_id=result['qa_id'],conversation_id=result['conversation_id'],t=t,
                         coarse=[pool[mid] for mid in t['cascade']['coarse']['selected_ids']]))
    assert jobs
    return jobs,manifest,conversations,{r['qa_id']:r for r in results},inputs


def main(args):
    root=args.run.resolve();snapshot=root/'source'
    sys.path.insert(0,str(snapshot/'memory_system'))
    sys.path.insert(0,str(snapshot/'memory_system'/'scripts'))
    from app import cascade_rerank,config,schemas
    from probe_full_option_ce import prepare,base,coverage
    from evaluate_recall_context import reference_turns
    from https_experiment_settings import fixture_values
    plan=json.loads((root/'plan.json').read_text())
    fixture=json.loads((root/'fixture.json').read_text())
    assert plan['toggle']=='RERANK_MAX_PROMPT_BYTES'
    values=fixture_values(fixture)
    assert plan['setting_values']==values
    assert sha(root/'fixture.json')==plan['fixture_sha256']
    if not args.progress:
        assert (root/'exit-code').read_text().strip()=='0'
    elif 'progress' not in args.output.name:
        raise ValueError('Progress output must have progress in its name')
    for name,expected in plan['source_files'].items():
        assert sha(snapshot/name)==expected,name
    selected=[];raw_by_arm={};conversations=None;checks={};snapshots={}
    with patch('socket.socket.connect',side_effect=AssertionError('Offline audit')):
        for arm in ('control','treatment'):
            run=root/arm
            if args.progress:
                jobs,manifest,conversations,results,inputs=progress_inputs(run,plan,fixture)
                snapshots[arm]=inputs
            else:
                jobs,manifest,conversations=prepare(run)
                assert len(jobs)==plan['questions']
                assert manifest['client_exit_code']==0 and manifest['audit_exit_code']==0
                results={r['qa_id']:r for r in base.records(run/'completed/data/results/personamem-v2-32k.jsonl')}
            checks[arm]=dict(actual_admissions=0,actual_prompt_bytes=0,total_reservations=0)
            for job in jobs:
                t=job['t'];settings=t['versions']['settings'];c=t['cascade']
                assert type(settings['RERANK_MAX_PROMPT_BYTES']) is int
                assert settings['RERANK_MAX_PROMPT_BYTES']==values[arm]==c['admission_budget']['prompt_limit_bytes']
                scores=[v['_ce_score'] for v in job['coarse']]
                actual=base.replay(job,scores)
                assert [v['id'] for v in actual['fine']]==c['fine']['selected_ids']
                p=copy.deepcopy(t['plan'])
                p['_coverage_requirements']=[{k:r[k] for k in ('id','text','origin') if k in r}
                    for r in t['coverage_manifest']['coverage']['requirements']]
                p['_fusion_edges']=t['fusion_edges'];p['_fusion_triples']=t['fusion_triples']
                p['_fine_path_groups']=c['fine']['selection']['adaptive']['path_groups']
                req=schemas.SearchRequest(user_id=t['user_id'],query=t['query'],options=t['options'])
                with patch.multiple(config,**settings):
                    request=cascade_rerank.listwise_request(req,p,actual['main'])
                headroom=c['admission_budget']['token_headroom']
                assert len(request['prompt'].encode())==actual['main_bytes']==c['listwise']['input_bytes']
                assert actual['main_bytes']<=values[arm]
                assert headroom is None or request['reservation']<=headroom
                assert [v['id'] for v in actual['main']]==c['listwise']['candidate_ids']
                assert len(actual['main'])<=settings['CASCADE_LLM_LIMIT']+settings['FINE_EXTRA_LIMIT']
                changed=copy.deepcopy(job)
                changed['t']['versions']['settings']['RERANK_MAX_PROMPT_BYTES']=24000
                baseline=base.replay(changed,scores)
                assert [v['id'] for v in baseline['fine']]==[v['id'] for v in actual['fine']]
                checks[arm]['actual_admissions']+=1
                checks[arm]['actual_prompt_bytes']+=actual['main_bytes']
                checks[arm]['total_reservations']+=request['reservation']
                selected.append((arm,job,baseline,actual,results[job['qa_id']]))
            raw_by_arm[arm]={}
            with sqlite3.connect((run/'memory.db').as_uri()+'?mode=ro',uri=True) as db:
                db.row_factory=sqlite3.Row
                for row in db.execute('SELECT * FROM source_messages'):
                    raw_by_arm[arm].setdefault(row['user_id'],{})[row['request_id'],row['message_index']]=dict(row)
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream:gold=list(csv.DictReader(stream))
    convs={c['conversation_id']:c for c in conversations};rows=[]
    import audit_https_funnel as funnel
    for arm,job,baseline,actual,result in selected:
        conv=convs[job['conversation_id']];q=next(q for q in conv['qa'] if q['id']==job['qa_id'])
        turns=reference_turns(q,conv['source_history'],gold[int(job['qa_id'])])
        raw=raw_by_arm[arm][job['t']['user_id']]
        before=coverage(job,baseline,turns,raw);after=coverage(job,actual,turns,raw)
        catalog=result.get('answer_source_catalog',[])+result.get('answer_constraint_catalog',[])
        final=[c['full'] for c in funnel.turn_coverage(turns,catalog,raw)]
        rows.append(dict(arm=arm,qa_id=job['qa_id'],conversation_id=job['conversation_id'],
            before=before,after=after,actual_catalog=final,actual_score=result['score'],
            added_ids=sorted(set(after['main']['ids'])-set(before['main']['ids'])),
            removed_ids=sorted(set(before['main']['ids'])-set(after['main']['ids']))))
    summaries={}
    for arm in ('control','treatment'):
        cohort=[r for r in rows if r['arm']==arm]
        summaries[arm]=dict(**checks[arm],
            counterfactual_main_reference_questions=sum(any(r['before']['main']['user']) for r in cohort),
            actual_main_reference_questions=sum(any(r['after']['main']['user']) for r in cohort),
            counterfactual_main_user_turns=sum(sum(r['before']['main']['user']) for r in cohort),
            actual_main_user_turns=sum(sum(r['after']['main']['user']) for r in cohort),
            gained=[r['qa_id'] for r in cohort if any(r['after']['main']['user']) and not any(r['before']['main']['user'])],
            lost=[r['qa_id'] for r in cohort if any(r['before']['main']['user']) and not any(r['after']['main']['user'])],
            fewer_turns=[r['qa_id'] for r in cohort if sum(r['after']['main']['user'])<sum(r['before']['main']['user'])],
            newly_admitted_reference_reaches_actual_catalog=[r['qa_id'] for r in cohort if any(
                a and not b and f for a,b,f in zip(r['after']['main']['user'],r['before']['main']['user'],r['actual_catalog']))],
            added_candidates=sum(len(r['added_ids']) for r in cohort),
            removed_candidates=sum(len(r['removed_ids']) for r in cohort))
    with args.output.open('x') as stream:
        json.dump(dict(status='passed',progress_snapshot=args.progress,
            sampled_at=datetime.now(timezone.utc).isoformat(),input_prefixes=snapshots,
            summaries=summaries,rows=rows,limitation=__doc__,
            auditor_sha256=sha(Path(__file__)),fixture_sha256=plan['fixture_sha256'],
            trace_sha256=({a:sha(root/a/'server-search-debug.jsonl') for a in ('control','treatment')}
                          if not args.progress else None)),
            stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps(summaries,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--progress',action='store_true')
    parser.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(parser.parse_args())
