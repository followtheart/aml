"""Audit actual fine/main selection and a fixed-upstream insurance-off replay.

Uses the experiment's frozen code and actual CE scores/headroom. This isolates
selection/admission only, not a counterfactual final packet or reader answer.
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


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    root=args.run.resolve();snapshot=root/'source'
    sys.path.insert(0,str(snapshot/'memory_system'));sys.path.insert(0,str(snapshot/'memory_system/scripts'))
    from app import config,fine_evidence
    from probe_full_option_ce import prepare,base,coverage
    from evaluate_recall_context import reference_turns
    from audit_main_prompt_https import progress_inputs
    import audit_https_funnel as funnel
    plan=json.loads((root/'plan.json').read_text());fixture=json.loads((root/'fixture.json').read_text())
    toggle=plan['toggle']
    assert toggle in ('FINE_NEW_COARSE_INSURANCE','FINE_NEW_COARSE_MISSING_ONLY')
    assert plan['setting_values']==dict(control=False,treatment=True)
    assert sha(root/'fixture.json')==plan['fixture_sha256']
    if args.progress:
        assert 'progress' in args.output.name
    else:assert (root/'exit-code').read_text().strip()=='0'
    for name,digest in plan['source_files'].items():assert sha(snapshot/name)==digest,name
    selected=[];raws={};snapshots={}
    with patch('socket.socket.connect',side_effect=AssertionError('Offline audit')):
        for arm in ('control','treatment'):
            run=root/arm
            if args.progress:
                jobs,manifest,conversations,results,snapshots[arm]=progress_inputs(run,plan,fixture)
            else:
                jobs,manifest,conversations=prepare(run)
                assert len(jobs)==plan['questions']
                assert manifest['client_exit_code']==manifest['audit_exit_code']==0
                assert manifest['server_exit_code'] in (0,-15)
                results={r['qa_id']:r for r in base.records(run/'completed/data/results/personamem-v2-32k.jsonl')}
            for job in jobs:
                t=job['t'];cfg=t['versions']['settings'];c=t['cascade']
                assert cfg[toggle] is (arm=='treatment')
                if toggle=='FINE_NEW_COARSE_MISSING_ONLY':assert not cfg['FINE_NEW_COARSE_INSURANCE']
                assert cfg['COARSE_EVIDENCE_EXTEND'] and cfg['FINE_COARSE_CORE_QUOTA'] and not cfg['FINE_SEMANTIC_RESERVE']
                assert cfg['RERANK_MAX_PROMPT_BYTES']==24000
                scores=[v['_ce_score'] for v in job['coarse']];observed=[];original=fine_evidence.propose
                def capture(*a,**kw):
                    chosen,trace=original(*a,**kw);observed.append(trace);return chosen,trace
                with patch.object(fine_evidence,'propose',capture):actual=base.replay(job,scores)
                assert [v['id'] for v in actual['fine']]==c['fine']['selected_ids']
                assert [v['id'] for v in actual['main']]==c['listwise']['candidate_ids']
                assert actual['main_bytes']==c['listwise']['input_bytes']<=24000
                adaptive=observed[-1]['adaptive'];recorded=c['fine']['selection']['adaptive']
                assert adaptive['core_ids']==recorded['core_ids']
                assert adaptive['additions']==recorded['additions']
                assert len(actual['fine'])<=adaptive['capacity']
                if arm=='treatment':
                    info=adaptive['new_coarse_insurance'];assert info==recorded['new_coarse_insurance']
                    fulltop={v['id'] for v in sorted(job['coarse'],key=fine_evidence.key,reverse=True)[:cfg['CASCADE_FINE_LIMIT']]}
                    assert set(info['eligible_ids'])<=fulltop-set(adaptive['core_pool_ids'])
                    used=[v for v in adaptive['additions'] if v['reason']=='new_coarse_core_insurance']
                    assert len(used)<=1 and info['used']==bool(used)
                    assert all(v['candidate_id'] in info['eligible_ids'] and len(v['ids'])==1 for v in used)
                changed=copy.deepcopy(job)
                changed['t']['versions']['settings'].update(FINE_NEW_COARSE_INSURANCE=False,FINE_NEW_COARSE_MISSING_ONLY=False)
                with patch.object(fine_evidence,'propose',capture):baseline=base.replay(changed,scores)
                assert observed[-1]['adaptive']['core_ids']==adaptive['core_ids']
                assert observed[-1]['adaptive']['capacity']==adaptive['capacity']
                if toggle=='FINE_NEW_COARSE_MISSING_ONLY' and arm=='treatment':
                    assert info['missing_only'] is True
                    assert info['baseline_fine_ids']==[v['id'] for v in baseline['fine']]
                    assert not set(info['eligible_ids'])&set(info['baseline_fine_ids'])
                    if not info['used']:assert baseline==actual
                if arm=='control':assert baseline==actual
                selected.append((arm,job,baseline,actual,results[job['qa_id']],adaptive))
            raws[arm]={}
            with sqlite3.connect((run/'memory.db').as_uri()+'?mode=ro',uri=True) as db:
                db.row_factory=sqlite3.Row
                for row in db.execute('SELECT * FROM source_messages'):
                    raws[arm].setdefault(row['user_id'],{})[row['request_id'],row['message_index']]=dict(row)
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream:gold=list(csv.DictReader(stream))
    convs={c['conversation_id']:c for c in conversations};rows=[]
    for arm,job,baseline,actual,result,adaptive in selected:
        conv=convs[job['conversation_id']];q=next(q for q in conv['qa'] if q['id']==job['qa_id'])
        turns=reference_turns(q,conv['source_history'],gold[int(job['qa_id'])]);raw=raws[arm][job['t']['user_id']]
        before=coverage(job,baseline,turns,raw);after=coverage(job,actual,turns,raw)
        catalog=result.get('answer_source_catalog',[])+result.get('answer_constraint_catalog',[])
        final=[v['full'] for v in funnel.turn_coverage(turns,catalog,raw)]
        rows.append(dict(arm=arm,qa_id=job['qa_id'],before=before,after=after,actual_catalog=final,
            insurance=adaptive.get('new_coarse_insurance'),actual_score=result['score']))
    summaries={}
    for arm in ('control','treatment'):
        cohort=[r for r in rows if r['arm']==arm]
        summaries[arm]=dict(questions=len(cohort),used=sum(bool(r['insurance'] and r['insurance']['used']) for r in cohort),stages={})
        for stage in ('fine','main'):
            summaries[arm]['stages'][stage]=dict(
                before_questions=sum(any(r['before'][stage]['user']) for r in cohort),after_questions=sum(any(r['after'][stage]['user']) for r in cohort),
                before_turns=sum(sum(r['before'][stage]['user']) for r in cohort),after_turns=sum(sum(r['after'][stage]['user']) for r in cohort),
                gained=[r['qa_id'] for r in cohort if any(r['after'][stage]['user']) and not any(r['before'][stage]['user'])],
                lost=[r['qa_id'] for r in cohort if any(r['before'][stage]['user']) and not any(r['after'][stage]['user'])],
                fewer_turns=[r['qa_id'] for r in cohort if sum(r['after'][stage]['user'])<sum(r['before'][stage]['user'])])
        summaries[arm]['new_main_reference_reaches_actual_catalog']=[r['qa_id'] for r in cohort if any(
            a and not b and f for a,b,f in zip(r['after']['main']['user'],r['before']['main']['user'],r['actual_catalog']))]
    with args.output.open('x') as stream:
        json.dump(dict(status='passed',progress_snapshot=args.progress,summary=summaries,rows=rows,input_prefixes=snapshots,
            auditor_sha256=sha(Path(__file__)),fixture_sha256=plan['fixture_sha256'],limitation=__doc__),stream,indent=2);stream.write('\n')
    print(json.dumps(summaries))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,required=True);p.add_argument('--progress',action='store_true')
    main(p.parse_args())
