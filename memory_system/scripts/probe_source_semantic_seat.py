"""One source-CE nominated extra seat, screening on original audited cohorts.

Uses full archived scores on their original inputs, not a fresh HTTPS result.
Preserves core, whole candidate contents, dependencies, redundancy and limits.
All selections precede reference scoring. No provider calls. Token sensitivity
subtracts the original complete-source CE usage; live scheduling is not replayed.
"""
import argparse
import copy
import csv
import inspect
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import fine_evidence,config
from probe_full_option_ce import prepare,base,coverage,sha
from probe_source_ce import build_views
from evaluate_recall_context import reference_turns


def transformed_propose():
    """Four guarded edits to existing selection; no separate mirrored algorithm."""
    original=inspect.getsource(fine_evidence.propose)
    replacements=[
        ('semantic_reserve=False):','semantic_reserve=False, source_scores=None):'),
        ("semantic_ids = {c['id'] for c in core_pool[:max(0, count)]\n"
         "                    if type(c.get('_ce_score')) in (int, float)\n"
         "                    and math.isfinite(c['_ce_score']) and visible[c['id']]}",
         "semantic_ids = {mid for mid in (source_scores or {}) if visible[mid]}"),
        ("key(c) if semantic else (-1e30, 0)",
         "(source_scores[c['id']], key(c)) if semantic else (-1e30, (0, 0))"),
        ("'ce_core_insurance' if semantic", "'source_semantic_insurance' if semantic"),
    ]
    revised=original
    for old,new in replacements:
        if revised.count(old)!=1:
            raise ValueError('Production selector changed; inspect transformation')
        revised=revised.replace(old,new)
    env=dict(vars(fine_evidence))
    exec(compile(revised,'<source_semantic_seat>','exec'),env)
    return env['propose'],replacements


def main(args):
    root=args.archive;cohort=args.cohort
    audited=json.loads((root/'audit.json').read_text())['cohorts'][cohort]
    for suffix,digest in audited['artifact_hashes'].items():
        if sha(root/f'{cohort}.{suffix}')!=digest:
            raise ValueError('CE archive changed')
    plan=json.loads((root/f'{cohort}.plan.json').read_text())
    run=Path(plan['run']);old={j['qa_id']:j for j in plan['jobs']}
    scores={};usage={}
    for e in base.records(root/f'{cohort}.live.jsonl'):
        qa=e['qa_id']
        if e.get('error') or e['view_ids']!=old[qa]['batches'][e['batch']]:
            raise ValueError('CE batch failed/mismatched')
        if len(e['view_ids'])!=len(e['scores']):raise ValueError('Missing CE scores')
        scores.setdefault(qa,{}).update(zip(e['view_ids'],e['scores']))
        usage[qa]=usage.get(qa,0)+e['tokens']
    revised,replacements=transformed_propose();fixed=[]
    with patch('socket.socket.connect',side_effect=AssertionError('Offline')):
        jobs,manifest,conversations=prepare(run)
        if {j['qa_id'] for j in jobs}!=set(old):raise ValueError('Cohort mismatch')
        if sha(run/'server-search-debug.jsonl')!=plan['source_trace_sha256']:
            raise ValueError('Source traces changed')
        for job in jobs:
            qa=job['qa_id'];t=job['t'];prior=old[qa]
            with patch.multiple(config,**t['versions']['settings']):
                if config.FINE_SEMANTIC_RESERVE:raise ValueError('Only source semantic intervention allowed')
                if t['cascade']['cross_encoder']['query']!=prior['query']:
                    raise ValueError('Query mismatch')
                views,members,batches,omitted=build_views(job['coarse'],t['user_id'],prior['query'],max_sources=None)
            if (list(views.values())!=prior['views'] or members!=prior['members'] or omitted!=prior['omitted']
                    or [[v['id'] for v in b] for b in batches]!=prior['batches'] or set(scores[qa])!=set(views)):
                raise ValueError('Source scoring plan did not reproduce')
            original_scores=[c['_ce_score'] for c in job['coarse']]
            original=base.replay(job,original_scores)
            historical=t['cascade']['fine']['selection']['adaptive']
            core=set(historical['core_ids'])
            represented={key for mid in core for key in members.get(mid,[])}
            # Source identity and qualification stay local to this exact input.
            nominated={mid:max(scores[qa][key] for key in keys if key not in represented)
                       for mid,keys in members.items() if mid not in core and any(key not in represented for key in keys)}
            traces=[]
            def proposed(*a,**kw):
                chosen,trace=revised(*a,**dict(kw,semantic_reserve=True,source_scores=nominated))
                assert trace['adaptive']['core_ids']==historical['core_ids']
                assert trace['adaptive']['capacity']==historical['capacity']
                semantic=[v for v in trace['adaptive']['additions'] if v['reason']=='source_semantic_insurance']
                assert len(semantic)<=1 and all(len(v['ids'])==1 for v in semantic)
                trace['adaptive']['semantic_reserve']=dict(limit=1,used=bool(semantic),
                    rank_scope='novel_complete_user_source_ce',eligible_ids=sorted(nominated))
                traces.append(trace)
                return chosen,trace
            # Prove transformed code with no nominations leaves baseline intact.
            def empty(*a,**kw):return revised(*a,**dict(kw,semantic_reserve=True,source_scores={}))
            with patch.object(fine_evidence,'propose',empty):
                empty_result=base.replay(job,original_scores)
            assert empty_result==original
            with patch.object(fine_evidence,'propose',proposed):
                candidate=base.replay(job,original_scores)
                adjusted=copy.deepcopy(job)
                headroom=adjusted['t']['cascade']['admission_budget']['token_headroom']
                if headroom is not None:
                    adjusted['t']['cascade']['admission_budget']['token_headroom']=max(0,headroom-usage[qa])
                cost_adjusted=base.replay(adjusted,original_scores)
            assert len(traces)==2
            fixed.append((job,dict(original=original,proposed=candidate,cost_adjusted=cost_adjusted),
                          nominated,traces[0]['adaptive']))
    # Freeze every cohort selection before reading any reference labels.
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream:gold=list(csv.DictReader(stream))
    raw={}
    with sqlite3.connect((run/'memory.db').resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        for row in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(row['user_id'],{})[row['request_id'],row['message_index']]=dict(row)
    convs={c['conversation_id']:c for c in conversations};rows=[]
    for job,variants,nominated,adaptive in fixed:
        conv=convs[job['conversation_id']];q=next(q for q in conv['qa'] if q['id']==job['qa_id'])
        turns=reference_turns(q,conv['source_history'],gold[int(job['qa_id'])])
        rows.append(dict(qa_id=job['qa_id'],source_scores=nominated,adaptive=adaptive,extra_tokens=usage[job['qa_id']],
            variants={name:coverage(job,value,turns,raw[job['t']['user_id']]) for name,value in variants.items()}))
    summary=dict(cohort=cohort,questions=len(rows),core_preserved=len(rows),empty_intervention_exact=len(rows),provider_calls=0,
        archived_source_ce_calls=audited['calls'],archived_source_ce_tokens=sum(usage.values()),
        source_seat_used=sum(any(v['reason']=='source_semantic_insurance' for v in r['adaptive']['additions']) for r in rows),variants={})
    for name in ('original','proposed','cost_adjusted'):
        summary['variants'][name]={stage:dict(
            reference_questions=sum(any(r['variants'][name][stage]['user']) for r in rows),
            user_turns=sum(sum(r['variants'][name][stage]['user']) for r in rows),
            changed_order=sum(r['variants'][name][stage]['ids']!=r['variants']['original'][stage]['ids'] for r in rows),
            gained=[r['qa_id'] for r in rows if any(r['variants'][name][stage]['user']) and not any(r['variants']['original'][stage]['user'])],
            lost=[r['qa_id'] for r in rows if any(r['variants']['original'][stage]['user']) and not any(r['variants'][name][stage]['user'])],
            fewer_turns=[r['qa_id'] for r in rows if sum(r['variants'][name][stage]['user'])<sum(r['variants']['original'][stage]['user'])]) for stage in ('fine','main')}
    with args.output.open('x') as stream:
        json.dump(dict(summary=summary,rows=rows,run=str(run.resolve()),limitation=__doc__,transformations=replacements,
            input_hashes={str(p.resolve()):sha(p) for p in (Path(__file__),Path(fine_evidence.__file__),root/'audit.json',
                run/'server-search-debug.jsonl',args.gold_csv)},archived_hashes=audited['artifact_hashes']),stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=Path('runs/all-user-source-ce-20261024'))
    p.add_argument('--cohort',choices=['regression','validation'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
