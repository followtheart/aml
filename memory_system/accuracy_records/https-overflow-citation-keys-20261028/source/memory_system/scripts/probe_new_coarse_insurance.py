"""Offline extra-seat eligibility for new-coarse candidates in full CE top core.

Preserve the original core pool and selected core. Spend at most one existing
extra seat on a full-pool top-count candidate excluded by the new-coarse quota.
No source changes, synthetic scores, provider calls or label-driven selection.
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
from app import config,fine_evidence
from probe_full_option_ce import prepare,base,coverage,sha
from evaluate_recall_context import reference_turns


def proposal():
    original=inspect.getsource(fine_evidence.propose)
    replacements=[
        ("semantic_ids = {c['id'] for c in core_pool[:max(0, count)]",
         "semantic_ids = {c['id'] for c in ordered[:max(0, count)]\n"
         "                    if c['id'] not in set(core_pool_ids or by_id)"),
        ("                    if type(c.get('_ce_score')) in (int, float)",
         "                    and type(c.get('_ce_score')) in (int, float)"),
        ("'ce_core_insurance' if semantic", "'new_coarse_core_insurance' if semantic"),
        ("rank_scope='core_pool'", "rank_scope='full_coarse_outside_core_pool'"),
        ("eligible_ids=[c['id'] for c in core_pool if c['id'] in semantic_ids]",
         "eligible_ids=[c['id'] for c in ordered if c['id'] in semantic_ids]"),
    ]
    revised=original
    for old,new in replacements:
        if revised.count(old)!=1:raise ValueError('Inspect changed production selector')
        revised=revised.replace(old,new)
    env=dict(vars(fine_evidence));exec(compile(revised,'<new_coarse_insurance>','exec'),env)
    return env['propose'],replacements


def main(args):
    revised,replacements=proposal();fixed=[]
    with patch('socket.socket.connect',side_effect=AssertionError('Offline')):
        jobs,manifest,conversations=prepare(args.run)
        for job in jobs:
            t=job['t'];settings=t['versions']['settings']
            if not settings['FINE_COARSE_CORE_QUOTA'] or not settings['COARSE_EVIDENCE_EXTEND'] or settings['FINE_SEMANTIC_RESERVE']:
                raise ValueError('Requires existing coarse quota and no other semantic intervention')
            saved=copy.deepcopy(job['coarse']);scores=[c['_ce_score'] for c in job['coarse']]
            original=base.replay(job,scores);historical=t['cascade']['fine']['selection']['adaptive']
            captured=[]
            def propose(*a,**kw):
                chosen,trace=revised(*a,**dict(kw,semantic_reserve=True))
                assert trace['adaptive']['core_ids']==historical['core_ids']
                assert trace['adaptive']['capacity']==historical['capacity']
                eligible=trace['adaptive']['semantic_reserve']['eligible_ids']
                full_top={c['id'] for c in sorted(a[0],key=fine_evidence.key,reverse=True)[:a[1]]}
                assert set(eligible)<=full_top-set(kw['core_pool_ids'])
                used=[v for v in trace['adaptive']['additions'] if v['reason']=='new_coarse_core_insurance']
                assert len(used)<=1 and all(len(v['ids'])==1 and v['candidate_id'] in eligible for v in used)
                captured.append(trace)
                return chosen,trace
            with patch.object(fine_evidence,'propose',propose):candidate=base.replay(job,scores)
            assert len(captured)==1 and job['coarse']==saved
            adaptive=captured[0]['adaptive']
            if not adaptive['semantic_reserve']['eligible_ids']:assert candidate==original
            fixed.append((job,original,candidate,adaptive))
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream:gold=list(csv.DictReader(stream))
    raw={}
    with sqlite3.connect((args.run/'memory.db').resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        for row in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(row['user_id'],{})[row['request_id'],row['message_index']]=dict(row)
    convs={c['conversation_id']:c for c in conversations};rows=[]
    for job,original,candidate,adaptive in fixed:
        conv=convs[job['conversation_id']];q=next(q for q in conv['qa'] if q['id']==job['qa_id'])
        turns=reference_turns(q,conv['source_history'],gold[int(job['qa_id'])])
        rows.append(dict(qa_id=job['qa_id'],adaptive=adaptive,
            variants={name:coverage(job,value,turns,raw[job['t']['user_id']])
                      for name,value in (('original',original),('candidate',candidate))}))
    summary=dict(questions=len(rows),core_and_capacity_preserved=len(rows),provider_calls=0,
        eligible_questions=sum(bool(r['adaptive']['semantic_reserve']['eligible_ids']) for r in rows),
        used_questions=sum(r['adaptive']['semantic_reserve']['used'] for r in rows),variants={})
    for name in ('original','candidate'):
        summary['variants'][name]={stage:dict(
            reference_questions=sum(any(r['variants'][name][stage]['user']) for r in rows),
            user_turns=sum(sum(r['variants'][name][stage]['user']) for r in rows),
            changed_order=sum(r['variants'][name][stage]['ids']!=r['variants']['original'][stage]['ids'] for r in rows),
            gained=[r['qa_id'] for r in rows if any(r['variants'][name][stage]['user']) and not any(r['variants']['original'][stage]['user'])],
            lost=[r['qa_id'] for r in rows if any(r['variants']['original'][stage]['user']) and not any(r['variants'][name][stage]['user'])],
            fewer_turns=[r['qa_id'] for r in rows if sum(r['variants'][name][stage]['user'])<sum(r['variants']['original'][stage]['user'])]) for stage in ('fine','main')}
    with args.output.open('x') as stream:
        json.dump(dict(summary=summary,rows=rows,limitation=__doc__,transformations=replacements,
            inputs={str(p.resolve()):sha(p) for p in (Path(__file__),Path(fine_evidence.__file__),
                args.run/'server-search-debug.jsonl',args.run/'completed/data/results/personamem-v2-32k.jsonl',
                Path(manifest['data']),args.gold_csv)}),stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
