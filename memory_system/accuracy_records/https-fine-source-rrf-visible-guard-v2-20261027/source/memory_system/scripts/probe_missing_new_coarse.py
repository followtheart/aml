"""Only missing fine candidates may use new-coarse insurance, offline screening.

The baseline is insurance-off selection on the same actual CE scores, even when
the input trace is from an insurance-on arm. No original fine member is promoted
by the insurance seat. Whole-source selection and main budgets stay unchanged.
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
from app import fine_evidence
from probe_full_option_ce import prepare,base,coverage,sha
from evaluate_recall_context import reference_turns


def proposal():
    text=inspect.getsource(fine_evidence.propose)
    edits=[('new_coarse_insurance=False):','new_coarse_insurance=False, excluded_insurance_ids=()):'),
           ("if c['id'] not in allowed\n","if c['id'] not in allowed and c['id'] not in excluded_insurance_ids\n")]
    for old,new in edits:
        if text.count(old)!=1:raise ValueError('Inspect changed production selector')
        text=text.replace(old,new)
    scope=dict(vars(fine_evidence));exec(compile(text,'<missing_new_coarse>','exec'),scope)
    return scope['propose'],edits


def main(args):
    proposed,edits=proposal();fixed=[]
    with patch('socket.socket.connect',side_effect=AssertionError('Offline')):
        jobs,manifest,conversations=prepare(args.run)
        for job in jobs:
            scores=[c['_ce_score'] for c in job['coarse']]
            actual=base.replay(job,scores)
            cfg=job['t']['versions']['settings']
            assert cfg['COARSE_EVIDENCE_EXTEND'] and cfg['FINE_COARSE_CORE_QUOTA'] and not cfg['FINE_SEMANTIC_RESERVE']
            baseline_job=copy.deepcopy(job);baseline_job['t']['versions']['settings']['FINE_NEW_COARSE_INSURANCE']=False
            observed=[];original=fine_evidence.propose
            def capture(*a,**kw):
                chosen,trace=original(*a,**kw);observed.append(trace);return chosen,trace
            with patch.object(fine_evidence,'propose',capture):baseline=base.replay(baseline_job,scores)
            assert len(observed)==1
            baseline_trace=observed[0];excluded={c['id'] for c in baseline['fine']};observed=[]
            def missing_only(*a,**kw):
                chosen,trace=proposed(*a,**dict(kw,new_coarse_insurance=True,excluded_insurance_ids=excluded))
                observed.append(trace);return chosen,trace
            with patch.object(fine_evidence,'propose',missing_only):candidate=base.replay(baseline_job,scores)
            assert len(observed)==1
            adaptive=observed[0]['adaptive'];info=adaptive['new_coarse_insurance']
            assert adaptive['core_ids']==baseline_trace['adaptive']['core_ids']
            assert adaptive['capacity']==baseline_trace['adaptive']['capacity']
            assert not set(info['eligible_ids'])&excluded
            additions=[a for a in adaptive['additions'] if a['reason']=='new_coarse_core_insurance']
            assert len(additions)<=1 and info['used']==bool(additions)
            assert all(a['candidate_id'] not in excluded and len(a['ids'])==1 for a in additions)
            if not info['used']:assert baseline==candidate
            fixed.append((job,dict(actual=actual,baseline=baseline,candidate=candidate),adaptive))
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream:gold=list(csv.DictReader(stream))
    raw={}
    with sqlite3.connect((args.run/'memory.db').resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        for row in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(row['user_id'],{})[row['request_id'],row['message_index']]=dict(row)
    convs={c['conversation_id']:c for c in conversations};rows=[]
    for job,variants,adaptive in fixed:
        conv=convs[job['conversation_id']];q=next(q for q in conv['qa'] if q['id']==job['qa_id'])
        turns=reference_turns(q,conv['source_history'],gold[int(job['qa_id'])])
        rows.append(dict(qa_id=job['qa_id'],adaptive=adaptive,variants={name:coverage(job,value,turns,raw[job['t']['user_id']])
                                                                   for name,value in variants.items()}))
    summary=dict(questions=len(rows),actual_fine_and_main_reproduced=len(rows),core_and_capacity_preserved=len(rows),provider_calls=0,
        eligible_questions=sum(bool(r['adaptive']['new_coarse_insurance']['eligible_ids']) for r in rows),
        used_questions=sum(r['adaptive']['new_coarse_insurance']['used'] for r in rows),variants={})
    for name in ('actual','baseline','candidate'):
        summary['variants'][name]={stage:dict(
            reference_questions=sum(any(r['variants'][name][stage]['user']) for r in rows),
            user_turns=sum(sum(r['variants'][name][stage]['user']) for r in rows),
            changed_order=sum(r['variants'][name][stage]['ids']!=r['variants']['baseline'][stage]['ids'] for r in rows),
            gained=[r['qa_id'] for r in rows if any(r['variants'][name][stage]['user']) and not any(r['variants']['baseline'][stage]['user'])],
            lost=[r['qa_id'] for r in rows if any(r['variants']['baseline'][stage]['user']) and not any(r['variants'][name][stage]['user'])],
            individual_turn_losses=[r['qa_id'] for r in rows if any(b and not a for b,a in zip(
                r['variants']['baseline'][stage]['user'],r['variants'][name][stage]['user']))]) for stage in ('fine','main')}
    with args.output.open('x') as stream:
        json.dump(dict(summary=summary,rows=rows,limitation=__doc__,transformations=edits,
            inputs={str(p.resolve()):sha(p) for p in (Path(__file__),Path(fine_evidence.__file__),
                args.run/'server-search-debug.jsonl',args.run/'completed/data/results/personamem-v2-32k.jsonl',
                Path(manifest['data']),args.gold_csv)}),stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
