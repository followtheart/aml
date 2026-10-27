"""Validate a cohort of frozen-first-selection paired review diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
from summarize_reader_replay import read_complete


def main(args):
    plan=json.loads((args.directory/'plan.json').read_text());source=Path(plan['input'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=plan['input_sha256']:raise ValueError('Archived input changed')
    expected={r['qa_id']:r for r in read_complete(source)}
    if set(expected)!=set(plan['questions']) or len(expected)!=len(plan['questions']):raise ValueError('Cohort identities changed')
    pairs=[];missing=[]
    for qid in plan['questions']:
        path=args.directory/(qid+'.jsonl')
        rows=read_complete(path,args.allow_partial) if path.exists() else []
        arms={}
        for r in rows:
            if plan.get('intervention') and r.get('intervention') != plan['intervention']:
                raise ValueError('Recorded intervention disagrees with cohort plan')
            if (r['qa_id']!=qid or r['repetition']!=1 or type(r['enabled']) is not bool
                    or r['packet_hash']!=expected[qid]['packet_hash'] or r.get('frozen_verified') is not True):
                raise ValueError('Question/arm/packet/preflight mismatch')
            if r['enabled'] in arms:raise ValueError('Duplicate arm')
            arms[r['enabled']]=r
        if set(arms)!={False,True}:missing.append(qid);continue
        a,b=arms[False],arms[True]
        pairs.append(dict(qa_id=qid,control=a['score'],treatment=b['score'],control_prediction=a['prediction'],
            treatment_prediction=b['prediction'],control_calls=len(a['live_calls']),treatment_calls=len(b['live_calls']),
            review_changed=[c['prompt'] for c in a['live_calls']]!=[c['prompt'] for c in b['live_calls']]))
    if missing and not args.allow_partial:raise ValueError('Incomplete questions: '+','.join(missing))
    report=dict(partial=bool(missing),complete_pairs=len(pairs),expected_pairs=len(expected),missing=missing,
        control=sum(r['control'] for r in pairs),treatment=sum(r['treatment'] for r in pairs),
        wins=[r['qa_id'] for r in pairs if r['treatment']>r['control']],
        losses=[r['qa_id'] for r in pairs if r['treatment']<r['control']],
        changed_reviews=[r['qa_id'] for r in pairs if r['review_changed']],
        live_calls=[sum(r[k] for r in pairs) for k in ('control_calls','treatment_calls')],pairs=pairs,
        caveat='Frozen initial answer and all earlier judgments; independent live review outcomes. Not fresh full-reader/HTTPS.')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('pairs','caveat')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true');main(p.parse_args())
