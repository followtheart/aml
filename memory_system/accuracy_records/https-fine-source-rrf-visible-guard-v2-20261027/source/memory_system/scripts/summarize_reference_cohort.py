"""Validate paired source-reference replay artifacts and summarize complete pairs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main(args):
    plan=json.loads((args.directory/'plan.json').read_text())
    for name,expected in plan['scripts'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Probe script changed during cohort: '+name)
    pairs=[]
    pending=[]
    unresolved=Counter()
    repair_statuses=Counter()
    refs=Counter()
    for case in plan['cases']:
        qid=case['qa_id']
        file=args.directory/(qid+'.jsonl')
        lines=file.read_bytes().splitlines(keepends=True) if file.exists() else []
        rows=[json.loads(l) for l in lines if l.endswith(b'\n')]
        if len(rows)>2 or len({r['enabled'] for r in rows})!=len(rows):
            raise ValueError('Duplicate or excess arms: '+qid)
        baseline=next(json.loads(l) for l in Path('runs/https-cycle-20261006-'+case['run']+
            '/completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines()
            if json.loads(l)['qa_id']==qid)
        for row in rows:
            if (row['qa_id']!=qid or row['mode']!='quote-refs' or row['repetition']!=1
                    or not row['frozen_verified'] or row['packet_hash']!=baseline['packet_hash']):
                raise ValueError('Replay identity mismatch: '+qid)
            sources={s['id']:s for s in row['diagnostics']['answer_source_catalog']}
            for call in row['live_calls']:
                if 'decoded_response' not in call:
                    continue
                if not row['enabled']:
                    raise ValueError('Unexpected reference conversion in control')
                # Independently audit the complete registry against unchanged
                # original source text, not only the model-selected quote.
                for span in call['reference_registry'].values():
                    text=sources[span['source_id']]['text']
                    if text[span['start']:span['end']]!=span['quote']:
                        raise ValueError('Reference span differs from original source')
                raw=call['response']['options'];decoded=call['decoded_response']['options']
                raw=list(raw.values()) if isinstance(raw,dict) else raw
                decoded=list(decoded.values()) if isinstance(decoded,dict) else decoded
                if len(raw)!=len(decoded):raise ValueError('Option count changed during decoding')
                for before,after in zip(raw,decoded):
                    if len(before['claims'])!=len(after['claims']):raise ValueError('Claims changed during decoding')
                    for bc,ac in zip(before['claims'],after['claims']):
                        if {k:v for k,v in bc.items() if k!='citations'}!={k:v for k,v in ac.items() if k!='citations'}:
                            raise ValueError('Claim status/text changed during decoding')
                        if len(bc['citations'])!=len(ac['citations']):raise ValueError('Citations changed during decoding')
                        for b,a in zip(bc['citations'],ac['citations']):
                            span=call['reference_registry'][b['span_id']]
                            if (span['source_id']!=b['source_id'] or a!={k:v for k,v in b.items() if k!='span_id'}|{'quote':span['quote']}):
                                raise ValueError('Citation identity changed during decoding')
                            refs['decoded_citations']+=1
                refs['decoded_calls']+=1
        if len(rows)!=2:
            pending.append(qid)
            continue
        by_arm={r['enabled']:r for r in rows}
        pairs.append(dict(qa_id=qid,control=by_arm[False]['score'],treatment=by_arm[True]['score'],
                          control_prediction=by_arm[False]['prediction'],treatment_prediction=by_arm[True]['prediction']))
        for enabled,row in by_arm.items():
            arm='treatment' if enabled else 'control'
            unresolved[arm]+=len(row['diagnostics'].get('answer_support_validation',{}).get('unresolved_options',[]))
            for call in row['diagnostics']['answer_calls']:
                if call['stage']=='eval.choice_support.repair':repair_statuses[arm+':'+call['status']]+=1
    report=dict(partial=bool(pending),expected_pairs=len(plan['cases']),complete_pairs=len(pairs),
        control_score=sum(p['control'] for p in pairs),treatment_score=sum(p['treatment'] for p in pairs),
        wins=[p['qa_id'] for p in pairs if p['treatment']>p['control']],
        losses=[p['qa_id'] for p in pairs if p['treatment']<p['control']],
        pending=pending,pairs=pairs,unresolved_options=dict(unresolved),repair_statuses=dict(repair_statuses),
        reference_audit=dict(refs),script_hashes_verified=True,
        caveat='Scores/status counts use complete pairs; reference audit includes completed rows of pending pairs. '
        'Selected quote-failure cohort, frozen upstream and expanded diagnostic budget. Literal validity is not entailment.')
    if pending and not args.allow_partial:raise ValueError('Incomplete cohort')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='pairs'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    main(p.parse_args())
