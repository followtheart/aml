"""Decode every question-only response and replay fixed-stage source transport.

Review batch membership is frozen. Main/overflow selections are reconstructed,
not a new Search packet or answer score. New positives can still be semantically
wrong; invalid-retention gains are not successful relevance recovery.
"""
import argparse
import ast
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path

from probe_question_only_reviews import load_jobs,records,sha
from audit_candidate_sources import selected,status
from app import recovery_spans as spans
import audit_https_funnel as transport
from audit_https_stages import reference_set_coverage


def main(args):
    root=args.run
    plan=json.loads((root/'plan.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if not (root/'exit-code').exists():raise ValueError('Diagnostic has not reached terminal state')
    if sha(root/'plan.json')!=summary['plan_sha256'] or sha(root/'live.jsonl')!=summary['responses_sha256']:
        raise ValueError('Diagnostic data changed')
    if sha(Path(__file__).with_name('probe_question_only_reviews.py'))!=plan['script_sha256'] or sha(Path(__file__).with_name('probe_candidate_sources.py'))!=plan['loader_sha256']:
        raise ValueError('Request producer changed')
    source=Path(plan['source_run'])
    if sha(source/'server-search-debug.jsonl')!=plan['source_trace_sha256']:raise ValueError('Historical trace changed')
    jobs=load_jobs(source);live=records(root/'live.jsonl')
    if [m for m,*_ in jobs]!=plan['jobs']:raise ValueError('Request metadata did not reconstruct')
    if len(live)!=plan['expected_calls'] or {(r['job_id'],r['arm']) for r in live}!={(i,a) for i in range(len(jobs)) for a in ('control','treatment')}:
        raise ValueError('Incomplete paired cohort')
    ledger=[dict(p.split('=',1) for p in line.split() if '=' in p)
        for line in (root/'provider.log').read_text().splitlines() if 'provider_call ' in line]
    if len(ledger)!=summary['calls'] or sum(int(e['total_tokens']) for e in ledger)!=summary['tokens'] or any(e['model']!='gpt-4o-mini' for e in ledger):
        raise ValueError('Provider ledger differs')
    decisions={a:{} for a in ('control','treatment')};applied={};decoded=0
    for r in live:
        m,before,after,built=jobs[r['job_id']]
        if any(r[k]!=m[k] for k in ('qa_id','stage','batch','applied')):raise ValueError('Response identity differs')
        request=after if r['arm']=='treatment' else before
        if request['reservation']!=r['request_bound']:raise ValueError('Request budget changed')
        matching=[e for e in ledger if e['stage']==f'diagnostic.question_only_review.{r["job_id"]}.{r["arm"]}']
        if len(matching)!=r['calls'] or sum(int(e['total_tokens']) for e in matching)!=r['tokens']:
            raise ValueError('Per-request provider accounting differs')
        if 'error' not in r:
            restored,judgments=spans.decode_partial(copy.deepcopy(r['payload']),built[r['arm']])
            if sorted(restored)!=r['selected_ids'] or judgments!=r['judgments']:raise ValueError('Decoded judgments differ')
            decoded+=1
        qstage=decisions[r['arm']].setdefault(r['qa_id'],{}).setdefault(r['stage'],{})
        by_id={j['candidate_id']:j for j in r.get('judgments',[])}
        for mid in m['candidate_ids']:
            if mid in qstage:raise ValueError('Candidate reviewed twice within stage')
            qstage[mid]=by_id.get(mid)
            applied[r['qa_id'],r['stage'],mid]=m['applied']
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:gold=list(csv.DictReader(f))
    raw={}
    for event in records(source/'server-memory-debug.jsonl'):
        raw.setdefault(event['user_id'],{}).update({transport.identity(s):s for s in event['source_messages']})
    traces={t['search_id']:t for t in records(source/'server-search-debug.jsonl')}
    stage_audit=json.loads((source/'funnel/stage-transport.json').read_text())
    if stage_audit['summary']['progress_snapshot']:raise ValueError('Historical source transport incomplete')
    stages={r['qa_id']:r for r in stage_audit['rows']}
    rows,changes=[],[]
    for r in records(source/'completed/data/results/personamem-v2-32k.jsonl'):
        q=r['qa_id'];t=traces[r['search_id']];s=stages[q]
        old={}
        reviews=[('main',b) for b in t['cascade']['listwise'].get('recovery',{}).get('batches',[])]
        reviews.append(('overflow',t['cascade'].get('overflow',{}).get('review',{})))
        for stage,b in reviews:
            js={j['candidate_id']:j for j in b.get('judgments',[])}
            old.setdefault(stage,{}).update({mid:js.get(mid) for mid in b.get('candidate_ids',[])})
        historical=selected(t,old)
        if historical['main']!=set(t['cascade']['listwise']['selected_ids']) or historical['overflow']!=set(t['cascade']['overflow']['selected_ids']):
            raise ValueError('Historical selection did not replay')
        snippet=gold[int(q)]['related_conversation_snippet']
        try:snippet=json.loads(snippet)
        except ValueError:snippet=ast.literal_eval(snippet)
        turns=[transport.norm(z['content']) for z in snippet if z['role']=='user']
        items={c['id']:c.get('_packet_item',{}) for c in t['ranked']}
        def coverage(ids):
            cards=[dict(z,id=f'{mid}:{i}',text=z.get('content','')) for mid in sorted(ids) for i,z in enumerate(items[mid].get('sources',[]))]
            cov=transport.turn_coverage(turns,cards,raw[t['user_id']])
            return dict(full_turns=sum(z['full'] for z in cov),turns=[z['full'] for z in cov],
                **reference_set_coverage(s['raw_turn_coverage'],cov))
        for stage,oldstage in (('main','listwise_selected'),('overflow','overflow_selected')):
            if any(s['visible'][oldstage][k]!=v for k,v in coverage(historical[stage]).items() if k!='turns'):
                raise ValueError('Historical source transport changed')
        arms={a:{stage:dict(coverage(ids),ids=sorted(ids)) for stage,ids in selected(t,decisions[a].get(q,{})).items()}
            for a in ('control','treatment')}
        rows.append(dict(qa_id=q,arms=arms))
        carriers=set().union(*(set(ids) for ids in s['carrier_ids'].values()))
        for stage in ('main','overflow'):
            a,b=[decisions[arm].get(q,{}).get(stage,{}) for arm in ('control','treatment')]
            if a.keys()!=b.keys():raise ValueError('Candidate sets differ across arms')
            for mid in a:
                if status(a[mid])!=status(b[mid]):
                    changes.append(dict(qa_id=q,stage=stage,candidate_id=mid,reference_carrier=mid in carriers,
                        applied=applied[q,stage,mid],control=status(a[mid]),treatment=status(b[mid]),
                        control_judgment=a[mid],treatment_judgment=b[mid]))
    if len(rows)!=56:raise ValueError('Incomplete stage reconstruction')
    def summarize_rows(rs):
        return {a:dict(requests=len([r for r in rs if r['arm']==a]),calls=sum(r['calls'] for r in rs if r['arm']==a),
            tokens=sum(r['tokens'] for r in rs if r['arm']==a),errors=sum('error' in r for r in rs if r['arm']==a),
            judgments=dict(Counter(status(j) for r in rs if r['arm']==a for j in r.get('judgments',[]))),
            validation_errors=dict(Counter(e for r in rs if r['arm']==a for j in r.get('judgments',[]) for e in j['validation_errors'])))
            for a in ('control','treatment')}
    final=dict(decoded_responses=decoded,historical_selections_verified=len(rows),provider_calls=len(ledger),
        recorded_tokens=summary['tokens'],provider_statuses=dict(Counter(e['status'] for e in ledger)),
        overall=summarize_rows(live),by_layout={name:summarize_rows([r for r in live if r['applied']==value])
            for name,value in (('changed',True),('unchanged_fallback',False))},
        fixed_selection_transport={a:{stage:dict(reference_questions=sum(r['arms'][a][stage]['full_turns']>0 for r in rows),
            full_turns=sum(r['arms'][a][stage]['full_turns'] for r in rows),
            complete_sets=sum(r['arms'][a][stage]['available_set_complete'] for r in rows)) for stage in ('main','overflow')}
            for a in ('control','treatment')},
        transport_changes={stage:dict(gained=[r['qa_id'] for r in rows if r['arms']['treatment'][stage]['full_turns']>r['arms']['control'][stage]['full_turns']],
            lost=[r['qa_id'] for r in rows if any(a and not b for a,b in zip(r['arms']['control'][stage]['turns'],r['arms']['treatment'][stage]['turns'],strict=True))])
            for stage in ('main','overflow')},
        reference_transitions=[{k:v for k,v in c.items() if not k.endswith('_judgment')} for c in changes if c['reference_carrier']])
    final['valid_positive_reference_recoveries']=[{k:v for k,v in c.items() if not k.endswith('_judgment')}
        for c in changes if c['reference_carrier'] and c['treatment']=='positive']
    final['invalid_retention_reference_gains']=[{k:v for k,v in c.items() if not k.endswith('_judgment')}
        for c in changes if c['reference_carrier'] and c['control']=='negative' and c['treatment']=='invalid']
    # Every positive addition remains available for semantic review, including
    # candidates unrelated to the benchmark's annotated reference snippet.
    additions=[c for c in changes if c['treatment']=='positive']
    final['positive_addition_count']=len(additions)
    output=dict(summary=final,positive_additions=additions,changes=changes,rows=rows,plan_sha256=sha(root/'plan.json'),responses_sha256=sha(root/'live.jsonl'),
        auditor_sha256=sha(Path(__file__)),limitation=__doc__)
    with args.output.open('x') as f:json.dump(output,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(final,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'))
    main(p.parse_args())
