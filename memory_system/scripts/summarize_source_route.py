"""Verify source-route packet identities and summarize complete reader pairs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from replay_source_vector_route import source_route
from app import evidence_packet
from summarize_reader_replay import read_complete


def main(args):
    probe=json.loads(args.probe.read_text())
    actual_hash=hashlib.sha256(json.dumps([s['content'] for s in probe['sources']],ensure_ascii=False).encode()).hexdigest()
    if actual_hash!=probe['source_text_sha256']:
        raise ValueError('Source universe changed')
    questions={q['qa_id']:q for q in probe['questions']}
    if len(questions)!=len(probe['questions']):raise ValueError('Duplicate probe question')
    baseline={r['qa_id']:r for r in read_complete(args.run/'completed/data/results/personamem-v2-32k.jsonl')}
    traces={r['search_id']:r for r in read_complete(args.run/'server-search-debug.jsonl')}
    expected={}
    for qid,q in questions.items():
        r=baseline[qid];t=traces[q['search_id']]
        if (r['search_id']!=q['search_id'] or t['user_id']!=probe['user_id'] or t['query_specs']!=q['query_specs']
                or evidence_packet.digest(t['returned'])!=r['packet_hash']):
            raise ValueError('Probe/baseline identity mismatch')
        _,route=source_route(probe,q,t['returned'],t['evidence_token_budget'])
        expected[qid]=(r['packet_hash'],route)
    pairs=defaultdict(dict);focus=Counter();errors=[];calls=Counter()
    for r in read_complete(args.replay,args.allow_partial):
        qid=r['qa_id'];rep=r['repetition'];enabled=r['enabled']
        if qid not in expected or type(enabled) is not bool or not 1<=rep<=args.repeat:
            raise ValueError('Unexpected question/arm/repetition')
        old,route=expected[qid]
        if (r['baseline_packet_hash']!=old or r['packet_hash']!=(route['packet_hash'] if enabled else old)
                or r.get('route')!=(route if enabled else None)
                or bool(r.get('query_context',False))!=bool(enabled and args.query_context)):
            raise ValueError('Packet, route or setting mismatch')
        if enabled in pairs[qid,rep]:raise ValueError('Duplicate arm')
        pairs[qid,rep][enabled]=r
        arm='treatment' if enabled else 'control';d=r['diagnostics']
        calls[arm]+=len(d.get('answer_calls',[]))
        focus[arm+':'+d.get('answer_context_focus',{}).get('status','not_requested')]+=1
        if d.get('answer_status')=='error':errors.append(dict(qa_id=qid,arm=arm,error=d.get('error_type')))
    complete=[dict(qa_id=qid,repetition=rep,control=a[False]['score'],treatment=a[True]['score'],
                   control_prediction=a[False]['prediction'],treatment_prediction=a[True]['prediction'])
              for (qid,rep),a in pairs.items() if set(a)=={False,True}]
    count=sum(len(a) for a in pairs.values());partial=count!=2*len(questions)*args.repeat
    if partial and not args.allow_partial:raise ValueError('Incomplete paired experiment')
    report=dict(partial=partial,rows=count,expected_rows=2*len(questions)*args.repeat,complete_pairs=len(complete),
        control=sum(r['control'] for r in complete),treatment=sum(r['treatment'] for r in complete),
        wins=[r['qa_id'] for r in complete if r['treatment']>r['control']],
        losses=[r['qa_id'] for r in complete if r['treatment']<r['control']],
        errors=errors,answer_calls=dict(calls),focus_status=dict(focus),pairs=complete,
        caveat='Packet/query/source identities verified. Full reader live; vectors and query costs excluded. '
               'Sources prioritized before existing packet under the same byte cap; not fresh HTTPS.')
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs','caveat')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','probe','replay','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--query-context',action='store_true')
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--repeat',type=int,default=1,choices=(1,2,3))
    main(p.parse_args())
