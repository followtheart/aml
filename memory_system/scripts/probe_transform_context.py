"""Fixed review requests: system-only change, optional repeated provider pairs.

No gold/reference annotations are loaded. This is a diagnostic, not HTTPS or a
full-pipeline score. Historical candidates and sources remain immutable.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget,config,listwise_recovery as recovery,schemas


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def main(args):
    traces={t['search_id']:t for t in records(args.run/'server-search-debug.jsonl')}
    checks=[]
    live=[]
    for result in records(args.run/'completed/data/results/personamem-v2-32k.jsonl'):
        t=traces[result['search_id']]
        req=schemas.SearchRequest(**{k:t[k] for k in ('user_id','query','options','top_k','reference_time','evidence_token_budget')})
        reviews=[('main',r) for r in t['cascade']['listwise'].get('recovery',{}).get('batches',[])]
        reviews.append(('overflow',t['cascade'].get('overflow',{}).get('review',{})))
        with patch.multiple(config,**t['versions']['settings']):
            for stage,review in reviews:
                cards=review.get('sources',[])
                if not cards or not review.get('prompt_sha256'):
                    continue
                with patch.object(config,'LISTWISE_TRANSFORM_CONTEXT',False):
                    before=recovery.request_for(req,cards)
                with patch.object(config,'LISTWISE_TRANSFORM_CONTEXT',True):
                    after=recovery.request_for(req,cards)
                if hashlib.sha256(before['prompt'].encode()).hexdigest()!=review['prompt_sha256'] or before['reservation']!=review['request_bound']:
                    raise ValueError('Historical request did not reproduce')
                if before['prompt']!=after['prompt'] or before['schema']!=after['schema']:
                    raise ValueError('Source prompt or schema changed')
                checks.append(dict(qa_id=result['qa_id'],stage=stage,candidates=len(cards),
                    reservation_growth=after['reservation']-before['reservation']))
                if stage=='overflow' and result['qa_id']==args.live_qa:
                    if config.FAKE:
                        raise ValueError('Real provider required for live diagnostic')
                    by_id={c['id']:c for c in t['ranked']}
                    pending=[by_id[c['candidate_id']] for c in cards]
                    for repeat in range(args.repeats):
                        for enabled in ((False,True) if repeat%2==0 else (True,False)):
                            with patch.object(config,'LISTWISE_TRANSFORM_CONTEXT',enabled),budget.scope(seconds=45,calls=1,tokens=64000) as limits:
                                selected,trace=await recovery._recover_batch(req,pending,[],set(range(len(pending))),
                                    stage='diagnostic.transform_context.'+('treatment' if enabled else 'control'))
                            live.append(dict(qa_id=result['qa_id'],repeat=repeat,enabled=enabled,
                                selected_ids=[c['id'] for c in selected],trace=trace,calls=limits.calls,tokens=limits.tokens))
                            print(json.dumps({k:v for k,v in live[-1].items() if k!='trace'}),flush=True)
    summary=dict(reproduced_requests=len(checks),source_prompts_and_schemas_unchanged=True,
                 reservation_growths=sorted({r['reservation_growth'] for r in checks}),live_calls=len(live),
                 live_errors=sum(r['trace']['status']=='error' for r in live))
    with args.output.open('x') as out:
        json.dump(dict(summary=summary,checks=checks,live=live,limitation=__doc__),out,ensure_ascii=False,indent=2)
        out.write('\n')
    print(json.dumps(summary))
    if args.live_qa and (len(live)!=2*args.repeats or summary['live_errors']):
        raise RuntimeError('Live preflight incomplete or provider failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--live-qa')
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if args.repeats<1:
        p.error('Positive repeat count required')
    asyncio.run(main(args))
