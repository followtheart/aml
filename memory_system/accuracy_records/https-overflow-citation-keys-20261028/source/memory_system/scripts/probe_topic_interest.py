"""Fixed review diagnostic: distinguish topical questions from personal premises.

Source prompts, schemas and validators are unchanged. Only a system instruction
clarifies when an original user question is useful topical context. No gold is
included in model requests and no production policy is changed by this script.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget,config,listwise_recovery as recovery,llm,recovery_spans as spans,schemas
from evaluation_model_guard import REQUIRED_MODEL, check_runtime, check_versions

GUIDANCE = ' A user question can be useful as topical context even when it does not prove a personal premise in an answer option. Evaluate these separately: does the user ask about a specific subject that is directly relevant to the current question or a proposed activity, material, or genre? If so, retain that original question as topic_interest and cite its exact meaningful parts. Lack of evidence for ownership, expertise, repeated behavior, preference, or another personal premise is not by itself a reason to discard this topical question. The topic_interest label records only that the user asked about the subject, never that the premise is true. Do not require the source to support every clause of a target. Shared generic words, unrelated curiosity, assistant-only explanations, or questions attributed to someone else do not qualify. Preserve the existing role, attribution, negation and hypothetical checks; use a negative judgment when no concrete topical connection or other supported use exists.'


def proposal(request):
    result=copy.deepcopy(request)
    result['system']+=GUIDANCE
    result['reservation']+=len(GUIDANCE.encode())
    return result


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def main(args):
    check_runtime()
    plan_path=args.output.with_suffix('.plan.json');live_path=args.output.with_suffix('.live.jsonl')
    if any(p.exists() for p in (args.output,plan_path,live_path)):
        raise FileExistsError(args.output)
    if (args.run/'exit-code').read_text().strip()!='0':
        raise ValueError('Use a completed successful arm')
    traces={t['search_id']:t for t in records(args.run/'server-search-debug.jsonl')}
    checks=[];jobs=[];live=[]
    qas=set(args.live_qas.split(',')) if args.live_qas else set()
    for r in records(args.run/'completed/data/results/personamem-v2-32k.jsonl'):
        t=traces[r['search_id']]
        check_versions(t['versions'], 'source trace')
        req=schemas.SearchRequest(**{k:t[k] for k in ('user_id','query','options','top_k','reference_time','evidence_token_budget')})
        reviews=[('main',i,v) for i,v in enumerate(t['cascade']['listwise'].get('recovery',{}).get('batches',[]))]
        reviews.append(('overflow',0,t['cascade'].get('overflow',{}).get('review',{})))
        with patch.multiple(config,**t['versions']['settings']):
            check_runtime()
            if not config.LISTWISE_SPAN_REFS or config.LISTWISE_ROLE_BOUND_REFS:
                raise ValueError('Requires flat span reference schema')
            for stage,batch,v in reviews:
                cards=v.get('sources',[])
                if not cards or not v.get('prompt_sha256'):continue
                before=recovery.request_for(req,cards);after=proposal(before)
                if (hashlib.sha256(before['prompt'].encode()).hexdigest()!=v['prompt_sha256']
                        or before['reservation']!=v['request_bound']):
                    raise ValueError('Historical request did not reproduce')
                canonical=copy.deepcopy(after)
                canonical['system']=canonical['system'][:-len(GUIDANCE)]
                canonical['reservation']-=len(GUIDANCE.encode())
                if canonical!=before:
                    raise ValueError('More than the system guidance changed')
                built=spans.build(req,cards,abbreviations=config.LISTWISE_ABBREVIATION_PARTS,
                                  transform_context=config.LISTWISE_TRANSFORM_CONTEXT)
                meta=dict(qa_id=r['qa_id'],stage=stage,batch=batch,candidates=len(cards),
                    prompt_sha256=v['prompt_sha256'],reservation_growth=after['reservation']-before['reservation'])
                checks.append(meta)
                if r['qa_id'] in qas and stage in args.stages.split(','):
                    jobs.append((meta,before,after,built))
    if qas!={m['qa_id'] for m,*_ in jobs}:
        raise ValueError('Not all selected questions have reviews')
    plan=dict(run=str(args.run),model=REQUIRED_MODEL,guidance=GUIDANCE,repeats=args.repeats,expected_calls=len(jobs)*args.repeats*2,
              paired_requests=[m for m,*_ in jobs],checks=checks,limitation=__doc__)
    plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    if jobs and config.FAKE:raise ValueError('Real providers required')
    for meta,before,after,built in jobs:
        for repeat in range(args.repeats):
            for enabled in ((False,True) if repeat%2==0 else (True,False)):
                request=copy.deepcopy(after if enabled else before);reservation=request.pop('reservation')
                row=dict(**meta,enabled=enabled,repeat=repeat,request_bound=reservation)
                with budget.scope(seconds=45,calls=1,tokens=reservation) as limits:
                    try:
                        async with asyncio.timeout(45):
                            payload=await llm.complete_json(**request,attempts=1,timeout=45,
                                stage='diagnostic.topic_interest.'+('treatment' if enabled else 'control'))
                        restored,judgments=spans.decode_partial(payload,built)
                        row.update(selected_ids=sorted(restored),judgments=judgments,payload=payload)
                    except Exception as exc:
                        row.update(error=type(exc).__name__,detail=str(exc))
                row.update(calls=limits.calls,tokens=limits.tokens);live.append(row)
                with live_path.open('a') as out:out.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('judgments','payload','detail')}),flush=True)
    summary=dict(reproduced_requests=len(checks),source_data_unchanged=True,
        original_decision_contract_unchanged=True,only_system_guidance_changed=True,
        reservation_growths=sorted({r['reservation_growth'] for r in checks}),
        live_calls=sum(r['calls'] for r in live),live_errors=sum('error' in r for r in live))
    args.output.write_text(json.dumps(dict(summary=summary,checks=checks,live=live,limitation=__doc__),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if len(live)!=plan['expected_calls'] or summary['live_errors']:
        raise RuntimeError('Live diagnostic incomplete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--live-qas',default='')
    p.add_argument('--stages',choices=['main','overflow','main,overflow'],default='main')
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if args.repeats<1:p.error('Positive repeats required')
    asyncio.run(main(args))
