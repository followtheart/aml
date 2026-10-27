"""Fixed review diagnostic: describe source subjects before judging usefulness.

The generated subject is a reading aid only, never evidence or a fact. It is
not passed to citation validation. No production selection rule is changed.
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

GUIDANCE = (' For EVERY candidate, including those you will exclude, first emit source_subject: '
            'a short literal description of the concrete subjects discussed in that candidate\'s '
            'own original sources, in at most 160 characters. Read all of its source parts. '
            'Describe the material, not the targets, and do not infer a biography or preference. '
            'Then emit the normal usefulness judgment and exact part references. '
            'The description is only a reading aid; it is not evidence, a quotation, or a personal fact. '
            'Keep the existing role, relevance, negation, authorship and hypothetical distinctions. '
            'An unrelated subject still requires the normal negative decision. '
            'Do not invent a connection merely because a description was requested.')


def proposal(request):
    result=copy.deepcopy(request)
    old_size=len(json.dumps(result['schema']).encode())
    for decision in result['schema']['properties'].values():
        if set(decision.get('properties',{}))!={'useful','target_id','evidence_kind','citation_parts'}:
            raise ValueError('Requires flat span reference schema')
        decision['properties']={'source_subject':dict(type='string',minLength=1,maxLength=160),**decision['properties']}
        decision['required']=['source_subject']+decision['required']
    result['system']+=GUIDANCE
    result['reservation']+=len(GUIDANCE.encode())+len(json.dumps(result['schema']).encode())-old_size
    return result


def decode(payload,built):
    """Remove only well-formed reading aids; invalid extra fields stay rejected."""
    if not isinstance(payload,dict):
        raise ValueError('Expected keyed decisions')
    clean=copy.deepcopy(payload);subjects={}
    for key,value in clean.items():
        if not isinstance(value,dict):
            continue
        subject=value.get('source_subject')
        if isinstance(subject,str) and subject.strip() and len(subject)<=160:
            subjects[key]=value.pop('source_subject')
        elif 'source_subject' not in value:
            # The existing decoder rejects this extra marker; missing assessment
            # must not silently turn a malformed treatment response into valid.
            value['source_subject']=None
    restored,judgments=spans.decode_partial(clean,built)
    return restored,judgments,subjects


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
                canonical=copy.deepcopy(after['schema'])
                for decision in canonical['properties'].values():
                    del decision['properties']['source_subject'];decision['required'].remove('source_subject')
                if canonical!=before['schema'] or before['prompt']!=after['prompt'] or before['max_tokens']!=after['max_tokens']:
                    raise ValueError('Evidence, original decision contract or output limit changed')
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
                                stage='diagnostic.review_subjects.'+('treatment' if enabled else 'control'))
                        if enabled:
                            restored,judgments,subjects=decode(payload,built)
                        else:
                            restored,judgments=spans.decode_partial(payload,built);subjects={}
                        row.update(selected_ids=sorted(restored),judgments=judgments,source_subjects=subjects,payload=payload)
                    except Exception as exc:
                        row.update(error=type(exc).__name__,detail=str(exc))
                row.update(calls=limits.calls,tokens=limits.tokens);live.append(row)
                with live_path.open('a') as out:out.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('judgments','source_subjects','payload','detail')}),flush=True)
    summary=dict(reproduced_requests=len(checks),source_prompts_unchanged=True,
        original_decision_contract_unchanged=True,reading_aid_is_not_evidence=True,
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
