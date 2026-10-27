"""Reproduce historical reviews and test exact abbreviation-aware source parts.

Live calls use fixed cards and do not measure whole-pipeline admission or recall.
Historical quotes are never enlarged to obtain a more favorable validation.
"""
import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, listwise_recovery as recovery, llm, recovery_spans as spans, schemas


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_parts(before, after):
    if before['candidate_map'] != after['candidate_map'] or before['cards'] != after['cards']:
        raise ValueError('Source ownership changed')
    if before['system'] != after['system'] or before['targets'] != after['targets']:
        raise ValueError('Instructions or targets changed')
    changed = []
    for pid, part in after['spans'].items():
        old = before['spans'][pid]
        if old['start'] != part['start'] or old['global_source'] != part['global_source']:
            raise ValueError('Stable part ID changed starting position or source')
        covered = [p for p in before['spans'].values() if p['global_source'] == part['global_source']
                   and p['start'] >= part['start'] and p['end'] <= part['end']]
        if not covered or ''.join(p['quote'] for p in covered) != part['quote']:
            raise ValueError('New part is not a complete exact union of old parts')
        if old != part:
            changed.append(dict(part_id=pid, original=old['quote'], merged=part['quote']))
    for sid in {p['global_source'] for p in before['spans'].values()}:
        raw = lambda built: ''.join(p['quote'] for p in built['spans'].values() if p['global_source'] == sid)
        if raw(before) != raw(after):
            raise ValueError('Original text lost, duplicated, reordered or normalized')
    return changed


async def main(args):
    plan_path = args.output.with_suffix('.plan.json')
    live_path = args.output.with_suffix('.live.jsonl')
    if any(p.exists() for p in (args.output, plan_path, live_path)):
        raise FileExistsError(args.output)
    traces = {t['search_id']: t for t in records(args.run/'server-search-debug.jsonl')}
    checks, jobs, live = [], [], []
    qas = set(args.live_qas.split(',')) if args.live_qas else set()
    for r in records(args.run/'completed/data/results/personamem-v2-32k.jsonl'):
        t = traces[r['search_id']]
        req = schemas.SearchRequest(**{k: t[k] for k in
            ('user_id','query','options','top_k','reference_time','evidence_token_budget')})
        reviews = [('main',i,v) for i,v in enumerate(t['cascade']['listwise'].get('recovery',{}).get('batches',[]))]
        reviews.append(('overflow',0,t['cascade'].get('overflow',{}).get('review',{})))
        with patch.multiple(config, **t['versions']['settings']):
            if not config.LISTWISE_SPAN_REFS or config.LISTWISE_ABBREVIATION_PARTS:
                raise ValueError('Requires baseline span-reference run')
            for stage, batch, review in reviews:
                cards = review.get('sources',[])
                if not cards or not review.get('prompt_sha256'):
                    continue
                before = recovery.request_for(req,cards)
                if (hashlib.sha256(before['prompt'].encode()).hexdigest()!=review['prompt_sha256']
                        or before['reservation']!=review['request_bound']):
                    raise ValueError('Historical request did not reproduce')
                with patch.object(config,'LISTWISE_ABBREVIATION_PARTS',True):
                    after = recovery.request_for(req,cards)
                opts = dict(role_bound=config.LISTWISE_ROLE_BOUND_REFS, transform_context=config.LISTWISE_TRANSFORM_CONTEXT)
                old = spans.build(req,cards,**opts)
                new = spans.build(req,cards,abbreviations=True,**opts)
                changes = check_parts(old,new)
                if after['reservation']>before['reservation']:
                    raise ValueError('Merging parts increased reservation')
                reproduced, affected = 0, []
                for judgment in review.get('judgments',[]):
                    if 'span_reference_error' in judgment['validation_errors']:
                        continue
                    card = next(c for c in cards if c['candidate_id']==judgment['candidate_id'])
                    decision = {k:judgment[k] for k in recovery.Decision.model_fields}
                    _, validated = recovery.validate(dict(decisions=[decision]),[card],old['targets'])
                    if any(validated[0][k]!=judgment[k] for k in ('valid','validation_errors')):
                        raise ValueError('Historical judgment did not reproduce')
                    reproduced+=1
                    key = next(k for k,v in new['candidate_map'].items() if v==judgment['candidate_id'])
                    if any(not any(owner==ref['source_id'] and new['spans'][pid]['quote'].strip()==ref['quote']
                                       for pid,owner in new['owners'][key].items()) for ref in judgment['citations']):
                        affected.append(judgment)
                meta = dict(qa_id=r['qa_id'],stage=stage,batch=batch,candidates=len(cards),
                    changed_parts=changes, affected_historical_judgments=affected,
                    reproduced_judgments=reproduced, reservation_delta=after['reservation']-before['reservation'],
                    prompt_delta=len(after['prompt'].encode())-len(before['prompt'].encode()),
                    baseline_prompt_sha256=review['prompt_sha256'])
                checks.append(meta)
                if r['qa_id'] in qas and changes:
                    jobs.append((meta,before,after,old,new))
    if qas!={m['qa_id'] for m,*_ in jobs}:
        raise ValueError('Not all requested questions have affected review requests')
    plan = dict(run=str(args.run), repeats=args.repeats, expected_calls=len(jobs)*args.repeats*2,
                paired_requests=[m for m,*_ in jobs],checks=checks,limitation=__doc__)
    plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    if jobs and config.FAKE:
        raise ValueError('Real providers required')
    for meta,before,after,old,new in jobs:
        for repeat in range(args.repeats):
            for enabled in ((False,True) if repeat%2==0 else (True,False)):
                request=copy.deepcopy(after if enabled else before)
                reservation=request.pop('reservation')
                row={k:meta[k] for k in ('qa_id','stage','batch')}
                row.update(enabled=enabled,repeat=repeat,request_bound=reservation)
                with budget.scope(seconds=45,calls=1,tokens=reservation) as limits:
                    try:
                        async with asyncio.timeout(45):
                            payload=await llm.complete_json(**request,attempts=1,timeout=45,
                                stage='diagnostic.abbreviation_parts.'+('treatment' if enabled else 'control'))
                        restored,judgments=spans.decode_partial(payload,new if enabled else old)
                        row.update(selected_ids=sorted(restored),judgments=judgments,payload=payload)
                    except Exception as exc:
                        row.update(error=type(exc).__name__,detail=str(exc))
                row.update(calls=limits.calls,tokens=limits.tokens)
                live.append(row)
                with live_path.open('a') as out:
                    out.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('judgments','payload','detail')}),flush=True)
    summary=dict(reproduced_requests=len(checks),reproduced_judgments=sum(c['reproduced_judgments'] for c in checks),
        changed_requests=sum(bool(c['changed_parts']) for c in checks),
        changed_questions=sorted({c['qa_id'] for c in checks if c['changed_parts']},key=int),
        changed_parts=sum(len(c['changed_parts']) for c in checks),
        affected_historical_judgments=sum(len(c['affected_historical_judgments']) for c in checks),
        reservation_deltas=dict(Counter(c['reservation_delta'] for c in checks)),
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
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if args.repeats<1:p.error('Positive repeats required')
    asyncio.run(main(args))
