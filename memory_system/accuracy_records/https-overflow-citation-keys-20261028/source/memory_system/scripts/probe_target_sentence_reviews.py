"""Paired whole-cohort deletion reviews with lossless target sentence layout.

Only the display of question/option targets changes. Source cards, source part
IDs, ownership, output schema, target IDs, validator and output limit are fixed.
Each target remains one complete target; its consecutive parts preserve every
character and qualification. Oversized transformed prompts fall back exactly.
Frozen membership and independent request budgets are not a live Search trial.
"""
import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
import logging
from pathlib import Path
import time

from probe_candidate_sources import load_jobs as original_jobs, records, reservation, exception_chain
from app import budget, config, llm, recovery_spans as spans
from evaluation_model_guard import check_runtime

NOTE = ('Targets below are lists of consecutive exact text parts. Each list is ONE complete target, '
        'not separate targets. Read all its parts together, including conditions and qualifications. '
        'Useful evidence can concern one concrete subject within a target without establishing all '
        'of its statements. This does not establish any personal premise. Return only the original '
        'target ID from the schema, never a part index.\n')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def proposal(before, cap):
    result=copy.deepcopy(before)
    prefix,raw=before['prompt'].rsplit('\n',1)
    payload=json.loads(raw)
    if set(payload)!= {'targets','sources','candidates'}:raise ValueError('Unexpected review format')
    original=copy.deepcopy(payload['targets'])
    payload['targets']={key:[text[start:end] for _,start,end in spans.source_parts(text,abbreviations=True)]
        for key,text in original.items()}
    if {key:''.join(parts) for key,parts in payload['targets'].items()}!=original:
        raise ValueError('Target characters or order changed')
    result['prompt']=prefix+'\n'+NOTE+json.dumps(payload,ensure_ascii=False)
    result['reservation']=reservation(result)
    if any(result[k]!=before[k] for k in ('schema','system','max_tokens')):
        raise ValueError('Provider output contract changed')
    applied=len(result['prompt'].encode())<=cap
    return (result if applied else copy.deepcopy(before)),dict(applied=applied,
        raw_prompt_bytes_delta=len(result['prompt'].encode())-len(before['prompt'].encode()),
        raw_reservation_delta=result['reservation']-before['reservation'],
        target_parts={key:len(parts) for key,parts in payload['targets'].items()})


def load_jobs(run):
    # Existing loader independently checks every historical prompt hash,
    # reservation and valid/invalid judgment before any live requests.
    jobs=[]
    for meta,before,_,built in original_jobs(run):
        after,details=proposal(before,meta['original_prompt_limit_bytes'])
        job_meta={k:meta[k] for k in ('job_id','qa_id','stage','batch','candidates','candidate_ids',
            'prompt_sha256','reproduced_judgments','original_prompt_limit_bytes','original_time_limit_seconds')}
        job_meta.update(details,
            original_request_sha256=hashlib.sha256(json.dumps(before,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
            treatment_request_sha256=hashlib.sha256(json.dumps(after,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
            actual_prompt_bytes_delta=len(after['prompt'].encode())-len(before['prompt'].encode()),
            actual_reservation_delta=after['reservation']-before['reservation'])
        for req in (before,after):
            data=json.loads(req['prompt'].rsplit('\n',1)[1])
            original=json.loads(before['prompt'].rsplit('\n',1)[1])
            if any(data[key]!=original[key] for key in ('sources','candidates')):
                raise ValueError('Source, ownership or candidate text changed')
            target={k:''.join(v) if isinstance(v,list) else v for k,v in data['targets'].items()}
            if target!=built['targets']:raise ValueError('Full target reconstruction differs')
        jobs.append((job_meta,before,after,built))
    return jobs


async def main(args):
    check_runtime()
    if config.FAKE or config.CE_MODEL!='qwen3-rerank':raise ValueError('Require real unchanged models')
    jobs=load_jobs(args.run)
    if len({m['qa_id'] for m,*_ in jobs})!=56:raise ValueError('Require whole 56-question review cohort')
    args.output.mkdir(parents=True,exist_ok=False)
    plan=dict(source_run=str(args.run.resolve()),model=config.LLM_MODEL,protocol=__doc__,note=NOTE,
        script_sha256=sha(Path(__file__)),loader_sha256=sha(Path(__file__).with_name('probe_candidate_sources.py')),
        source_trace_sha256=sha(args.run/'server-search-debug.jsonl'),jobs=[m for m,*_ in jobs],
        questions=56,expected_calls=2*len(jobs) if args.live else 0,live=args.live,
        changed_pairs=sum(m['applied'] for m,*_ in jobs),fallback_pairs=sum(not m['applied'] for m,*_ in jobs),
        schedule=[dict(job_id=m['job_id'],order=['control','treatment'] if m['job_id']%2==0 else ['treatment','control']) for m,*_ in jobs],
        advance_criterion='Require improved reference questions or complete turns after overflow, no main transport loss, and a genuine valid positive recovery. Invalid/unreviewed retention or Q47 alone does not suffice. Inspect every loss and provider failure before HTTPS.',
        limitation='Frozen batch membership and independent 45-second budgets; historical latency/token headroom not simulated. No end-to-end Search or reader accuracy claim.')
    (args.output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:plan[k] for k in ('questions','expected_calls','changed_pairs','fallback_pairs')}),flush=True)
    rows=[]
    if args.live:
        handler=logging.FileHandler(args.output/'provider.log',mode='x')
        logger=logging.getLogger('aml.metrics');logger.addHandler(handler);logger.setLevel(logging.INFO)
        semaphore=asyncio.Semaphore(2)
        async def pair(job,schedule):
            meta,before,after,built=job
            async with semaphore:
                for arm in schedule['order']:
                    request=copy.deepcopy(after if arm=='treatment' else before);bound=request.pop('reservation')
                    row=dict(job_id=meta['job_id'],qa_id=meta['qa_id'],stage=meta['stage'],batch=meta['batch'],
                        arm=arm,applied=meta['applied'],request_bound=bound)
                    start=time.monotonic()
                    with budget.scope(seconds=45,calls=1,tokens=bound) as usage:
                        try:
                            async with asyncio.timeout(45):
                                payload=await llm.complete_json(**request,attempts=1,timeout=45,
                                    stage=f'diagnostic.target_sentences.{meta["job_id"]}.{arm}')
                            row['payload']=payload
                            restored,judgments=spans.decode_partial(payload,built)
                            row.update(selected_ids=sorted(restored),judgments=judgments)
                        except Exception as exc:
                            row.update(error=type(exc).__name__,exception_chain=exception_chain(exc))
                    row.update(calls=usage.calls,tokens=usage.tokens,seconds=time.monotonic()-start)
                    rows.append(row)
                    with (args.output/'live.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    print(json.dumps(dict(completed=len(rows),planned=plan['expected_calls'],job_id=row['job_id'],
                        qa_id=row['qa_id'],arm=arm,error=row.get('error'),positive=len(row.get('selected_ids',[])))),flush=True)
        try:await asyncio.gather(*(pair(j,s) for j,s in zip(jobs,plan['schedule'],strict=True)))
        finally:logger.removeHandler(handler);handler.close()
    summary=dict(rows=len(rows),calls=sum(r['calls'] for r in rows),tokens=sum(r['tokens'] for r in rows),
        errors=sum('error' in r for r in rows),plan_sha256=sha(args.output/'plan.json'),
        responses_sha256=sha(args.output/'live.jsonl') if rows else None,
        arms={arm:dict(calls=sum(r['calls'] for r in rows if r['arm']==arm),
            tokens=sum(r['tokens'] for r in rows if r['arm']==arm),
            judgments=dict(Counter('invalid' if not j['valid'] else 'positive' if j['useful'] else 'negative'
                for r in rows if r['arm']==arm for j in r.get('judgments',[])))) for arm in ('control','treatment')})
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    complete=len(rows)==plan['expected_calls'] and not summary['errors']
    (args.output/'exit-code').write_text('0\n' if complete else '1\n')
    print(json.dumps(summary),flush=True)
    if not complete:raise RuntimeError('Diagnostic contains original recorded failures')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--live',action='store_true')
    asyncio.run(main(p.parse_args()))
