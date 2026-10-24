"""Paired recovery reviews with only the original question as relevance target.

Control retains all original options. Treatment omits options from this review
and its target enum, preserving every source, owner, validator and output field.
This probes whether option premises suppress question-relevant context. It is
not a live Search or reader score; option-only evidence may be lost.
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proposal(before, built):
    after = copy.deepcopy(before)
    prefix, raw = before['prompt'].rsplit('\n', 1)
    data = json.loads(raw)
    data['targets'] = {'question': data['targets']['question']}
    after['prompt'] = prefix + '\n' + json.dumps(data, ensure_ascii=False)
    for decision in after['schema']['properties'].values():
        if decision['properties']['target_id']['enum'] != [''] + list(built['targets']):
            raise ValueError('Unexpected target enumeration')
        decision['properties']['target_id']['enum'] = ['', 'question']
    after['reservation'] = reservation(after)
    changed = copy.deepcopy(built)
    changed.update(prompt=after['prompt'], schema=after['schema'], targets=data['targets'])
    # Invert the only two changes and verify every original field.
    restored = copy.deepcopy(after)
    data['targets'] = built['targets']
    restored['prompt'] = prefix + '\n' + json.dumps(data, ensure_ascii=False)
    for decision in restored['schema']['properties'].values():
        decision['properties']['target_id']['enum'] = [''] + list(built['targets'])
    restored['reservation'] = reservation(restored)
    if restored != before or after['reservation'] > before['reservation']:
        raise ValueError('Unexpected request or budget change')
    return after, changed


def load_jobs(run):
    jobs = []
    for meta, before, _, built in original_jobs(run):
        after, changed = proposal(before, built)
        fingerprint = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        job_meta = {k: meta[k] for k in ('job_id', 'qa_id', 'stage', 'batch', 'candidates',
            'candidate_ids', 'prompt_sha256', 'reproduced_judgments',
            'original_prompt_limit_bytes', 'original_time_limit_seconds')}
        job_meta.update(applied=after != before,
            original_request_sha256=fingerprint(before), treatment_request_sha256=fingerprint(after),
            actual_prompt_bytes_delta=len(after['prompt'].encode())-len(before['prompt'].encode()),
            actual_reservation_delta=after['reservation']-before['reservation'])
        jobs.append((job_meta, before, after, {'control': built, 'treatment': changed}))
    return jobs


async def main(args):
    check_runtime()
    if config.FAKE or config.CE_MODEL != 'qwen3-rerank':
        raise ValueError('Require fixed real models')
    jobs = load_jobs(args.run)
    if len({m['qa_id'] for m, *_ in jobs}) != 56:
        raise ValueError('Require all 56 questions')
    args.output.mkdir(parents=True, exist_ok=False)
    plan = dict(source_run=str(args.run.resolve()), model=config.LLM_MODEL, protocol=__doc__,
        script_sha256=sha(Path(__file__)), loader_sha256=sha(Path(__file__).with_name('probe_candidate_sources.py')),
        source_trace_sha256=sha(args.run/'server-search-debug.jsonl'), jobs=[m for m, *_ in jobs],
        questions=56, expected_calls=2*len(jobs) if args.live else 0, live=args.live,
        changed_pairs=sum(m['applied'] for m, *_ in jobs),
        fallback_pairs=sum(not m['applied'] for m, *_ in jobs),
        schedule=[dict(job_id=m['job_id'], order=['control', 'treatment'] if m['job_id']%2 == 0
            else ['treatment', 'control']) for m, *_ in jobs],
        advance_criterion='Require semantically relevant valid-positive reference recovery, improved overflow reference transport, and no main-stage reference loss. Invalid retention and Q47 alone do not qualify. Inspect all positive additions, not just reference carriers.',
        limitation='Frozen review membership, original request budget upper bound and independent 45-second single-attempt calls. Not a full Search or answer score. Omitting options can lose option-only evidence.')
    if args.live:
        preflight = json.loads((args.preflight/'plan.json').read_text())
        for key in ('source_run', 'script_sha256', 'loader_sha256', 'source_trace_sha256', 'jobs', 'schedule'):
            if plan[key] != preflight[key]:
                raise ValueError('Preflight mismatch: '+key)
        plan['preflight_plan_sha256'] = sha(args.preflight/'plan.json')
    (args.output/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: plan[k] for k in ('questions', 'expected_calls', 'changed_pairs')}), flush=True)
    rows = []
    if args.live:
        handler = logging.FileHandler(args.output/'provider.log', mode='x')
        logger = logging.getLogger('aml.metrics')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        semaphore = asyncio.Semaphore(2)

        async def pair(job, schedule):
            meta, before, after, built = job
            async with semaphore:
                for arm in schedule['order']:
                    request = copy.deepcopy(after if arm == 'treatment' else before)
                    bound = request.pop('reservation')
                    row = dict(job_id=meta['job_id'], qa_id=meta['qa_id'], stage=meta['stage'],
                        batch=meta['batch'], arm=arm, applied=meta['applied'], request_bound=bound)
                    start = time.monotonic()
                    with budget.scope(seconds=45, calls=1, tokens=bound) as usage:
                        try:
                            async with asyncio.timeout(45):
                                payload = await llm.complete_json(**request, attempts=1, timeout=45,
                                    stage=f'diagnostic.question_only_review.{meta["job_id"]}.{arm}')
                            row['payload'] = payload
                            restored, judgments = spans.decode_partial(payload, built[arm])
                            row.update(selected_ids=sorted(restored), judgments=judgments)
                        except Exception as exc:
                            row.update(error=type(exc).__name__, exception_chain=exception_chain(exc))
                    row.update(calls=usage.calls, tokens=usage.tokens, seconds=time.monotonic()-start)
                    rows.append(row)
                    with (args.output/'live.jsonl').open('a') as f:
                        f.write(json.dumps(row, ensure_ascii=False)+'\n')
                    print(json.dumps(dict(completed=len(rows), planned=plan['expected_calls'],
                        job_id=row['job_id'], arm=arm, error=row.get('error'))), flush=True)
        try:
            await asyncio.gather(*(pair(j, s) for j, s in zip(jobs, plan['schedule'], strict=True)))
        finally:
            logger.removeHandler(handler)
            handler.close()
    summary = dict(rows=len(rows), calls=sum(r['calls'] for r in rows), tokens=sum(r['tokens'] for r in rows),
        errors=sum('error' in r for r in rows), plan_sha256=sha(args.output/'plan.json'),
        responses_sha256=sha(args.output/'live.jsonl') if rows else None,
        arms={arm:dict(calls=sum(r['calls'] for r in rows if r['arm']==arm),
            tokens=sum(r['tokens'] for r in rows if r['arm']==arm),
            judgments=dict(Counter('invalid' if not j['valid'] else 'positive' if j['useful'] else 'negative'
                for r in rows if r['arm']==arm for j in r.get('judgments', [])))) for arm in ('control', 'treatment')})
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    complete = len(rows) == plan['expected_calls'] and not summary['errors']
    (args.output/'exit-code').write_text('0\n' if complete else '1\n')
    print(json.dumps(summary), flush=True)
    if not complete:
        raise RuntimeError('Diagnostic has original recorded failures')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if args.live and args.preflight is None:
        parser.error('--live requires --preflight')
    asyncio.run(main(args))
