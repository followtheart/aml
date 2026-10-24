"""Paired full-cohort diagnostic of the non-personal review context label.

Only the wire label and its short definition change. Original source payloads,
candidate ownership, attribution validation, and output limits remain intact.
Fixed-request retention is not final packet recall or answer accuracy. Historical
reviews share a Search deadline; these isolated calls have separate equal limits.
"""
import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, listwise_recovery as recovery, llm, recovery_spans as spans, schemas
from evaluation_model_guard import REQUIRED_MODEL, check_runtime, check_versions, check_provider_log

OLD = 'third_party_context'
NEW = 'text_context'
OLD_DEFINITION = OLD + ' for others/fictional context;'
NEW_DEFINITION = NEW + ' for quoted/supplied/fictional text;'


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def reservation(request):
    return (len(request['prompt'].encode()) + len(json.dumps(request['schema']).encode())
            + len(request['system'].encode()) + 1024 + request['max_tokens'])


def proposal(request):
    result = copy.deepcopy(request)
    prefix, data = result['prompt'].rsplit('\n', 1)
    if prefix.count(OLD_DEFINITION) != 1 or OLD in result['system']:
        raise ValueError('Requires original flat review instructions')
    result['prompt'] = prefix.replace(OLD_DEFINITION, NEW_DEFINITION) + '\n' + data
    for decision in result['schema']['properties'].values():
        kinds = decision['properties']['evidence_kind']['enum']
        if kinds != spans.KINDS:
            raise ValueError('Unexpected evidence-kind contract')
        decision['properties']['evidence_kind']['enum'] = [NEW if k == OLD else k for k in kinds]
    result['reservation'] = reservation(result)
    if (result['prompt'].rsplit('\n', 1)[1] != request['prompt'].rsplit('\n', 1)[1]
            or result['system'] != request['system'] or result['max_tokens'] != request['max_tokens']
            or result['reservation'] > request['reservation']
            or len(result['prompt'].encode()) > len(request['prompt'].encode())):
        raise ValueError('Source payload, limits, or size invariant failed')
    return result


def decode(payload, built, treatment):
    clean = copy.deepcopy(payload)
    if treatment and isinstance(clean, dict):
        for decision in clean.values():
            if not isinstance(decision, dict):
                continue
            kind = decision.get('evidence_kind')
            if kind == NEW:
                decision['evidence_kind'] = OLD
            elif kind == OLD:
                # An off-contract old label is invalid, never silently salvaged.
                decision['evidence_kind'] = '__invalid_wire_label__'
    return spans.decode_partial(clean, built)


def exception_chain(exc):
    names, seen = [], set()
    while exc is not None and id(exc) not in seen and len(names) < 5:
        seen.add(id(exc))
        names.append(type(exc).__name__)
        exc = exc.__cause__ or exc.__context__
    return names


def load_jobs(run):
    if (run/'exit-code').read_text().strip() != '0':
        raise ValueError('Require a completed successful arm')
    results = records(run/'completed/data/results/personamem-v2-32k.jsonl')
    summary = json.loads((run/'summary.json').read_text())
    if len(results) != summary['questions'] or summary['errors'] or any(r.get('error_stage') for r in results):
        raise ValueError('Incomplete source cohort')
    traces = {t['search_id']: t for t in records(run/'server-search-debug.jsonl')}
    jobs = []
    for result in results:
        t = traces[result['search_id']]
        check_versions(t['versions'], 'source trace')
        req = schemas.SearchRequest(**{k: t[k] for k in
            ('user_id', 'query', 'options', 'top_k', 'reference_time', 'evidence_token_budget')})
        reviews = [('main', i, v) for i, v in enumerate(t['cascade']['listwise'].get('recovery', {}).get('batches', []))]
        reviews.append(('overflow', 0, t['cascade'].get('overflow', {}).get('review', {})))
        with patch.multiple(config, **t['versions']['settings']):
            if not config.LISTWISE_SPAN_REFS or any((config.LISTWISE_ROLE_BOUND_REFS,
                    config.LISTWISE_TRANSFORM_CONTEXT, config.LISTWISE_TOPIC_INTEREST)):
                raise ValueError('Requires original flat span review')
            for stage, batch, review in reviews:
                cards = review.get('sources', [])
                if not cards or not review.get('prompt_sha256'):
                    continue
                before = recovery.request_for(req, cards)
                if (hashlib.sha256(before['prompt'].encode()).hexdigest() != review['prompt_sha256']
                        or before['reservation'] != review['request_bound']):
                    raise ValueError('Historical request did not reproduce')
                after = proposal(before)
                built = spans.build(req, cards, abbreviations=config.LISTWISE_ABBREVIATION_PARTS)
                reproduced = 0
                for judgment in review.get('judgments', []):
                    if 'span_reference_error' in judgment['validation_errors']:
                        continue
                    card = next(c for c in cards if c['candidate_id'] == judgment['candidate_id'])
                    _, checked = recovery.validate(dict(decisions=[
                        {k: judgment[k] for k in recovery.Decision.model_fields}]), [card], built['targets'])
                    if any(checked[0][k] != judgment[k] for k in ('valid', 'validation_errors')):
                        raise ValueError('Historical validator differs')
                    reproduced += 1
                meta = dict(job_id=len(jobs), qa_id=result['qa_id'], stage=stage, batch=batch,
                    candidates=len(cards), candidate_ids=[c['candidate_id'] for c in cards],
                    prompt_sha256=review['prompt_sha256'], original_request_sha256=digest(before),
                    treatment_request_sha256=digest(after), source_payload_sha256=digest(before['prompt'].rsplit('\n', 1)[1]),
                    reproduced_judgments=reproduced, reservation_delta=after['reservation']-before['reservation'],
                    prompt_bytes_delta=len(after['prompt'].encode())-len(before['prompt'].encode()),
                    original_time_limit_seconds=review['time_limit_seconds'])
                jobs.append((meta, before, after, built))
    return jobs


def summarize(jobs, rows):
    arms = {}
    for arm in ('control', 'treatment'):
        selected = [r for r in rows if r['arm'] == arm]
        judgments = [j for r in selected for j in r.get('judgments', [])]
        arms[arm] = dict(requests=len(selected), calls=sum(r['calls'] for r in selected),
            recorded_tokens=sum(r['tokens'] for r in selected), errors=dict(Counter(r['error'] for r in selected if 'error' in r)),
            valid_positive=sum(j['valid'] and j['useful'] for j in judgments),
            valid_negative=sum(j['valid'] and not j['useful'] for j in judgments),
            invalid=sum(not j['valid'] for j in judgments),
            validation_errors=dict(Counter(e for j in judgments for e in j['validation_errors'])),
            positive_kinds=dict(Counter(j['evidence_kind'] for j in judgments if j['valid'] and j['useful'])))
    return dict(reconstructed_requests=len(jobs), stages=dict(Counter(m['stage'] for m, *_ in jobs)),
        reproduced_judgments=sum(m['reproduced_judgments'] for m, *_ in jobs),
        reservation_deltas=sorted({m['reservation_delta'] for m, *_ in jobs}),
        prompt_bytes_deltas=sorted({m['prompt_bytes_delta'] for m, *_ in jobs}), arms=arms)


async def main(args):
    check_runtime()
    plan_path = args.output.with_suffix('.plan.json')
    live_path = args.output.with_suffix('.live.jsonl')
    log_path = args.output.with_suffix('.provider.log')
    if any(p.exists() for p in (args.output, plan_path, live_path, log_path)):
        raise FileExistsError('Diagnostic output already exists')
    jobs = load_jobs(args.run)
    if not jobs:
        raise ValueError('No review requests')
    # Pre-register every pair and arm order before any provider requests.
    schedule = [dict(job_id=m['job_id'], order=['control','treatment'] if m['job_id'] % 2 == 0
                     else ['treatment','control']) for m, *_ in jobs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan = dict(run=str(args.run.resolve()), model=REQUIRED_MODEL, live=args.live, concurrency=2,
        timeout_seconds=45, attempts=1, expected_calls=2*len(jobs) if args.live else 0,
        old_definition=OLD_DEFINITION, new_definition=NEW_DEFINITION,
        paired_requests=[m for m, *_ in jobs], schedule=schedule,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), limitation=__doc__)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    rows = []
    if args.live:
        if config.FAKE:
            raise ValueError('Real providers required')
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger = logging.getLogger('aml.metrics')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        semaphore = asyncio.Semaphore(2)

        async def pair(job, ordering):
            meta, before, after, built = job
            async with semaphore:
                for arm in ordering['order']:
                    request = copy.deepcopy(after if arm == 'treatment' else before)
                    bound = request.pop('reservation')
                    row = dict(job_id=meta['job_id'], qa_id=meta['qa_id'], stage=meta['stage'],
                               batch=meta['batch'], arm=arm, request_bound=bound)
                    start = time.monotonic()
                    with budget.scope(seconds=45, calls=1, tokens=bound) as limits:
                        try:
                            async with asyncio.timeout(45):
                                payload = await llm.complete_json(**request, attempts=1, timeout=45,
                                    stage=f'diagnostic.context_label.{meta["job_id"]}.{arm}')
                            row['payload'] = payload
                            restored, judgments = decode(payload, built, arm == 'treatment')
                            row.update(selected_ids=sorted(restored), judgments=judgments)
                        except Exception as exc:
                            row.update(error=type(exc).__name__, exception_chain=exception_chain(exc))
                    row.update(calls=limits.calls, tokens=limits.tokens, seconds=time.monotonic()-start)
                    rows.append(row)
                    with live_path.open('a') as stream:
                        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                    print(json.dumps(dict(completed=len(rows), planned=plan['expected_calls'],
                        **{k: row[k] for k in ('job_id','qa_id','stage','arm','calls','tokens')},
                        error=row.get('error'), positive=len(row.get('selected_ids', [])))), flush=True)
        try:
            await asyncio.gather(*(pair(job, order) for job, order in zip(jobs, schedule)))
        finally:
            logger.removeHandler(handler)
            handler.close()
    summary = summarize(jobs, rows)
    if rows:
        summary['provider_dispatches'] = check_provider_log(log_path.read_text(), 'diagnostic')
        if summary['provider_dispatches'] != sum(r['calls'] for r in rows):
            raise ValueError('Provider call accounting differs')
    args.output.write_text(json.dumps(dict(summary=summary, live=rows, limitation=__doc__),
                                      ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    if len(rows) != plan['expected_calls'] or any('error' in r for r in rows):
        raise RuntimeError('Diagnostic contains failures; original records retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except Exception as exc:
        print(json.dumps(dict(error=type(exc).__name__, exception_chain=exception_chain(exc))), flush=True)
        sys.exit(1)
