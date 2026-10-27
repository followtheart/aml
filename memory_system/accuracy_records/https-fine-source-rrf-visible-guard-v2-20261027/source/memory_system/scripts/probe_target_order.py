"""Factorial diagnostic of question/options presentation and target enum order.

All target text, source data, output fields and citation validation stay the
same. Only the order of existing target entries changes. No production edits.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path

from probe_review_presentation import (
    budget, check_runtime, config, fingerprint, load_batches, llm, payload,
    request_for, REQUIRED_MODEL, spans,
)


def reorder(built, *, prompt_options_first=False, enum_options_first=False):
    result = copy.deepcopy(built)
    prefix, data = payload(result)
    keys = list(data['targets'])
    if not keys or keys[0] != 'question':
        raise ValueError('Require original question-first target order')
    reordered = keys[1:] + keys[:1]
    if prompt_options_first:
        data['targets'] = {key: data['targets'][key] for key in reordered}
    if enum_options_first:
        for decision in result['schema']['properties'].values():
            field = decision['properties']['target_id']
            if field['enum'] != [''] + keys:
                raise ValueError('Unexpected target schema')
            field['enum'] = [''] + reordered
    result['prompt'] = prefix + json.dumps(data, ensure_ascii=False)
    return result


async def main(args):
    check_runtime()
    paths = [args.output, args.output.with_suffix('.plan.json'), args.output.with_suffix('.live.jsonl')]
    if any(path.exists() for path in paths):
        raise FileExistsError(args.output)
    jobs = []
    for qa_id in args.qas.split(','):
        _, batches = load_batches(args.run, qa_id)
        if args.first_batch_only:
            batches = batches[:1]
        if not batches:
            raise ValueError('Question has no main reviews: ' + qa_id)
        for batch, built in batches:
            baseline = request_for(built)
            for prompt_first, enum_first in ((False, False), (True, False), (False, True), (True, True)):
                transformed = reorder(built, prompt_options_first=prompt_first, enum_options_first=enum_first)
                request = request_for(transformed)
                if request['reservation'] != baseline['reservation']:
                    raise ValueError('An order-only change changed the byte reservation')
                original_payload, changed_payload = payload(built)[1], payload(transformed)[1]
                if original_payload != changed_payload or transformed['owners'] != built['owners']:
                    raise ValueError('Targets, sources, candidates or ownership changed')
                canonical = copy.deepcopy(transformed['schema'])
                for decision in canonical['properties'].values():
                    decision['properties']['target_id']['enum'] = [''] + list(built['targets'])
                if canonical != built['schema']:
                    raise ValueError('Schema changed beyond target enum ordering')
                meta = dict(qa_id=qa_id, batch=batch, prompt_options_first=prompt_first,
                            enum_options_first=enum_first, request_sha256=fingerprint(request),
                            request_bound=request['reservation'])
                jobs.append((meta, request, transformed))
    plan = dict(model=REQUIRED_MODEL, run=str(args.run), qas=args.qas, repeats=args.repeats,
                first_batch_only=args.first_batch_only, expected_calls=len(jobs)*args.repeats if args.live else 0,
                jobs=[meta for meta, _, _ in jobs], limitation=__doc__)
    paths[1].write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    rows = []
    if args.live:
        if config.FAKE:
            raise ValueError('Real provider required')
        for repeat in range(args.repeats):
            for meta, original, built in (jobs if repeat % 2 == 0 else list(reversed(jobs))):
                check_runtime()
                request = copy.deepcopy(original)
                reservation = request.pop('reservation')
                row = dict(meta, repeat=repeat, model=REQUIRED_MODEL)
                with budget.scope(seconds=45, calls=1, tokens=reservation) as limits:
                    try:
                        async with asyncio.timeout(45):
                            result = await llm.complete_json(**request, attempts=1, timeout=45, stage='diagnostic.target_order')
                        selected, judgments = spans.decode_partial(result, built)
                        row.update(selected_ids=sorted(selected), judgments=judgments, payload=result)
                    except Exception as exc:
                        row.update(error=type(exc).__name__, detail=str(exc))
                row.update(calls=limits.calls, tokens=limits.tokens)
                rows.append(row)
                with paths[2].open('a') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('payload','judgments','detail')}), flush=True)
    summary = dict(jobs=len(jobs), live_calls=sum(r['calls'] for r in rows), errors=sum('error' in r for r in rows))
    paths[0].write_text(json.dumps(dict(summary=summary, plan=plan, live=rows), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    if len(rows) != plan['expected_calls'] or summary['errors']:
        raise RuntimeError('Diagnostic incomplete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--qas', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first-batch-only', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('Positive repeats required')
    asyncio.run(main(args))
