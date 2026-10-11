"""Compare archived segment boundaries with complete user exchanges.

Uses original source messages and summary only; no evaluation questions or gold.
Runs normal grounding and semantic verification without database writes.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import add_pipeline, budget, config, memory_debug, schemas


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-log', type=Path, required=True)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--segment', type=int, required=True)
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    row = next(json.loads(line) for line in args.memory_log.open()
               if json.loads(line).get('request_id') == args.request_id)
    trace = next(t for t in row['extraction_trace']
                 if t['segment_index'] == args.segment)
    indices = trace['message_indices']
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError('Source segment must be contiguous')
    prompt = next(e['prompt'] for e in trace['events']
                  if e['stage'] == 'extract_request')
    summary = prompt.split('Conversation summary so far (global context, do not extract from it alone):\n', 1)[1].split('\n\nRecent messages (local context):', 1)[0]
    messages = []
    for i, item in enumerate(sorted(row['source_messages'], key=lambda s: s['message_index'])):
        if item['message_index'] != i:
            raise ValueError('Source indices must be contiguous')
        fields = {k: v for k, v in item.items() if k in schemas.Message.model_fields}
        if isinstance(fields.get('trust_scope'), str):
            fields['trust_scope'] = json.loads(fields['trust_scope'])
        messages.append(schemas.Message(**fields))
    req = schemas.AddRequest(request_id=row['request_id'], user_id=row['user_id'],
                             session_id=row['session_id'], messages=messages)
    starts = [i for i in indices if messages[i].role == 'user']
    if not starts or starts[0] != indices[0]:
        raise ValueError('Diagnostic requires a user-led segment')
    exchanges = [list(range(start, stop)) for start, stop in
                 zip(starts, starts[1:] + [indices[-1] + 1])]
    config.MEMORY_DEBUG_LOG = 'diagnostic-capture-only'
    with args.output.open('x') as output:
        plan = dict(event='plan', source_log=str(args.memory_log),
                    request_id=args.request_id, segment=args.segment,
                    source_sha256=hashlib.sha256(json.dumps(row['source_messages'],
                        sort_keys=True).encode()).hexdigest(),
                    boundaries={'segment': [indices], 'exchange': exchanges},
                    repeat=args.repeat, fake=config.FAKE,
                    scope='Live extraction and semantic checks; no DB writes or gold input')
        output.write(json.dumps(plan) + '\n')
        output.flush()
        for repeat in range(args.repeat):
            arms = ['segment', 'exchange'] if repeat % 2 == 0 else ['exchange', 'segment']
            for arm in arms:
                for unit in plan['boundaries'][arm]:
                    started = time.monotonic()
                    with budget.scope(seconds=240, calls=30, tokens=240000) as limits:
                        with memory_debug.extraction_scope():
                            facts = await add_pipeline._extract_segment(req, summary, args.segment, unit)
                            captured = copy.deepcopy(memory_debug._extraction.get())
                    result = dict(event='result', repeat=repeat, arm=arm,
                                  indices=unit, facts=facts, extraction_trace=captured,
                                  calls=limits.calls, tokens=limits.tokens,
                                  elapsed_seconds=time.monotonic() - started)
                    output.write(json.dumps(result, ensure_ascii=False) + '\n')
                    output.flush()
                    print(json.dumps({k: result[k] for k in
                        ['repeat', 'arm', 'indices', 'calls', 'elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
