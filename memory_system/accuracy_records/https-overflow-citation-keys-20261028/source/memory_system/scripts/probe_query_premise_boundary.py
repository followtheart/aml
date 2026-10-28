"""Compare query-only premise boundaries on every archived question; no gold input."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, llm, prompts


GUIDANCE = '''
Before writing each option query, separate the option's existing personal
premise from its proposed next action. Search for evidence of the existing
premise only. Locations, products, people and activities introduced solely as
recommendations must not appear in that option query, even if they are concrete
named entities. Preserve a named entity when the option says it ALREADY belongs
to the user's history, preferences or circumstances. Do not turn an "if" into
an established fact. Apply the same boundary to expanded_queries and entities.
Examples:
- "Since you enjoy ceramics, try visiting the Northgate Craft Fair" ->
  "user interest in ceramics" (not the suggested fair).
- "Since you regularly visit the Northgate Craft Fair, try another exhibition" ->
  "user regular visits to Northgate Craft Fair" (keep the existing venue).
- "Try visiting the Northgate Craft Fair" -> "" (no personal premise).
- "If you own a canoe, you could explore Silver Creek" ->
  "whether user owns a canoe" (not the suggested destination).
'''


async def main(args):
    if config.FAKE:
        raise ValueError('Real planner required')
    result_path = args.run / 'completed/data/results/personamem-v2-32k.jsonl'
    results = [json.loads(line) for line in result_path.read_text().splitlines()]
    trace_path = args.run / 'completed/logs/search-debug.jsonl'
    traces = {r['search_id']: r for r in
              (json.loads(line) for line in trace_path.read_text().splitlines())}
    cases = []
    for result in results:
        trace = traces[result['search_id']]
        if trace['user_id'] != result['user_id'] or trace['packet_hash'] != result['packet_hash']:
            raise ValueError('Search identity mismatch')
        cases.append(dict(qa_id=result['qa_id'], query=trace['query'], options=trace['options'],
                          time=trace['anchor_time'], search_id=trace['search_id']))
    plan = dict(run=str(args.run), questions=len(cases), guidance=GUIDANCE,
                input_sha256=hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                caveat='Query planning only. No memories, gold labels, preference labels, or scoring '
                       'in provider input. Alternating arm order; no retrieval or accuracy claim.')
    with args.output.with_suffix('.plan.json').open('x') as stream:
        json.dump(plan, stream, indent=2)
    with args.output.open('x') as stream:
        for index, case in enumerate(cases):
            prompt = prompts.render('04_query_understanding.txt', query=case['query'],
                                    options='\n'.join(case['options']), current_time=case['time'],
                                    user_profile='(none)')
            arms = [False, True] if index % 2 == 0 else [True, False]
            for treatment in arms:
                text = prompt + (GUIDANCE if treatment else '')
                started = time.monotonic()
                row = dict(qa_id=case['qa_id'], search_id=case['search_id'],
                           treatment=treatment, prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
                try:
                    row['plan'] = await llm.complete_json(text, json.dumps(llm.STRUCTURED_SCHEMAS['query']),
                        schema=llm.STRUCTURED_SCHEMAS['query'], stage='probe.query_premise_boundary',
                        attempts=1, timeout=20)
                except Exception as exc:
                    row['error_type'] = type(exc).__name__
                row['seconds'] = round(time.monotonic() - started, 3)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush()
                print(json.dumps({k: v for k, v in row.items() if k not in ('plan', 'prompt_sha256')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
