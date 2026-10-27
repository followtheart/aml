"""Full-cohort query-only diagnostic; no gold or retrieved evidence in requests."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, llm, query_premises


async def main(args):
    if config.FAKE:
        raise ValueError('Requires real provider')
    traces = [json.loads(l) for l in (args.run / 'completed/logs/search-debug.jsonl').read_text().splitlines()]
    results = [json.loads(l) for l in (args.run / 'completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines()]
    traces = {t['search_id']: t for t in traces}
    with args.output.open('x') as stream:
        for result in results:
            t = traces[result['search_id']]
            if t['user_id'] != result['user_id'] or t['packet_hash'] != result['packet_hash']:
                raise ValueError('Search identity mismatch')
            text = query_premises.prompt(t['query'], t['options'])
            row = dict(qa_id=result['qa_id'], search_id=t['search_id'],
                       prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
            started = time.monotonic()
            try:
                row['response'] = await llm.complete_json(text, json.dumps(query_premises.SCHEMA),
                    schema=query_premises.SCHEMA, stage='probe.option_premise_spans', attempts=1, timeout=20)
                row['option_queries'] = query_premises.compile_queries(t['options'], row['response'])
                row['valid'] = True
            except Exception as exc:
                row['valid'] = False
                row['error_type'] = type(exc).__name__
            row['seconds'] = round(time.monotonic() - started, 3)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush()
            print(json.dumps({k: v for k, v in row.items() if k not in ('response', 'option_queries')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
