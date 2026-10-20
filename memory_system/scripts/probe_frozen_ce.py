"""Isolated live CE transport probe on archived batches, separate from listwise."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, cross_encoder

async def main(args):
    if config.FAKE:
        raise RuntimeError('Real provider required')
    searches = {s['search_id']: s for s in map(json.loads, (args.run / 'completed/logs/search-debug.jsonl').read_text().splitlines())}
    results = {r['qa_id']: r for r in map(json.loads, (args.run / 'completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines())}
    with args.output.open('x') as out:
        for index, qid in enumerate(args.questions.split(',')):
            trace = searches[results[qid]['search_id']]
            ce = trace['cascade']['cross_encoder']
            batch = next((b for b in ce['batches'] if b['status'] != 'ok'), ce['batches'][0])
            by_id = {c['id']: c for c in trace['ranked']}
            docs = [by_id[mid].get('_rank_text', by_id[mid]['content']) for mid in batch['candidate_ids']]
            for timeout in ([10., 20.] if index % 2 == 0 else [20., 10.]):
                row = dict(qa_id=qid, original_status=batch['status'], original_error=batch.get('error'),
                    candidate_ids=batch['candidate_ids'], timeout=timeout,
                    input_hash=hashlib.sha256(json.dumps([ce['query'], docs]).encode()).hexdigest())
                start = time.monotonic()
                with patch.object(config, 'CE_TIMEOUT_SECONDS', timeout), budget.scope(seconds=timeout+2, calls=1, tokens=200000):
                    try:
                        scores = await cross_encoder.rerank(ce['query'], docs, stage='probe.frozen_ce')
                        row.update(status='ok', scored_count=len(scores))
                    except Exception as exc:
                        row.update(status='error', error=type(exc).__name__)
                row['seconds'] = time.monotonic() - start
                out.write(json.dumps(row) + '\n'); out.flush()
                print(json.dumps(row), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--questions', default='0,1,4,7,9')
    asyncio.run(main(parser.parse_args()))
