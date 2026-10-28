"""Score the full archived non-rule pool; no gold-directed candidate insertion."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from probe_ce_source_units import records, score_documents
from app import budget, cascade_rerank, config


async def main(args):
    if config.FAKE:
        raise ValueError('Real CE required')
    result = next(r for r in records(args.run / 'completed/data/results/personamem-v2-32k.jsonl')
                  if str(r['qa_id']) == args.question)
    trace = next(t for t in records(args.run / 'completed/logs/search-debug.jsonl')
                 if t['search_id'] == result['search_id'])
    if trace['packet_hash'] != result['packet_hash'] or trace['user_id'] != result['user_id']:
        raise ValueError('Search identity mismatch')
    rows = [r for r in trace['ranked'] if not cascade_rerank.protected(r)]
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids) or len(ids) != trace['cascade']['coarse']['input_count']:
        raise ValueError('Full ordinary pool mismatch')
    documents = [r.get('_rank_text', r['content']) for r in rows]
    query = trace['cascade']['cross_encoder']['query']
    with args.output.with_suffix('.plan.json').open('x') as stream:
        json.dump(dict(qa_id=args.question, search_id=trace['search_id'], query=query,
                       ids=ids, hashes=[hashlib.sha256(d.encode()).hexdigest() for d in documents],
                       intervention='CE on every archived non-rule candidate; unchanged query and rendering',
                       caveat='Selected diagnostic with expanded budget. No new retrieval or downstream reader.'), stream, indent=2)
    with budget.scope(seconds=120, calls=12, tokens=250000) as usage:
        scores, batches = await score_documents(query, documents)
    selected = set(trace['cascade']['coarse']['selected_ids'])
    ranked = sorted([dict(id=r['id'], type=r['type'], score=float(score),
                          original_coarse_selected=r['id'] in selected)
                     for r, score in zip(rows, scores)], key=lambda r: -r['score'])
    report = dict(qa_id=args.question, candidates=len(rows), calls=usage.calls,
                  tokens=usage.tokens, batches=batches, ranked=ranked,
                  caveat='Live CE only, fixed archived candidates and query; score ranks do not prove final admission or answer correctness.')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'ranked'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
