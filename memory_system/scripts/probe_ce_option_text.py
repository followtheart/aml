"""Live CE query diagnostic on unchanged archived coarse candidates.

Use every original option as a hypothesis; no gold label or target memory ID
enters scoring. This does not change retrieval, packet construction or answers.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from probe_ce_source_units import records, score_documents
from app import budget, config, rerank_query


async def main(args):
    if config.FAKE:
        raise ValueError('Real CE required')
    row = next(r for r in records(args.run / 'completed/data/results/personamem-v2-32k.jsonl')
               if r['qa_id'] == args.question)
    trace = next(t for t in records(args.run / 'server-search-debug.jsonl')
                 if t.get('search_id') == row['search_id'])
    by_id = {c['id']: c for c in trace['ranked']}
    ids = trace['cascade']['coarse']['selected_ids']
    documents = [by_id[mid].get('_rank_text', by_id[mid]['content']) for mid in ids]
    original = trace['cascade']['cross_encoder']['query']
    alternatives = ({f'option_{i + 1}': option for i, option in enumerate(trace['options'])}
                    if args.separate_options else {'treatment': '\n'.join(trace['options'])})
    queries = {'control': original}
    for name, text in alternatives.items():
        suffix = ('\n\nAlternative answer texts (hypotheses, not facts). '
                  'Rank evidence for their personal premises:\n' + text)
        if len(suffix.encode()) > rerank_query.MAX_AUXILIARY_BYTES:
            raise ValueError('Option texts must fit the existing auxiliary byte bound')
        queries[name] = original + suffix
        if len(queries[name].encode()) > config.CE_MAX_DOCUMENT_BYTES:
            raise ValueError('Query exceeds CE byte bound')
    plan = dict(run=str(args.run), qa_id=args.question, search_id=row['search_id'],
                candidate_ids=ids, document_hashes=[hashlib.sha256(d.encode()).hexdigest() for d in documents],
                queries=queries, separate_options=args.separate_options,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                caveat='Selected diagnostic; no gold labels in provider requests. Not answer accuracy.')
    with args.output.with_suffix('.plan.json').open('x') as stream:
        json.dump(plan, stream, indent=2)
    with args.output.open('x') as stream:
        for name, query in plan['queries'].items():
            with budget.scope(seconds=120, calls=10, tokens=200000) as usage:
                scores, batches = await score_documents(query, documents)
            result = dict(arm=name, candidate_ids=ids, scores=list(map(float, scores)),
                          order=sorted(ids, key=lambda mid: -scores[ids.index(mid)]),
                          calls=usage.calls, tokens=usage.tokens, batches=batches)
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            print(json.dumps(dict(arm=name, calls=usage.calls, candidates=len(ids))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--separate-options', action='store_true')
    asyncio.run(main(parser.parse_args()))
