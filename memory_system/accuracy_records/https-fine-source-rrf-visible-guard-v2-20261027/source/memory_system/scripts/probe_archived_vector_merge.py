"""Read-only archived dense-route diagnosis with fresh original query embeddings."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, embeddings, search_pipeline, vector_index


async def main(args):
    results = [json.loads(l) for l in (args.run / 'completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines()]
    wanted = set(args.questions.split(','))
    ids = {r['search_id']: r['qa_id'] for r in results if r['qa_id'] in wanted}
    assert set(ids.values()) == wanted
    traces = [json.loads(l) for l in (args.run / 'server-search-debug.jsonl').open()]
    output = []
    db = sqlite3.connect(f'file:{(args.run / "memory.db").resolve()}?mode=ro', uri=True)
    for trace in traces:
        if trace.get('search_id') not in ids:
            continue
        with patch.multiple(config, **trace['versions']['settings']):
            assert not config.FAKE
            assert not trace['include_history']
            sql = """SELECT id,embedding FROM amu WHERE user_id=? AND embedding_space=?
                     AND embedding IS NOT NULL AND valid_to IS NULL
                     AND sensitivity!='suppressed'
                     AND view_status='ready' AND resolution_status!='retracted' ORDER BY id"""
            if not args.include_sensitive:
                sql = sql.replace("AND sensitivity!='suppressed'", "AND sensitivity NOT IN ('sensitive','suppressed')")
            memories = db.execute(sql, (trace['user_id'], config.EMBEDDING_SPACE)).fetchall()
            query_specs = trace['query_specs']
            assert len(trace['rounds']) == 1, 'This diagnostic requires a single retrieval round'
            queries = [q['text'] for q in query_specs]
            with budget.scope(seconds=120, calls=20, tokens=100000):
                vectors = await embeddings.embed(queries, stage='probe.archived_vector_queries')
                index = vector_index.Index(memories)
                rankings, diagnostics = index.search(vectors, [m[0] for m in memories], len(memories))
            limit = config.RECALL_VECTOR_LIMIT
            routes = [[{'id': mid, 'score': score} for mid, score in ranking[:limit]] for ranking in rankings]
            merged = search_pipeline._merge_query_results(routes, limit)
            archived = next(r['candidates'] for r in trace['routes'] if r['channel'] == 'vector')
            output.append(dict(qa_id=ids[trace['search_id']], query_specs=query_specs,
                               rankings=rankings, diagnostics=diagnostics, limit=limit,
                               merged_ids=[r['id'] for r in merged],
                               archived_ids=[r['id'] for r in archived],
                               archived_route_exact=[r['id'] for r in merged] == [r['id'] for r in archived],
                               vectors_sha256=hashlib.sha256(b''.join(blob for _, blob in memories)).hexdigest()))
    db.close()
    assert len(output) == len(wanted)
    report = dict(rows=output, include_sensitive=args.include_sensitive,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  caveat='Fresh original-query embeddings against archived active memory vectors; no database mutation, new documents, gold, or query expansion. Post-run eligibility snapshot. Sensitive-memory inclusion must explicitly match original client/server policy; suppressed memories always excluded. Full rankings distinguish per-query retrieval from bounded route merge; archived-route equality is checked, not assumed. Not an answer intervention.')
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps([dict(qa_id=r['qa_id'], archived_route_exact=r['archived_route_exact']) for r in output]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--questions', required=True)
    parser.add_argument('--include-sensitive', action='store_true',
                        help='Match an archived run that allowed sensitive memory recall; does not change the database')
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
