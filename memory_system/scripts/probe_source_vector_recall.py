"""Read-only source granularity probe using archived retrieval queries.

Embeds complete eligible original messages, then ranks them per original query.
No expected answers, target source IDs or generated paraphrases enter requests.
This does not implement route fusion, runtime indexing or an answer evaluation.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, embeddings
from app.personal_evidence import _sentences


def records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


async def main(args):
    if config.FAKE:
        raise ValueError('Real embedding provider required')
    results = list(records(args.run / 'completed/data/results/personamem-v2-32k.jsonl'))
    wanted = set(args.questions.split(',')) if args.questions else {r['qa_id'] for r in results}
    ids = {r['search_id']: r['qa_id'] for r in results if r['qa_id'] in wanted}
    if set(ids.values()) != wanted:
        raise ValueError('Missing selected result')
    traces = [r for r in records(args.run / 'server-search-debug.jsonl') if r.get('search_id') in ids]
    users = {t['user_id'] for t in traces}
    if len(users) != 1 or len(traces) != len(ids):
        raise ValueError('Require one complete isolated conversation')
    user = users.pop()
    conn = sqlite3.connect(f'file:{(args.run / "memory.db").resolve()}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute('''SELECT s.request_id,s.message_index,s.role,s.content,
            s.timestamp,s.source_event_id,a.id AS amu_id FROM source_messages s
            JOIN amu_sources l ON l.request_id=s.request_id AND l.message_index=s.message_index
            JOIN amu a ON a.id=l.amu_id
            WHERE s.user_id=? AND a.user_id=? AND a.valid_to IS NULL
              AND a.sensitivity NOT IN ('sensitive','suppressed')
              AND s.sensitivity NOT IN ('sensitive','suppressed')
              AND a.view_status='ready' AND a.resolution_status!='retracted'
              AND a.tier!='cold'
            ORDER BY s.request_id,s.message_index,a.id''', (user, user)).fetchall()
    finally:
        conn.close()
    sources = {}
    for row in rows:
        key = (row['request_id'], row['message_index'])
        if key not in sources:
            sources[key] = {k: row[k] for k in row.keys() if k != 'amu_id'} | {'amu_ids': []}
        sources[key]['amu_ids'].append(row['amu_id'])
    sources = list(sources.values())
    if args.user_only:
        sources = [s for s in sources if s['role'] == 'user']
    documents = []
    for si, source in enumerate(sources):
        content = source['content']
        sentences = _sentences(content) if args.sentence_window else []
        spans = [(0, len(content))]
        if len(sentences) > args.sentence_window > 0:
            spans = [(sentences[i].start(), sentences[i + args.sentence_window - 1].end())
                     for i in range(len(sentences) - args.sentence_window + 1)]
        for start, end in spans:
            documents.append(dict(source_index=si, start=start, end=end, text=content[start:end]))
    queries = list(dict.fromkeys(q['text'] for t in traces for q in t['query_specs']))
    if not sources or not queries:
        raise ValueError('Empty source or query collection')
    if args.output.exists():
        raise FileExistsError(args.output)
    with budget.scope(seconds=240, calls=100, tokens=500000) as usage:
        vectors = await embeddings.embed([d['text'] for d in documents], stage='probe.source_vectors')
        query_vectors = await embeddings.embed(queries, stage='probe.source_queries')
    similarities = query_vectors @ vectors.T
    user_indices = {i for i, d in enumerate(documents) if sources[d['source_index']]['role'] == 'user'}
    rankings = {}
    for qi, query in enumerate(queries):
        ranked = sorted(range(len(documents)), key=lambda i: (-float(similarities[qi, i]), i))
        def hit(i):
            return dict(document_index=i, source_index=documents[i]['source_index'],
                        score=float(similarities[qi, i]))
        rankings[query] = {
            'all': [hit(i) for i in ranked],
            'user': [hit(i) for i in ranked if i in user_indices]}
    report = dict(user_id=user, model=config.EMBED_MODEL, dimension=config.EMBED_DIM,
                  sources=sources, documents=documents, rankings=rankings,
                  sentence_window=args.sentence_window, user_only=args.user_only,
                  questions=[dict(qa_id=ids[t['search_id']], search_id=t['search_id'],
                                  query_specs=t['query_specs'],
                                  baseline_fused_ids=[m['id'] for m in t['fused']]) for t in traces],
                  calls=usage.calls, tokens=usage.tokens,
                  source_text_sha256=hashlib.sha256(json.dumps(
                      [s['content'] for s in sources], ensure_ascii=False).encode()).hexdigest(),
                  caveat='Diagnostic only. Original messages or exact sentence windows embedded with existing model; '
                  'no new source facts, gold or query expansion. Role/time retained separately. '
                  'Post-run active linked source view; not historical replay, route fusion, '
                  'production index performance or downstream accuracy.')
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(sources=len(sources), documents=len(documents), queries=len(queries), questions=len(traces),
                          calls=usage.calls, tokens=usage.tokens)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--questions')
    p.add_argument('--sentence-window', type=int, choices=(0, 1, 2, 3), default=0,
                   help='0 means complete messages; positive values preserve exact consecutive sentence spans')
    p.add_argument('--user-only', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    asyncio.run(main(p.parse_args()))
