"""Measure source semantic ranks on a complete frozen HTTPS cohort.

Use source_search's linked-source eligibility joins and actual history/sensitive
permissions. Exclude persona documents from the user-conversation channel.
Compare whole messages with exact two-sentence windows, collapsed per source.
No gold, summaries, translations, route fusion, or production index writes.
"""
import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, embeddings, persona_source, personal_evidence
from evaluation_model_guard import check_runtime, check_versions


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def main(args):
    check_runtime()
    if config.FAKE:
        raise ValueError('Real embedding provider required')
    root = args.run.resolve()
    if (root / 'exit-code').read_text().strip() != '0':
        raise ValueError('Require completed successful HTTPS cohort')
    plan = json.loads((root / 'plan.json').read_text())
    if plan['protocol'] != 'shared-search-reader-https-v1':
        raise ValueError('Require shared-Search archive')
    traces = records(root / 'search/server-search-debug.jsonl')
    jobs = records(root / 'http-reader-inputs.jsonl')
    ids = {job['payload']['search_id']: job['qa']['id'] for job in jobs}
    if len(ids) != plan['questions'] or len(traces) != plan['questions']:
        raise ValueError('Require whole cohort, no score-based selection')
    database = root / 'frozen-memory.db'
    if digest(database) != plan['database_sha256']:
        raise ValueError('Frozen database hash differs')
    groups = {}
    for trace in traces:
        check_versions(trace['versions'], 'Search trace')
        if trace['versions']['embedding_model'] != config.EMBED_MODEL:
            raise ValueError('Embedding model differs from Search')
        history = trace['include_history']
        sensitive = bool(trace['requested_include_sensitive'] and trace['versions']['settings']['SENSITIVE_RECALL_ENABLED'])
        groups.setdefault((trace['user_id'], history, sensitive), []).append(trace)
    conn = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    sources, documents, cohorts = [], [], []
    try:
        for group, members in sorted(groups.items()):
            user, history, sensitive = group
            rows = conn.execute('''SELECT s.*, a.id AS amu_id FROM source_messages s
                JOIN amu_sources l ON l.request_id=s.request_id AND l.message_index=s.message_index
                JOIN amu a ON a.id=l.amu_id
                WHERE s.user_id=? AND a.user_id=? AND s.role='user'
                  AND (? OR a.valid_to IS NULL)
                  AND (? OR (a.sensitivity!='sensitive' AND s.sensitivity!='sensitive'))
                  AND a.sensitivity!='suppressed' AND s.sensitivity!='suppressed'
                  AND a.view_status='ready' AND a.resolution_status!='retracted'
                  AND a.tier!='cold'
                ORDER BY s.request_id,s.message_index,a.id''', (user, user, history, sensitive)).fetchall()
            by_source = {}
            for row in rows:
                if persona_source.is_persona_message(row['content']) or not row['content'].strip():
                    continue
                key = (row['request_id'], row['message_index'])
                source = by_source.setdefault(key, dict({k: row[k] for k in row.keys() if k != 'amu_id'}, amu_ids=[]))
                source['amu_ids'].append(row['amu_id'])
            indices = []
            for source in by_source.values():
                si = len(sources)
                sources.append(source)
                indices.append(si)
                content = source['content']
                documents.append(dict(source_index=si, mode='whole', start=0, end=len(content), text=content))
                sentences = personal_evidence._sentences(content)
                spans = [(0, len(content))] if len(sentences) <= 2 else [
                    (sentences[i].start(), sentences[i + 1].end()) for i in range(len(sentences) - 1)]
                for start, end in spans:
                    documents.append(dict(source_index=si, mode='window2', start=start, end=end, text=content[start:end]))
            cohorts.append(dict(user_id=user, include_history=history, include_sensitive=sensitive,
                source_indices=indices, questions=[dict(qa_id=ids[t['search_id']], search_id=t['search_id'],
                    query_specs=t['query_specs']) for t in members]))
    finally:
        conn.close()
    queries = list(dict.fromkeys(q['text'] for c in cohorts for question in c['questions'] for q in question['query_specs']))
    texts = list(dict.fromkeys([d['text'] for d in documents] + queries))
    args.output.mkdir(parents=True, exist_ok=False)
    meta = dict(protocol=__doc__, source_run=str(root), database_sha256=digest(database),
        traces_sha256=digest(root / 'search/server-search-debug.jsonl'), model=config.EMBED_MODEL,
        dimension=config.EMBED_DIM, source_sha256=digest(Path(__file__)), sources=sources,
        documents=documents, queries=queries, texts=texts, cohorts=cohorts,
        texts_sha256=hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
        limitation='Source-channel eligibility only; downstream expiry, dependencies, memory coalescing, CE and packing are not simulated. Windows are retrieval keys, never self-contained evidence. Source vectors were measured now, not recovered from historical requests.')
    (args.output / 'plan.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    handler = logging.FileHandler(args.output / 'provider.log', mode='x')
    logging.getLogger('aml.metrics').addHandler(handler)
    logging.getLogger('aml.metrics').setLevel(logging.INFO)
    with budget.scope(seconds=300, calls=300, tokens=1000000) as usage:
        vectors = await embeddings.embed(texts, stage='diagnostic.current_source_vectors')
    if vectors.shape != (len(texts), config.EMBED_DIM) or not np.isfinite(vectors).all():
        raise ValueError('Invalid vector matrix')
    np.savez_compressed(args.output / 'vectors.npz', vectors=vectors)
    lookup = {text: i for i, text in enumerate(texts)}
    query_vectors = vectors[[lookup[q] for q in queries]]
    document_vectors = vectors[[lookup[d['text']] for d in documents]]
    scores = query_vectors @ document_vectors.T
    query_index = {q: i for i, q in enumerate(queries)}
    rankings = []
    for cohort in cohorts:
        allowed = set(cohort['source_indices'])
        for question in cohort['questions']:
            modes = {}
            for mode in ('whole', 'window2'):
                eligible = [i for i, d in enumerate(documents) if d['mode'] == mode and d['source_index'] in allowed]
                ranked_queries = []
                for query in question['query_specs']:
                    qi = query_index[query['text']]
                    ordered = sorted(eligible, key=lambda i: (-float(scores[qi, i]), i))
                    seen, hits = set(), []
                    for di in ordered:
                        si = documents[di]['source_index']
                        if si in seen:
                            continue
                        seen.add(si)
                        hits.append(dict(source_index=si, document_index=di, score=float(scores[qi, di])))
                    if seen != allowed:
                        raise ValueError('Collapsed ranks did not cover each source exactly once')
                    ranked_queries.append(dict(query=query, hits=hits))
                modes[mode] = ranked_queries
            rankings.append(dict(qa_id=question['qa_id'], user_id=cohort['user_id'], search_id=question['search_id'], modes=modes))
    result = dict(questions=len(rankings), sources=len(sources), documents=len(documents), unique_texts=len(texts),
        calls=usage.calls, tokens=usage.tokens, rankings=rankings,
        plan_sha256=digest(args.output / 'plan.json'), vectors_sha256=digest(args.output / 'vectors.npz'))
    (args.output / 'rankings.json').write_text(json.dumps(result, ensure_ascii=False) + '\n')
    (args.output / 'exit-code').write_text('0\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rankings'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
