"""Embed original adjacent user/assistant exchanges as retrieval keys.

An assistant reply can locate a generic editing request without becoming a user
fact. Pair only same-request consecutive eligible messages, normally sharing an
AMU. The explicit diagnostic option permits distinct eligible carriers. Preserve
both roles and exact full text. Reuse previously measured query vectors.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, embeddings
from evaluation_model_guard import check_runtime


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exchange_text(sources):
    if len(sources) == 1:
        return sources[0]['content']
    return '\n'.join(f"[{s['role']}]\n{s['content']}" for s in sources)


async def main(args):
    check_runtime()
    parent = args.probe.resolve()
    previous = json.loads((parent / 'plan.json').read_text())
    measured = json.loads((parent / 'rankings.json').read_text())
    if (config.FAKE or previous['model'] != config.EMBED_MODEL or previous['dimension'] != config.EMBED_DIM
            or digest(parent / 'plan.json') != measured['plan_sha256']
            or digest(parent / 'vectors.npz') != measured['vectors_sha256']):
        raise ValueError('Real model or frozen source/query vectors differ')
    root = Path(previous['source_run'])
    database = root / 'frozen-memory.db'
    if digest(database) != previous['database_sha256']:
        raise ValueError('Source database changed')
    conn = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    exchanges, pairs = {}, []
    try:
        for cohort in previous['cohorts']:
            rows = conn.execute('''SELECT s.*, a.id AS amu_id FROM source_messages s
                JOIN amu_sources l ON l.request_id=s.request_id AND l.message_index=s.message_index
                JOIN amu a ON a.id=l.amu_id
                WHERE s.user_id=? AND a.user_id=? AND s.role='assistant'
                  AND (? OR a.valid_to IS NULL)
                  AND (? OR (a.sensitivity!='sensitive' AND s.sensitivity!='sensitive'))
                  AND a.sensitivity!='suppressed' AND s.sensitivity!='suppressed'
                  AND a.view_status='ready' AND a.resolution_status!='retracted'
                  AND a.tier!='cold'
                ORDER BY s.request_id,s.message_index,a.id''',
                (cohort['user_id'], cohort['user_id'], cohort['include_history'], cohort['include_sensitive'])).fetchall()
            replies = {}
            for row in rows:
                key = row['request_id'], row['message_index']
                reply = replies.setdefault(key, dict({k: row[k] for k in row.keys() if k != 'amu_id'}, amu_ids=[]))
                reply['amu_ids'].append(row['amu_id'])
            for si in cohort['source_indices']:
                user = previous['sources'][si]
                reply = replies.get((user['request_id'], user['message_index'] + 1))
                shared = sorted(set(user['amu_ids']) & set(reply['amu_ids'])) if reply else []
                sources = [user]
                if (reply and (shared or args.allow_separate_carriers)
                        and reply['session_id'] == user['session_id'] and reply['content'].strip()):
                    sources.append(reply)
                    pairs.append(dict(source_index=si, common_amu_ids=shared,
                        user_source_event_id=user['source_event_id'], assistant_source_event_id=reply['source_event_id']))
                exchanges[str(si)] = sources
    finally:
        conn.close()
    documents = [dict(source_index=si, mode='exchange', text=exchange_text(exchanges[str(si)]))
                 for si in range(len(previous['sources']))]
    old_index = {t: i for i, t in enumerate(previous['texts'])}
    additional = list(dict.fromkeys(d['text'] for d in documents if d['text'] not in old_index))
    args.output.mkdir(parents=True, exist_ok=False)
    meta = dict(copy.deepcopy(previous), protocol=__doc__, source_sha256=digest(Path(__file__)),
        source_parent=str(parent), parent_plan_sha256=measured['plan_sha256'],
        pairing_policy='eligible_adjacent' if args.allow_separate_carriers else 'common_amu_adjacent',
        parent_vectors_sha256=measured['vectors_sha256'], documents=documents, exchange_sources=exchanges,
        paired_sources=pairs, new_texts=additional, texts=previous['texts'] + additional,
        limitation='Adjacent context is a retrieval key, not an assertion by the user. Same-request consecutive messages require independent eligible carriers; pairing_policy states whether they must share an AMU. Source eligibility joins are preserved, but downstream filters and packet limits still need testing. Query vectors and unchanged document vectors are reused from the parent measurement.')
    meta['texts_sha256'] = hashlib.sha256(json.dumps(meta['texts'], ensure_ascii=False).encode()).hexdigest()
    (args.output / 'plan.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    handler = logging.FileHandler(args.output / 'provider.log', mode='x')
    logging.getLogger('aml.metrics').addHandler(handler)
    logging.getLogger('aml.metrics').setLevel(logging.INFO)
    with budget.scope(seconds=300, calls=100, tokens=500000) as usage:
        fresh = await embeddings.embed(additional, stage='diagnostic.source_exchange_vectors')
    old_vectors = np.load(parent / 'vectors.npz', allow_pickle=False)['vectors']
    vectors = np.concatenate([old_vectors, fresh], axis=0)
    if vectors.shape != (len(meta['texts']), config.EMBED_DIM) or not np.isfinite(vectors).all():
        raise ValueError('Invalid exchange vector matrix')
    np.savez_compressed(args.output / 'vectors.npz', vectors=vectors)
    index = {t: i for i, t in enumerate(meta['texts'])}
    queries = meta['queries']
    sims = vectors[[index[q] for q in queries]] @ vectors[[index[d['text']] for d in documents]].T
    qi = {q: i for i, q in enumerate(queries)}
    rankings = []
    for cohort in meta['cohorts']:
        for question in cohort['questions']:
            qs = []
            for spec in question['query_specs']:
                ordered = sorted(cohort['source_indices'], key=lambda si: (-float(sims[qi[spec['text']], si]), si))
                qs.append(dict(query=spec, hits=[dict(source_index=si, document_index=si,
                    score=float(sims[qi[spec['text']], si])) for si in ordered]))
            rankings.append(dict(qa_id=question['qa_id'], user_id=cohort['user_id'],
                search_id=question['search_id'], modes=dict(exchange=qs)))
    result = dict(questions=len(rankings), sources=len(meta['sources']), documents=len(documents),
        paired_sources=len(pairs), new_texts=len(additional), reused_queries=len(queries),
        calls=usage.calls, tokens=usage.tokens, rankings=rankings,
        plan_sha256=digest(args.output / 'plan.json'), vectors_sha256=digest(args.output / 'vectors.npz'))
    (args.output / 'rankings.json').write_text(json.dumps(result, ensure_ascii=False) + '\n')
    (args.output / 'exit-code').write_text('0\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rankings'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-separate-carriers', action='store_true')
    asyncio.run(main(parser.parse_args()))
