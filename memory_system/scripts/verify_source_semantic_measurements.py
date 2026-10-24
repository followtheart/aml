"""Independently reproduce semantic ranks and verify source eligibility/content."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import persona_source


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(probe):
    plan = json.loads((probe / 'plan.json').read_text())
    ranks = json.loads((probe / 'rankings.json').read_text())
    assert digest(probe / 'plan.json') == ranks['plan_sha256']
    assert digest(probe / 'vectors.npz') == ranks['vectors_sha256']
    database = Path(plan['source_run']) / 'frozen-memory.db'
    assert digest(database) == plan['database_sha256']
    conn = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    verified_sources = 0
    try:
        for cohort in plan['cohorts']:
            permitted = {}
            for row in conn.execute('''SELECT s.*,a.id AS amu_id FROM source_messages s
                JOIN amu_sources l ON l.request_id=s.request_id AND l.message_index=s.message_index
                JOIN amu a ON a.id=l.amu_id
                WHERE s.user_id=? AND a.user_id=s.user_id AND (? OR a.valid_to IS NULL)
                AND (? OR (s.sensitivity!='sensitive' AND a.sensitivity!='sensitive'))
                AND s.sensitivity!='suppressed' AND a.sensitivity!='suppressed'
                AND a.resolution_status!='retracted' AND a.view_status='ready' AND a.tier!='cold'
                ORDER BY s.request_id,s.message_index,a.id''',
                (cohort['user_id'], cohort['include_history'], cohort['include_sensitive'])):
                key = row['request_id'], row['message_index']
                original = permitted.setdefault(key, dict({k: row[k] for k in row.keys() if k != 'amu_id'}, amu_ids=[]))
                original['amu_ids'].append(row['amu_id'])
            user_keys = {k for k, s in permitted.items() if s['role'] == 'user' and s['content'].strip()
                         and not persona_source.is_persona_message(s['content'])}
            selected = [plan['sources'][si] for si in cohort['source_indices']]
            assert {(s['request_id'], s['message_index']) for s in selected} == user_keys
            for si in cohort['source_indices']:
                source = plan['sources'][si]
                assert source == permitted[source['request_id'], source['message_index']]
                for member in plan.get('exchange_sources', {}).get(str(si), []):
                    assert member == permitted[member['request_id'], member['message_index']]
                verified_sources += 1
    finally:
        conn.close()
    vectors = np.load(probe / 'vectors.npz', allow_pickle=False)['vectors']
    reused = 0
    if 'source_parent' in plan:
        parent = Path(plan['source_parent'])
        assert digest(parent / 'plan.json') == plan['parent_plan_sha256']
        assert digest(parent / 'vectors.npz') == plan['parent_vectors_sha256']
        old = np.load(parent / 'vectors.npz', allow_pickle=False)['vectors']
        assert np.array_equal(vectors[:len(old)], old)
        reused = len(old)
    index = {text: i for i, text in enumerate(plan['texts'])}
    qindex = {text: i for i, text in enumerate(plan['queries'])}
    docs = plan['documents']
    matrix = vectors[[index[q] for q in plan['queries']]] @ vectors[[index[d['text']] for d in docs]].T
    verified_queries, max_drift = 0, 0.0
    for row in ranks['rankings']:
        for mode, queries in row['modes'].items():
            eligible = [i for i, d in enumerate(docs) if d['mode'] == mode
                        and plan['sources'][d['source_index']]['user_id'] == row['user_id']]
            for query in queries:
                scores = matrix[qindex[query['query']['text']]]
                ordered = sorted(eligible, key=lambda di: (-float(scores[di]), di))
                expected, seen = [], set()
                for di in ordered:
                    si = docs[di]['source_index']
                    if si not in seen:
                        expected.append((si, di))
                        seen.add(si)
                assert [(h['source_index'], h['document_index']) for h in query['hits']] == expected
                for h in query['hits']:
                    delta = abs(float(scores[h['document_index']]) - h['score'])
                    max_drift = max(max_drift, delta)
                    assert delta <= 1e-6
                verified_queries += 1
    result = dict(questions=len(ranks['rankings']), source_rows_verified=verified_sources,
        query_rankings_reproduced=verified_queries, max_score_drift=max_drift, reused_vector_rows_exact=reused,
        database_sha256=plan['database_sha256'], rankings_sha256=digest(probe / 'rankings.json'),
        auditor_sha256=digest(Path(__file__)), provider_calls=0,
        limitation='Verifies measured source ranks and eligibility; does not establish downstream retention, semantic entailment, or production index lifecycle correctness.')
    with (probe / 'verification.json').open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    main(parser.parse_args().probe)
