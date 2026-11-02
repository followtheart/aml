"""Compare actual pre-hotfix and hotfix source_search on one production copy."""
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time

os.environ['AML_FAKE'] = '1'
os.environ['AML_SOURCE_FTS_BACKFILL'] = '0'
os.environ['AML_SESSION_FTS_BACKFILL'] = '0'
sys.path.insert(0, '/root/aml/memory_system')
from app import config, personal_evidence, store

ROOT = Path(__file__).resolve().parent
INDEX = 'idx_amu_sources_source'
DDL = f'CREATE INDEX {INDEX} ON amu_sources(request_id,message_index,amu_id)'
TIMEOUT = 20


def legacy_method():
    module = ast.parse((ROOT / 'baseline_store.py').read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'Store')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'source_search')
    namespace = vars(store).copy()
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'baseline_store.py', 'exec'), namespace)
    return namespace['source_search']


def measure(st, method, case):
    steps = 0
    started = time.monotonic()
    sqls = []

    def progress():
        nonlocal steps
        steps += 1000
        return int(time.monotonic() - started >= TIMEOUT)

    def trace(sql):
        if 'FROM source_fts' in sql and 'JOIN amu_sources' in sql:
            sqls.append(sql)

    st.conn.set_trace_callback(trace)
    st.conn.set_progress_handler(progress, 1000)
    try:
        rows = method(st, case['user_id'], case['query'], 60,
                      include_history=False, include_sensitive=True, include_cold=True)
        canonical = [(r['id'], r['_score'], r['content']) for r in rows]
        result = {'status': 'ok', 'returned': len(rows), 'digest': hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()}
    except sqlite3.OperationalError as exc:
        if 'interrupted' not in str(exc):
            raise
        result = {'status': 'timeout', 'timeout_s': TIMEOUT}
    finally:
        st.conn.set_progress_handler(None, 0)
        st.conn.set_trace_callback(None)
    result.update(elapsed_s=round(time.monotonic()-started, 6), approximate_sqlite_steps=steps)
    if sqls:
        result['query_plan'] = [r[3] for r in st.conn.execute('EXPLAIN QUERY PLAN '+sqls[0])]
    return result


def main():
    request = json.loads(Path('/root/aml/runs/https-status-20261102-1503/private-probe-request.json').read_text())
    c = sqlite3.connect(ROOT / 'snapshot.sqlite')
    c.row_factory = sqlite3.Row
    st = store.Store.__new__(store.Store)
    st.conn, st._lock = c, threading.RLock()
    c.execute('PRAGMA cache_size=-16384')
    index_started = time.monotonic()
    c.execute(DDL)
    c.commit()
    index_seconds = time.monotonic() - index_started

    log = Path('/root/aml/memory_system/logs/search-debug.jsonl')
    with log.open('rb') as f:
        f.seek(max(0, log.stat().st_size-4*1024*1024)); f.readline()
        traces = []
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('user_id') == request['user_id'] and row.get('query') == request['query']:
                traces.append(row)
    trace = next(r for r in reversed(traces) if r.get('query_specs'))
    queries = [r['text'] for r in trace['query_specs']]
    # Add narrow, actually present source terms for completed control comparisons.
    frequencies = Counter()
    for row in c.execute('''SELECT s.content FROM amu a JOIN amu_sources l ON l.amu_id=a.id
            JOIN source_messages s ON s.request_id=l.request_id AND s.message_index=l.message_index
            WHERE a.user_id=? AND a.valid_to IS NULL AND a.view_status='ready'
            AND a.resolution_status!='retracted' LIMIT 40''', (request['user_id'],)):
        frequencies.update(personal_evidence.terms(row['content']))
    rare = sorted((word for word, count in frequencies.items() if count == 1 and word.isascii()
                   and word.isalpha() and len(word) >= 8), key=lambda word: (-len(word), word))[:3]
    cases = [{'label': f'failed-query-spec-{i}', 'user_id': request['user_id'], 'query': q}
             for i, q in enumerate(queries)]
    cases += [{'label': f'narrow-source-term-{i}', 'user_id': request['user_id'], 'query': q}
              for i, q in enumerate(rare)]
    private = ROOT / 'private-cases.json'
    private.write_text(json.dumps(cases, ensure_ascii=False)); private.chmod(0o600)
    report = {'baseline_commit': '4e00cab', 'index_create_s': round(index_seconds, 3),
              'timeout_per_query_s': TIMEOUT, 'source_recall_diversity': config.SOURCE_RECALL_DIVERSITY,
              'scope': 'source_search only; one full-corpus production snapshot; no provider calls',
              'order': 'all candidate queries first, then all baseline queries; no cache eviction', 'cases': []}
    for case in cases:
        result = measure(st, store.Store.source_search, case)
        report['cases'].append({'label': case['label'], 'query_sha256': hashlib.sha256(case['query'].encode()).hexdigest(),
                                'candidate': result})
        print(json.dumps({'label': case['label'], 'arm': 'candidate', **result}), flush=True)
    c.execute(f'DROP INDEX {INDEX}'); c.commit()
    before = legacy_method()
    for case, row in zip(cases, report['cases']):
        result = measure(st, before, case)
        row['baseline'] = result
        if result['status'] == row['candidate']['status'] == 'ok':
            row['exact_output_match'] = result['digest'] == row['candidate']['digest']
        print(json.dumps({'label': case['label'], 'arm': 'baseline', **result}), flush=True)
        (ROOT / 'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
    c.execute(DDL); c.commit(); c.close()
    report['completed_pairs'] = sum('exact_output_match' in r for r in report['cases'])
    report['matched_pairs'] = sum(r.get('exact_output_match', False) for r in report['cases'])
    report['candidate_timeouts'] = sum(r['candidate']['status'] != 'ok' for r in report['cases'])
    report['baseline_timeouts'] = sum(r['baseline']['status'] != 'ok' for r in report['cases'])
    (ROOT / 'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}), flush=True)
    assert report['candidate_timeouts'] == 0
    assert report['completed_pairs'] == report['matched_pairs']


if __name__ == '__main__':
    main()
