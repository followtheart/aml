"""Replay a complete historical HTTPS cohort against memory_connection.

The frozen legacy reader is used only client-side. Add/Search always use the
running HTTPS service; neither its configuration nor legacy files are modified.
"""
import argparse
import asyncio
from collections import Counter
from contextvars import ContextVar
import hashlib
import http.client
import json
import logging
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import ssl
import statistics
import sys
import time
import uuid
from urllib.parse import quote, urlsplit

QUESTION = ContextVar('eval_question', default=None)


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def append(path, data):
    with Path(path).open('a') as out:
        out.write(json.dumps(data, ensure_ascii=False) + '\n')


def plan(baseline):
    manifest = json.loads((baseline / 'isolated-manifest.json').read_text())
    data = Path(manifest['data'])
    if digest(data) != manifest['data_sha256']:
        raise ValueError('Baseline dataset hash mismatch')
    conversations = {r['conversation_id']: r for r in records(data)}
    rows = records(baseline / 'completed/data/results/personamem-v2-32k.jsonl')
    requests = records(baseline / 'requests.jsonl')
    searches = [r['payload'] for r in requests if r['path'] == 'search']
    if len(searches) != len(rows) or len({(r['conversation_id'], r['qa_id']) for r in rows}) != len(rows):
        raise ValueError('Incomplete/duplicate baseline questions')
    groups = {}
    for row, search in zip(rows, searches):
        conv = conversations[row['conversation_id']]
        qa = next(q for q in conv['qa'] if q['id'] == row['qa_id'])
        if search['query'] != qa['question'] or search.get('options') != qa.get('options'):
            raise ValueError('Baseline request/result identities differ')
        group = groups.setdefault(row['user_id'], dict(conversation_id=row['conversation_id'], questions=[], adds=[]))
        if search['user_id'] != row['user_id']:
            raise ValueError('Baseline tenant mismatch')
        group['questions'].append(dict(qa=qa, baseline=row, request=search))
    for request in requests:
        if request['path'] == 'add':
            groups[request['payload']['user_id']]['adds'].append(request['payload'])
    for group in groups.values():
        if not group['adds']:
            raise ValueError('Missing full-history ingestion')
        if len(group['questions']) != len(conversations[group['conversation_id']]['qa']):
            raise ValueError('Expected complete conversation question sets')
    return manifest, data, rows, groups


class UsageLog(logging.Handler):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def emit(self, record):
        message = record.getMessage()
        if message.startswith('provider_call '):
            fields = dict(part.split('=', 1) for part in message.split()[1:] if '=' in part)
            append(self.path, dict(question=QUESTION.get(), **fields))


async def evaluate(args):
    started = time.monotonic()
    baseline = args.baseline.resolve()
    manifest, data, old_rows, groups = plan(baseline)
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    run.chmod(0o700)
    run_id = 'mc-https-' + uuid.uuid4().hex
    # Freeze the reader used by the historical baseline, including its resources.
    frozen = run / 'reader_source'
    for part in ['memory_system/app', 'prompts']:
        shutil.copytree(baseline / 'source' / part, frozen / part,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    lexicon = baseline / 'source/memory_system/data/lexicons/wordnet.zip'
    if lexicon.exists():
        target = frozen / 'memory_system/data/lexicons/wordnet.zip'
        target.parent.mkdir(parents=True)
        shutil.copyfile(lexicon, target)
    shutil.copyfile(data, run / 'prepared.jsonl')
    shutil.copyfile(baseline / 'requests.jsonl', run / 'baseline-requests.jsonl')
    shutil.copyfile(baseline / 'completed/data/results/personamem-v2-32k.jsonl', run / 'baseline-results.jsonl')
    sys.path.insert(0, str(frozen / 'memory_system'))
    from app import config, eval_scoring, run_metadata
    config.FAKE = False
    config.EVAL_CHOICE_POLICY = 'local'
    for key, value in old_rows[0]['versions']['settings'].items():
        setattr(config, key, value)
    config.LLM_MODEL = old_rows[0]['model']
    run_metadata.versions.cache_clear()
    reader_versions = run_metadata.versions()
    if reader_versions != old_rows[0]['versions']:
        mismatched = [k for k in reader_versions if reader_versions[k] != old_rows[0]['versions'].get(k)]
        raise ValueError(f'Frozen reader metadata differs: {mismatched}')
    logging.basicConfig(level=logging.WARNING)
    usage = logging.getLogger('aml.metrics')
    usage.setLevel(logging.INFO)
    usage.propagate = False
    usage.addHandler(UsageLog(run / 'reader-calls.jsonl'))
    for name in ['httpx', 'LiteLLM']:
        logging.getLogger(name).setLevel(logging.WARNING)
    address = urlsplit(args.url)
    if address.scheme != 'https' or not address.hostname or address.username or address.password:
        raise ValueError('HTTPS URL without credentials required')
    token = os.environ['MC_API_KEY']
    tls = ssl.create_default_context()
    users = {old: f'{run_id}:{i}' for i, old in enumerate(groups, 1)}
    if args.reuse_index:
        source_manifest = json.loads((args.reuse_index / 'manifest.json').read_text())
        if source_manifest['data_sha256'] != manifest['data_sha256']:
            raise ValueError('Frozen index uses a different dataset')
        if set(source_manifest['users']) != set(users):
            raise ValueError('Frozen index uses different tenants')
        users = source_manifest['users']
    server_settings = {k: v for k, v in os.environ.items() if k.startswith('MC_') and not any(s in k for s in ['KEY', 'TOKEN', 'SECRET'])}
    source_files = [p for base in [args.server_source / 'app', args.server_source / 'prompts']
                    for p in base.iterdir() if p.is_file() and p.suffix in ['.py', '.txt']]
    metadata = dict(run_id=run_id, baseline=str(baseline), url=args.url, connect_host=args.connect_host,
        tls_verified=True, data_sha256=digest(data), conversations=len(groups), questions=len(old_rows),
        add_chunks=sum(len(g['adds']) for g in groups.values()),
        messages=sum(len(a['messages']) for g in groups.values() for a in g['adds']),
        request_contract='baseline extended', reader_versions=reader_versions,
        server_settings=server_settings, source_hashes={str(p): digest(p) for p in source_files},
        concurrency=args.concurrency, users=users, status='running',
        fresh_ingestion=not bool(args.reuse_index),
        index_source=str(args.reuse_index.resolve()) if args.reuse_index else None,
        caveat=('Frozen indexed events; query expansion, retrieval and reader are rerun with model variability.'
                if args.reuse_index else
                'Fresh ingestion and a different retrieval system; historical comparison, not a randomized paired experiment.'))
    save(run / 'manifest.json', metadata)
    save(run / 'progress.json', dict(phase='preflight', adds=0, questions=0))
    counters = dict(adds=0, questions=0)
    responses, results, cleanup = [], [], []

    def request_sync(method, path, body):
        connection = http.client.HTTPSConnection(address.hostname, address.port or 443, context=tls, timeout=args.timeout)
        if args.connect_host:
            connection.sock = tls.wrap_socket(socket.create_connection((args.connect_host, address.port or 443), args.timeout),
                                               server_hostname=address.hostname)
        try:
            connection.request(method, address.path.rstrip('/') + path,
                body=json.dumps(body).encode() if body is not None else None,
                headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
            response = connection.getresponse()
            payload = json.loads(response.read())
            if response.status != 200 or response.getheader('X-Memory-System') != 'memory_connection':
                raise ValueError(f'{method} {path}: HTTP {response.status}, unexpected service or response')
            return payload, response.getheader('X-Request-ID')
        finally:
            connection.close()

    async def request(method, path, body=None, identity=None):
        begin = time.monotonic()
        record = dict(method=method, path=path, identity=identity, payload=body)
        append(run / 'requests.jsonl', record)
        try:
            payload, rid = await asyncio.to_thread(request_sync, method, path, body)
            row = dict(**record, response=payload, request_id=rid, seconds=time.monotonic()-begin, success=True)
        except Exception as exc:
            append(run / 'responses.jsonl', dict(**record, seconds=time.monotonic()-begin,
                                               success=False, error_type=type(exc).__name__))
            raise
        responses.append(row)
        append(run / 'responses.jsonl', row)
        return payload, row['seconds']

    def progress(phase):
        save(run / 'progress.json', dict(phase=phase, **counters, elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(phase=phase, **counters), ensure_ascii=False), flush=True)

    async def ingest(old, group):
        revision = 0
        for body in group['adds']:
            copied = dict(body, user_id=users[old], request_id=body['request_id'].replace(old, users[old]),
                          session_id=body['session_id'].replace(old, users[old]))
            result, _ = await request('POST', '/add', copied, group['conversation_id'])
            assert result['success'] and all(result[k] == copied[k] for k in ['request_id', 'user_id', 'session_id'])
            revision = max(revision, result['write_revision'])
            counters['adds'] += 1
            progress('ingestion')
        group['revision'] = revision

    def audit_ingestion():
        if not args.server_db:
            return None
        audit = []
        with sqlite3.connect(args.server_db.resolve().as_uri() + '?mode=ro', uri=True) as database:
            for old, group in groups.items():
                stored = [json.loads(row[0]) for row in database.execute('SELECT payload FROM events WHERE user_id=?', (users[old],))]
                if args.archive_index:
                    # Only this run's two isolated tenants, never a production
                    # database copy. Enables paired retrieval diagnostics later.
                    for event in stored:
                        append(run / 'index-events.jsonl', dict(user_id=users[old], event=event))
                sources = {e['logical_id']: next(s for s in e['sources'] if s['source_id'] == e['primary_source_id']) for e in stored}
                expected = {a['request_id'].replace(old, users[old]) + ':' + str(i): m
                            for a in group['adds'] for i, m in enumerate(a['messages'])}
                assert set(sources) == set(expected), 'Stored source identities differ'
                for key, message in expected.items():
                    assert all(sources[key].get(k) == v for k, v in message.items() if k != 'message_id'), 'Stored source differs'
                audit.append(dict(conversation_id=group['conversation_id'], messages=len(sources), verbatim=True,
                    expansion_status=dict(Counter(e['expansion_status'] for e in stored)),
                    indexed_views=sum(len(e['views']) for e in stored),
                    zero_derived_events=sum(len(e['views']) == 1 for e in stored)))
        save(run / 'ingestion-audit.json', audit)
        return audit

    semaphore = asyncio.Semaphore(args.concurrency)

    async def score(old, group, task):
        async with semaphore:
            qa, previous = task['qa'], task['baseline']
            key = f"{group['conversation_id']}:{qa['id']}"
            context_token = QUESTION.set(key)
            try:
                body = dict(task['request'], user_id=users[old], min_revision=group['revision'])
                search, search_seconds = await request('POST', '/search', body, key)
                begin = time.monotonic()
                prediction, value, diagnostics = await eval_scoring.evaluate(qa, search['data'])
                result = dict(conversation_id=group['conversation_id'], qa_id=qa['id'],
                    category=qa.get('category'), question=qa['question'], gold_labels=qa.get('gold_labels'),
                    prediction=prediction, score=value, baseline_score=previous['score'],
                    baseline_prediction=previous['prediction'], search_seconds=search_seconds,
                    answer_seconds=time.monotonic()-begin, returned_items=len(search['data']),
                    evidence_status=search.get('evidence_status'), verification_status=search.get('verification_status'),
                    search_id=search.get('search_id'), reader_diagnostics=diagnostics)
            except Exception as exc:
                result = dict(conversation_id=group['conversation_id'], qa_id=qa['id'], category=qa.get('category'),
                    score=0, baseline_score=previous['score'], error_type=type(exc).__name__)
            finally:
                QUESTION.reset(context_token)
            results.append(result)
            append(run / 'results.jsonl', result)
            counters['questions'] += 1
            progress('questions')

    try:
        health, _ = await request('GET', '/health')
        assert health == {'status': 'ok'}
        if args.reuse_index:
            for group in groups.values():
                group['revision'] = max(task['request'].get('min_revision', 0) for task in group['questions'])
            progress('frozen_index')
        else:
            async with asyncio.TaskGroup() as tasks:
                for old, group in groups.items():
                    tasks.create_task(ingest(old, group))
        audit_ingestion()
        await asyncio.gather(*(score(old, group, task) for old, group in groups.items() for task in group['questions']))
        assert len(results) == len(old_rows)
        metadata['status'] = 'completed'
    except BaseException as exc:
        metadata.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        progress('cleanup')
        for user in users.values():
            try:
                receipt, _ = await request('DELETE', '/memory/' + quote(user, safe=''))
                cleanup.append(receipt)
            except Exception as exc:
                cleanup.append(dict(user_id=user, success=False, error_type=type(exc).__name__))
        save(run / 'cleanup.json', cleanup)
        metadata['cleanup_complete'] = all(r.get('success') for r in cleanup)
        if args.server_db:
            # A successful DELETE against the wrong service/database is not a
            # cleanup receipt for the audited index. Check the same DB directly.
            with sqlite3.connect(args.server_db.resolve().as_uri() + '?mode=ro', uri=True) as database:
                remaining = {table: sum(database.execute(
                    f'SELECT COUNT(*) FROM {table} WHERE user_id=?', (user,)).fetchone()[0]
                    for user in users.values()) for table in ('events', 'requests', 'feedback_events', 'experiences')}
            metadata['cleanup_database_remaining'] = remaining
            metadata['cleanup_complete'] &= not any(remaining.values())
        metadata['elapsed_seconds'] = time.monotonic()-started
        save(run / 'manifest.json', metadata)
    def distribution(values):
        values = sorted(values)
        return dict(mean=statistics.mean(values), p50=statistics.median(values), p95=values[min(len(values)-1, int(.95*len(values)))]) if values else {}
    new_total, old_total = sum(r['score'] for r in results), sum(r['baseline_score'] for r in results)
    summary = dict(status=metadata['status'], questions=len(results), score=new_total, accuracy=new_total/len(results),
        baseline_score=old_total, baseline_accuracy=old_total/len(results),
        wins=sum(r['score'] > r['baseline_score'] for r in results),
        losses=sum(r['score'] < r['baseline_score'] for r in results),
        ties=sum(r['score'] == r['baseline_score'] for r in results),
        terminal_errors=sum(bool(r.get('error_type') or r.get('reader_diagnostics', {}).get('error_stage')) for r in results),
        evidence_status=dict(Counter(r.get('evidence_status') for r in results)),
        verification_status=dict(Counter(r.get('verification_status') for r in results)),
        empty_results=sum(r.get('returned_items') == 0 for r in results),
        search_seconds=distribution([r['search_seconds'] for r in results if 'search_seconds' in r]),
        answer_seconds=distribution([r['answer_seconds'] for r in results if 'answer_seconds' in r]),
        add_seconds=distribution([r['seconds'] for r in responses if r['path'] == '/add']),
        cleanup_complete=metadata['cleanup_complete'], elapsed_seconds=metadata['elapsed_seconds'],
        server_usage='Not exposed by deployed API; total retrieval token cost is unavailable.',
        caveat=metadata['caveat'])
    for field in ['conversation_id', 'category']:
        summary[field] = {str(v): dict(questions=len(part), score=sum(r['score'] for r in part),
            baseline_score=sum(r['baseline_score'] for r in part)) for v in dict.fromkeys(r.get(field) for r in results)
            if (part := [r for r in results if r.get(field) == v])}
    save(run / 'summary.json', summary)
    progress('completed')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return int(summary['terminal_errors'] > 0 or not summary['cleanup_complete'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('memory_system/runs/https-validation-baseline-20261024'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='https://www.dustman.com.cn')
    parser.add_argument('--connect-host')
    parser.add_argument('--server-db', type=Path)
    parser.add_argument('--archive-index', action='store_true',
                        help='Archive only this run\'s indexed events during the read-only ingestion audit')
    parser.add_argument('--reuse-index', type=Path,
                        help='Search-only replay of an archived run, already restored into an isolated service database')
    parser.add_argument('--server-source', type=Path, default=Path('memory_connection'),
                        help='Source directory of the service being evaluated')
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error('concurrency must be positive')
    return asyncio.run(evaluate(args))


if __name__ == '__main__':
    sys.exit(main())
