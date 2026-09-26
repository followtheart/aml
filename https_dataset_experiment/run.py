#!/usr/bin/env python3
"""Run dataset ingestion and retrieval experiments over verified HTTPS."""
import argparse
import json
import os
from pathlib import Path
import ssl
import sys
import time
import urllib.request
import urllib.parse
import uuid


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def records(path):
    with path.open(encoding='utf-8') as f:
        if path.suffix == '.jsonl':
            for line in f:
                if line.strip():
                    yield json.loads(line)
        else:
            rows = json.load(f)
            if not isinstance(rows, list):
                raise ValueError('JSON must contain an array of conversations')
            yield from rows


def selected(args):
    for i, row in enumerate(records(args.data)):
        if i >= args.convs:
            break
        yield i, row


def preflight(args):
    counts = dict(conversations=0, sessions=0, messages=0, questions=0)
    for _, row in selected(args):
        if not isinstance(row.get('sessions'), list) or not isinstance(row.get('qa'), list):
            raise ValueError('Expected sessions and qa arrays; convert raw datasets first')
        counts['conversations'] += 1
        for session in row['sessions']:
            if not isinstance(session, list):
                raise ValueError('Each session must be a message array')
            counts['sessions'] += 1
            for message in session:
                if message.get('role') not in ('user', 'assistant') or not isinstance(message.get('content'), str) or not message['content'].strip():
                    raise ValueError('Each message requires user/assistant role and nonempty content')
                counts['messages'] += 1
        for qa in row['qa'][:args.limit]:
            if not isinstance(qa.get('question'), str) or not qa['question'].strip():
                raise ValueError('Each QA requires a nonempty question')
            counts['questions'] += 1
    if not counts['conversations']:
        raise ValueError('No conversations selected')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='https://www.dustman.com.cn')
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'memory_system/data/sample_eval.json')
    parser.add_argument('--convs', type=int, default=1)
    parser.add_argument('--limit', type=int, default=5, help='Questions per conversation')
    parser.add_argument('--chunk-messages', type=int, default=20)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--token-file', type=Path, help='Alternatively set AML_API_KEY')
    parser.add_argument('--ca', help='Optional trusted CA PEM')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'runs')
    parser.add_argument('--dry-run', action='store_true', help='Validate dataset without network requests')
    parser.add_argument('--health-only', action='store_true')
    args = parser.parse_args()
    u = urllib.parse.urlsplit(args.url)
    if u.scheme != 'https' or not u.hostname or u.username or u.password or u.query or u.fragment:
        parser.error('--url requires HTTPS without credentials, query or fragment')
    if min(args.convs, args.limit, args.chunk_messages, args.timeout) <= 0 or not 1 <= args.top_k <= 100:
        parser.error('Limits and timeout must be positive; top-k must be 1..100')
    counts = None if args.health_only else preflight(args)
    if counts:
        print(json.dumps(counts), flush=True)
    if args.dry_run:
        return 0
    token = args.token_file.read_text().strip() if args.token_file else os.environ.get('AML_API_KEY', '')
    if not args.health_only and not token:
        parser.error('Provide --token-file or AML_API_KEY')
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=args.ca)), NoRedirect())

    def request(path, body=None):
        headers = {'Accept': 'application/json'}
        data = None
        if body is not None:
            headers.update({'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
            data = json.dumps(body, ensure_ascii=False).encode()
        with opener.open(urllib.request.Request(args.url.rstrip('/') + path, data=data, headers=headers), timeout=args.timeout) as response:
            return json.load(response)

    health = request('/health')
    if health.get('status') != 'ok':
        raise ValueError('Health check failed')
    print('HTTPS health: ok', flush=True)
    if args.health_only:
        return 0
    run_id = uuid.uuid4().hex
    out = args.output / run_id
    out.mkdir(parents=True, mode=0o700)
    summary = dict(run_id=run_id, url=args.url, data=str(args.data.resolve()), selected=counts,
                   questions_ok=0, questions_failed=0, ingestion_failed=0, nonempty_results=0)
    start = time.monotonic()
    print(f'Results: {out}', flush=True)
    with (out / 'events.jsonl').open('w', encoding='utf-8') as log:
        def emit(event):
            log.write(json.dumps(event, ensure_ascii=False) + '\n')
            log.flush()
        try:
            for ci, row in selected(args):
                user = f'https-dataset:{run_id}:{ci}'
                revision = 0
                stage = 'add'
                try:
                    for si, session in enumerate(row['sessions']):
                        for offset in range(0, len(session), args.chunk_messages):
                            body = dict(user_id=user, session_id=f'{user}:{si}', request_id=f'{user}:{si}:{offset}', messages=session[offset:offset + args.chunk_messages])
                            t = time.monotonic()
                            result = request('/add', body)
                            if result.get('success') is not True or any(result.get(k) != body[k] for k in ('request_id', 'user_id', 'session_id')):
                                raise ValueError('Invalid add acknowledgement')
                            revision = max(revision, result['write_revision'])
                            emit(dict(stage=stage, user_id=user, session=si, offset=offset, elapsed_s=time.monotonic()-t, response=result))
                            print(f'conversation {ci + 1}: session {si + 1}, messages {offset + len(body["messages"])}/{len(session)}', flush=True)
                except Exception as exc:
                    summary['ingestion_failed'] += 1
                    emit(dict(stage=stage, user_id=user, error=str(exc)))
                    print(f'Ingestion failed for conversation {ci + 1}: {exc}', file=sys.stderr)
                    continue  # Never evaluate a partially ingested conversation.
                for qi, qa in enumerate(row['qa'][:args.limit]):
                    t = time.monotonic()
                    event = dict(stage='search', user_id=user, conversation_id=row.get('conversation_id'), question_index=qi, qa=qa)
                    try:
                        body = dict(user_id=user, query=qa['question'], top_k=args.top_k, min_revision=revision)
                        for source, target in [('options', 'options'), ('question_date', 'reference_time'), ('as_of', 'as_of')]:
                            if qa.get(source) is not None:
                                body[target] = qa[source]
                        result = request('/search', body)
                        if not isinstance(result.get('data'), list):
                            raise ValueError('Invalid search response')
                        event['response'] = result
                        summary['questions_ok'] += 1
                        summary['nonempty_results'] += bool(result['data'])
                    except Exception as exc:
                        summary['questions_failed'] += 1
                        event['error'] = str(exc)
                    event['elapsed_s'] = time.monotonic() - t
                    emit(event)
                    print(f'conversation {ci + 1}, question {qi + 1}: {"error" if "error" in event else "ok"} ({event["elapsed_s"]:.1f}s)', flush=True)
        finally:
            summary['elapsed_s'] = time.monotonic() - start
            (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(bool(summary['questions_failed'] or summary['ingestion_failed']))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f'Experiment failed: {exc}', file=sys.stderr)
        sys.exit(1)
