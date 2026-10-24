"""Verify TLS, auth and real add/search/purge; always remove the isolated test user."""
import argparse
import http.client
import json
import os
import socket
import ssl
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit, quote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='https://www.dustman.com.cn')
    parser.add_argument('--connect-host', help='Optional connection address; TLS still verifies the URL hostname')
    parser.add_argument('--ca')
    parser.add_argument('--output')
    args = parser.parse_args()
    url = urlsplit(args.url)
    if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment:
        parser.error('Expected an HTTPS URL without credentials, query or fragment')
    token = os.environ.get('MC_API_KEY') or os.environ.get('AML_API_KEY')
    if not token:
        parser.error('Set MC_API_KEY (or AML_API_KEY)')
    context = ssl.create_default_context(cafile=args.ca)
    rows = []

    def call(method, path, body=None, authenticated=True, expected=200):
        started = time.monotonic()
        conn = http.client.HTTPSConnection(url.hostname, url.port or 443, context=context, timeout=360)
        if args.connect_host:
            conn.sock = context.wrap_socket(socket.create_connection((args.connect_host, url.port or 443), 360),
                                            server_hostname=url.hostname)
        headers = {'Content-Type': 'application/json'}
        if authenticated:
            headers['Authorization'] = 'Bearer ' + token
        try:
            conn.request(method, url.path.rstrip('/') + path,
                         body=json.dumps(body).encode() if body is not None else None, headers=headers)
            response = conn.getresponse()
            result = json.loads(response.read())
            assert response.status == expected, (method, path, response.status, result)
            assert response.getheader('X-Memory-System') == 'memory_connection'
            rows.append(dict(method=method, route=path.split('/https-smoke-')[0], status=response.status,
                             seconds=round(time.monotonic()-started, 3)))
            print(method, path.split('/https-smoke-')[0], response.status, flush=True)
            return result
        finally:
            conn.close()

    user = 'https-smoke-' + uuid.uuid4().hex
    added = None
    try:
        assert call('GET', '/health', authenticated=False) == {'status': 'ok'}
        query = dict(user_id=user, query='When did I start photography?', top_k=10)
        call('POST', '/search', query, authenticated=False, expected=401)
        body = dict(request_id='smoke', user_id=user, session_id='smoke', messages=[dict(
            role='user', content='I started photography in March 2024.', timestamp=1711929600000,
            source_event_id='smoke-source', trust_scope=['personal'])])
        added = call('POST', '/add', body)
        assert added['success'] and added['user_id'] == user
        assert call('POST', '/add', body) == added
        result = call('POST', '/search', dict(query, min_revision=added['write_revision']))
        assert any('March 2024' in item['content'] for item in result['data']), result
        rows[-1].update(evidence_status=result['evidence_status'], verification_status=result['verification_status'],
                        result_count=len(result['data']))
        feedback = call('POST', '/feedback', dict(user_id=user, session_id='smoke', task='HTTPS smoke',
            outcome='success', trace='Add and search completed.', distill=False, feedback_event_id='smoke'))
        assert feedback['memory_ids'] == []
        assert call('GET', '/feedback/' + quote(user) + '/smoke')['kind'] == 'observation'
    finally:
        receipt = call('DELETE', '/memory/' + quote(user))
        assert receipt['success'] and receipt['receipt_id']
    assert call('POST', '/search', query)['data'] == []
    report = dict(status='passed', url=args.url, tls_verified=True, checks=rows)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: TLS, auth, idempotence, real-model recall, feedback, and purge.')


if __name__ == '__main__':
    main()
