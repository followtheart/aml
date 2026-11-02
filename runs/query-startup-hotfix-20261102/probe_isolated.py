"""Replay the previously failing HTTPS request against the isolated startup hotfix."""
from datetime import datetime, timezone
import http.client
import json
from pathlib import Path
import socket
import ssl
import time

root = Path(__file__).resolve().parent
cfg = json.loads(Path('/root/memory-system-runtime/service.json').read_text())
request = json.loads(Path('/root/aml/runs/https-status-20261102-1503/private-probe-request.json').read_text())
host = 'www.dustman.com.cn'


def connection(timeout):
    conn = http.client.HTTPSConnection(host, 18680, timeout=timeout)
    conn.sock = ssl.create_default_context().wrap_socket(
        socket.create_connection(('127.0.0.1', 18680), timeout=timeout), server_hostname=host)
    return conn


started = time.monotonic()
while time.monotonic() - started < 90:
    try:
        conn = connection(3)
        conn.request('GET', '/health')
        response = conn.getresponse()
        assert response.status == 200 and json.loads(response.read())['status'] == 'ok'
        conn.close()
        break
    except (OSError, http.client.HTTPException):
        time.sleep(1)
else:
    raise RuntimeError('Isolated service failed to become healthy')

report = {'readiness_wait_s': round(time.monotonic()-started, 3), 'started_at': datetime.now(timezone.utc).isoformat(),
          'target': '127.0.0.1:18680 using production hostname for verified TLS',
          'scope': 'first HTTPS search after ASGI SDK initialization; existing isolated copy; real providers',
          'baseline_live_elapsed_s': 300.25, 'baseline_live_http_status': 500,
          'limitation': 'Earlier live run and isolated run are not resource-matched performance trials.'}
print('Isolated HTTPS healthy; submitting the previously failing request.', flush=True)
started = time.monotonic()
conn = connection(330)
try:
    conn.request('POST', '/search', body=json.dumps(request).encode(), headers={
        'Content-Type': 'application/json', 'Authorization': 'Bearer '+cfg['env']['AML_API_KEY']})
    response = conn.getresponse()
    body = json.loads(response.read())
    report.update(http_status=response.status, request_id=response.getheader('x-request-id'),
                  evidence_status=body.get('evidence_status'), returned_items=len(body.get('data', [])),
                  missing_evidence_count=len(body.get('missing_evidence', [])),
                  provider_calls=body.get('coverage_manifest', {}).get('provider_calls'),
                  provider_tokens=body.get('coverage_manifest', {}).get('provider_tokens'))
    if response.status != 200:
        report['error'] = body.get('detail')
finally:
    conn.close()
    report['elapsed_s'] = round(time.monotonic() - started, 3)
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    (root / 'isolated-https.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
assert report.get('http_status') == 200 and report.get('returned_items', 0) > 0
