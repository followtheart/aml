"""Read-only, transaction-consistent fidelity check for one HTTPS conversation."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.eval_data import ingestion_chunks


def audit(run, plan_path):
    manifest = json.loads((run / 'isolated-manifest.json').read_text())
    plan = json.loads(plan_path.read_text())
    data = Path(manifest['data'])
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    if data.resolve() != Path(plan['input']).resolve() or digest != plan['sha256']:
        raise ValueError('Input differs from plan')
    conversations = [json.loads(l) for l in data.read_text().splitlines() if l.strip()]
    if len(conversations) != 1:
        raise ValueError('Only single-conversation plans supported')
    conv = conversations[0]
    if len(conv['qa']) != plan['expected_qa']:
        raise ValueError('Question count differs from plan')
    ids = re.findall(r'^run_id=([0-9a-f]{32}) ', (run / 'output.log').read_text(), re.M)
    if len(ids) != 1:
        raise ValueError('Ambiguous run identity')
    uid = f"https:{ids[0]}:1:{conv.get('dataset', 'normalized')}:{conv['conversation_id']}"
    chunks = list(ingestion_chunks(conv, plan.get('chunk_messages', 20), plan.get('chunk_words', 2000)))
    expected = [(f'{uid}:session:{sid}:chunk:{ci}', batch) for sid, ci, batch in chunks]
    if len(expected) != plan['expected_add_chunks'] or sum(len(m) for _, m in expected) != plan['source_messages']:
        raise ValueError('Chunk/message counts differ from plan')
    conn = sqlite3.connect(f'file:{(run / "memory.db").resolve()}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute('BEGIN')
        requests = list(conn.execute('SELECT * FROM requests ORDER BY rowid'))
        if [r['request_id'] for r in requests] != [key for key, _ in expected[:len(requests)]]:
            raise ValueError('Committed requests are not an input prefix')
        count = 0
        for request, (key, wanted) in zip(requests, expected):
            sid = key.rsplit(':chunk:', 1)[0]
            if request['user_id'] != uid or request['session_id'] != sid:
                raise ValueError('Request ownership mismatch')
            actual = list(conn.execute('SELECT * FROM source_messages WHERE request_id=? ORDER BY message_index', (key,)))
            if len(actual) != len(wanted):
                raise ValueError('Source count mismatch')
            for i, (row, message) in enumerate(zip(actual, wanted)):
                if (row['message_index'] != i or row['user_id'] != uid or row['session_id'] != sid
                        or any(row[k] != message.get(k) for k in ('role', 'content', 'timestamp'))):
                    raise ValueError(f'Original source mismatch: {key}, message {i}')
            count += len(actual)
        if count != conn.execute('SELECT count(*) FROM source_messages').fetchone()[0]:
            raise ValueError('Orphan source messages')
    finally:
        conn.close()
    return dict(run_id=ids[0], complete=len(requests) == len(expected),
                committed_chunks=len(requests), expected_chunks=len(expected),
                source_messages=count, expected_messages=plan['source_messages'],
                original_messages_exact=True, orphan_sources=False, input_sha256=digest,
                checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = audit(args.run, args.plan)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
