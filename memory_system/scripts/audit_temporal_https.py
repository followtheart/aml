"""Audit isolated temporal dataset runs without the PersonaMem-v2 gold CSV."""
import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.eval_data import ingestion_chunks, iter_plan, read_records, validate
from app.eval_scoring import aggregate_rubrics, choice_score
from app.personamem_v3 import PROTOCOL as V3_PROTOCOL
from app.perma import PROTOCOL as PERMA_PROTOCOL
from audit_https_funnel import archive


def audit(run, prepared, memory_log, search_log, dataset='personamem-v3'):
    protocol = {'personamem-v3': V3_PROTOCOL, 'perma': PERMA_PROTOCOL}[dataset]
    m = json.loads((run / 'isolated-manifest.json').read_text())
    if int((run / 'exit-code').read_text()) != 0 or m['client_exit_code'] != 0:
        raise ValueError('Incomplete client run')
    if hashlib.sha256(prepared.read_bytes()).hexdigest() != m['data_sha256']:
        raise ValueError('Prepared input changed')
    cfg = m['evaluation']
    selected = list(iter_plan(read_records(prepared), cfg['AML_EVAL_LIMIT'], cfg['AML_EVAL_CONVS']))
    paths = list((run / 'completed/data/results').glob('*.jsonl'))
    if len(paths) != 1:
        raise ValueError('Ambiguous results')
    rows = list(read_records(paths[0]))
    expected = [(c['conversation_id'], q['id']) for c, qs in selected for q in qs]
    if [(r['conversation_id'], r['qa_id']) for r in rows] != expected or not rows:
        raise ValueError('Result population/order mismatch')
    if len({r['run_id'] for r in rows}) != 1:
        raise ValueError('Mixed runs')
    run_id = rows[0]['run_id']
    conn = sqlite3.connect(f'file:{(run / "memory.db").resolve()}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    request_ids = []; message_count = 0; offset = 0
    try:
        for ci, (conv, qs) in enumerate(selected, 1):
            validate(conv)
            if conv['dataset'] != dataset or conv['protocol'] != protocol:
                raise ValueError('Wrong dataset/protocol')
            uid = f"https:{run_id}:{ci}:{dataset}:{conv['conversation_id']}"
            for sid, chunk, messages in ingestion_chunks(conv, cfg['AML_EVAL_CHUNK_MESSAGES'], cfg['AML_EVAL_CHUNK_WORDS']):
                request_id = f'{uid}:session:{sid}:chunk:{chunk}'
                request_ids.append(request_id)
                actual = list(conn.execute('SELECT * FROM source_messages WHERE request_id=? ORDER BY message_index', (request_id,)))
                if len(actual) != len(messages):
                    raise ValueError('Missing source messages')
                for i, (a, b) in enumerate(zip(actual, messages)):
                    if a['user_id'] != uid or a['message_index'] != i or any(a[k] != b.get(k) for k in ('role', 'content', 'timestamp')):
                        raise ValueError('Ingested original changed')
                    if b['timestamp'] >= conv['cutoff_seconds'] * 1000:
                        raise ValueError('Future/current event leakage')
                message_count += len(messages)
            for q, r in zip(qs, rows[offset:offset + len(qs)]):
                if r['dataset'] != dataset or r['protocol'] != protocol or r['transport'] != 'https' or r['fake'] or r['user_id'] != uid:
                    raise ValueError('Wrong execution identity')
                if not r.get('error_stage'):
                    score = (choice_score(r['prediction'], q['gold_labels'], q['qa_type']) if dataset == 'perma'
                             else aggregate_rubrics({'scores': r['rubric_scores']}, len(q['rubrics']), 'rubric_mean'))
                    if score != r['score']:
                        raise ValueError('Rubric score mismatch')
            offset += len(qs)
        actual_ids = [r[0] for r in conn.execute('SELECT request_id FROM requests ORDER BY rowid')]
        if actual_ids != request_ids or conn.execute('SELECT count(*) FROM source_messages').fetchone()[0] != message_count:
            raise ValueError('Extra/missing ingestion')
    finally:
        conn.close()
    users = {r['user_id'] for r in rows}
    archive(memory_log, run / 'completed/logs/memory-debug.jsonl', users)
    searches = archive(search_log, run / 'completed/logs/search-debug.jsonl', users)
    ids = {s['search_id'] for s in searches}
    if any(r.get('search_id') not in ids for r in rows if r.get('error_stage') != 'search'):
        raise ValueError('Missing search trace')
    report = dict(status='passed', protocol=protocol,
                count=len(rows), score=sum(r['score'] for r in rows),
                mean_score=sum(r['score'] for r in rows)/len(rows),
                errors=dict(Counter(r['error_stage'] for r in rows if r.get('error_stage'))),
                source_messages=message_count, ingestion_chunks=len(request_ids),
                original_messages_exact=True, temporal_isolation=True,
                by_category={cat: dict(count=len(group), mean_score=sum(r['score'] for r in group)/len(group))
                             for cat in sorted({r.get('category', 'unknown') for r in rows})
                             for group in [[r for r in rows if r.get('category', 'unknown') == cat]]})
    report['official_personamem_v3_score' if dataset == 'personamem-v3' else 'official_perma_score'] = False
    report['caveat'] = ('Two-dimension local rubric proxy; chatbot subset only. No v2 gold-snippet funnel or official v3 aggregate implied.'
                       if dataset == 'personamem-v3' else 'PERMA MCQ exact-label scores with local memory/reader policy. Checkpoint type 1 has no memory; later checkpoints preserve supplied same-day history. Not interactive task success or an official benchmark run.')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['run', 'prepared', 'memory-log', 'search-log']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--dataset', choices=['personamem-v3', 'perma'], default='personamem-v3')
    a = p.parse_args(); result = audit(a.run, a.prepared, a.memory_log, a.search_log, a.dataset)
    out = a.run / 'funnel'; out.mkdir(exist_ok=True)
    (out / 'https-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
