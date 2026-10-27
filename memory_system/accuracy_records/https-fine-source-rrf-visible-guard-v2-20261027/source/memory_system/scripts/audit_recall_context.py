"""Independently score archived recall runs using CSV row/history identity.

No providers or pipeline execution. Keeps original results untouched and writes
a separate audit so corrections to reference matching remain reviewable.
"""
import argparse
import csv
import json
from pathlib import Path
import sqlite3
import statistics

from evaluate_recall_context import CSV, packet_coverage, records, reference_turns


def audit(run, prepared):
    conversations = {d['conversation_id']: d for d in records(prepared)}
    csv.field_size_limit(32 * 1024 * 1024)
    with CSV.open() as source:
        gold = list(csv.DictReader(source))
    traces = {t['search_id']: t for t in records(run / 'search-debug.jsonl')}
    raw = {}
    with sqlite3.connect(f'file:{(run / "memory.db").resolve()}?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for s in conn.execute('SELECT * FROM source_messages'):
            raw.setdefault(s['user_id'], {})[s['request_id'], s['message_index']] = dict(s)
    rows = []
    for r in records(run / 'results.jsonl'):
        conv = conversations[r['conversation_id']]
        q = next(q for q in conv['qa'] if q['id'] == r['qa_id'])
        turns = reference_turns(q, conv['source_history'], gold[int(r['qa_id'])])
        if r.get('search_id') not in traces:
            rows.append(dict(qa_id=r['qa_id'], error=True, score=r['score']))
            continue
        t = traces[r['search_id']]
        if t['query'] != q['question'] or t['options'] != q['options']:
            raise ValueError('Trace/query mismatch')
        items = {c['id']: c.get('_packet_item', {}) for c in t['ranked']}
        def covered(ids):
            return packet_coverage(turns, [items[mid] for mid in ids if mid in items], raw[t['user_id']])
        stages = {'prepared': covered(items)}
        for name, ids in [('coarse', t['cascade']['coarse']['selected_ids']),
                          ('fine', t['cascade']['fine']['selected_ids']),
                          ('listwise_input', t['cascade']['listwise']['candidate_ids']),
                          ('listwise_selected', t['cascade']['listwise']['selected_ids'])]:
            stages[name] = covered(ids)
        stages['packet'] = packet_coverage(turns, t['returned'], raw[t['user_id']])
        rows.append(dict(qa_id=r['qa_id'], conversation_id=r['conversation_id'], score=r['score'],
            stages=stages, search_seconds=r['search_seconds'], provider_calls=r['provider_calls'],
            provider_tokens=r['provider_tokens'], degraded=r['degraded'],
            reader_error=r.get('diagnostics', {}).get('answer_status') == 'error'))
    good = [r for r in rows if 'stages' in r]
    times = sorted(r['search_seconds'] for r in good)
    summary = dict(questions=len(rows), search_errors=len(rows)-len(good),
        reader_errors=sum(r['reader_error'] for r in good), score=sum(r['score'] for r in rows),
        reference_present=sum(r['stages']['packet']['reference_present'] for r in good),
        visible={stage: sum(r['stages'][stage]['visible'] for r in good) for stage in good[0]['stages']} if good else {},
        full_turns={stage: sum(r['stages'][stage]['full_turns'] for r in good) for stage in good[0]['stages']} if good else {},
        median_seconds=statistics.median(times) if times else None,
        p95_seconds=times[min(len(times)-1, int(len(times)*.95))] if times else None,
        search_calls=sum(r['provider_calls'] for r in good), search_tokens=sum(r['provider_tokens'] for r in good),
        degraded=sum(r['degraded'] for r in good))
    report = dict(summary=summary, rows=rows, reference_policy='csv_row_question_answer_history_v1',
        limitation='Literal complete-user-turn coverage, not semantic entailment. Single stochastic run; concurrent latency.')
    (run / 'coverage-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    summary_path = run / 'summary.json'
    if summary_path.exists():
        original = summary_path.read_text()
        backup = run / 'summary-before-reference-audit.json'
        if not backup.exists():
            backup.write_text(original)
        updated = json.loads(original)
        updated.update(reference_present=summary['reference_present'],
                       visible_questions=summary['visible'].get('packet', 0),
                       full_turns=summary['full_turns'].get('packet', 0),
                       reference_policy=report['reference_policy'], coverage_audit='coverage-audit.json')
        summary_path.write_text(json.dumps(updated, indent=2) + '\n')
    print(json.dumps(dict(arm=run.name, **summary)), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--prepared', type=Path, default=Path('memory_system/data/prepared/personamem-v2-32k.jsonl'))
    args = p.parse_args()
    audit(args.run, args.prepared)
