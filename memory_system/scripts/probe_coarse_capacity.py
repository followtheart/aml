"""Offline coarse-capacity upper bounds; preserve the recorded baseline prefix."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
from unittest.mock import patch

from evaluate_recall_context import CSV, packet_coverage, records, reference_turns
from app import cascade_rerank, config


def main(args):
    traces = {t['search_id']: t for t in records(args.run/'server-search-debug.jsonl')}
    conversations = {r['conversation_id']: r for r in records(args.prepared)}
    csv.field_size_limit(32*1024*1024)
    with CSV.open() as source:
        gold = list(csv.DictReader(source))
    raw = {}
    with sqlite3.connect((args.run/'memory.db').resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(row['user_id'], {})[row['request_id'], row['message_index']] = dict(row)
    rows = []
    for result in records(args.run/'completed/data/results/personamem-v2-32k.jsonl'):
        t = traces[result['search_id']]
        candidates = [c for c in t['ranked'] if not cascade_rerank.protected(c)]
        requirements = t['coverage_manifest']['coverage']['requirements']
        limit = t['cascade']['coarse'].get('baseline_limit',t['cascade']['coarse']['limit'])
        key = lambda c: (c.get('_fused', 0),)
        with patch.multiple(config, **t['versions']['settings']), patch('socket.socket.connect', side_effect=AssertionError('Offline')):
            baseline = cascade_rerank.shortlist(candidates, limit, requirements, key)
            if [c['id'] for c in baseline] != t['cascade']['coarse'].get('baseline_ids',t['cascade']['coarse']['selected_ids']):
                raise ValueError('Recorded coarse baseline did not reproduce: '+result['qa_id'])
            baseline_ids = {c['id'] for c in baseline}
            proposals = {0: baseline}
            for extra in args.extra:
                wider = cascade_rerank.shortlist(candidates, limit+extra, requirements, key)
                proposals[extra] = baseline + [c for c in wider if c['id'] not in baseline_ids][:extra]
                assert proposals[extra][:len(baseline)] == baseline
            if 8 in proposals:
                with patch.object(config, 'COARSE_EVIDENCE_EXTEND', True):
                    implemented, _ = cascade_rerank.coarse_shortlist(candidates, requirements)
                assert [c['id'] for c in implemented] == [c['id'] for c in proposals[8]]
                if t['versions']['settings'].get('COARSE_EVIDENCE_EXTEND'):
                    assert [c['id'] for c in implemented] == t['cascade']['coarse']['selected_ids']
        conv = conversations[result['conversation_id']]
        q = next(q for q in conv['qa'] if q['id'] == result['qa_id'])
        assert t['query'] == q['question'] and t['options'] == q['options']
        turns = reference_turns(q, conv['source_history'], gold[int(result['qa_id'])])
        choices = {}
        for extra, proposal in proposals.items():
            choices[str(extra)] = dict(ids=[c['id'] for c in proposal],
                document_bytes=sum(len(c.get('_rank_text', c.get('content', '')).encode()) for c in proposal),
                coverage=packet_coverage(turns, [c['_packet_item'] for c in proposal], raw[t['user_id']]))
        rows.append(dict(qa_id=result['qa_id'], choices=choices))
    summary = dict(questions=len(rows), reproduced_baselines=len(rows), variants={})
    for extra in [0, *args.extra]:
        k = str(extra)
        summary['variants'][k] = dict(
            candidates=sum(len(r['choices'][k]['ids']) for r in rows),
            document_bytes=sum(r['choices'][k]['document_bytes'] for r in rows),
            visible=sum(r['choices'][k]['coverage']['visible'] for r in rows),
            turns=sum(r['choices'][k]['coverage']['full_turns'] for r in rows),
            gained=[r['qa_id'] for r in rows if r['choices'][k]['coverage']['visible'] and not r['choices']['0']['coverage']['visible']])
    with args.output.open('x') as out:
        json.dump(dict(summary=summary, rows=rows,
            trace_sha256=hashlib.sha256((args.run/'server-search-debug.jsonl').read_bytes()).hexdigest(),
            limitation='Pre-CE upper bound only; no new scores, budget admission, final packing or answer evaluation. Document bytes exclude query/schema overhead.'), out, indent=2)
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--prepared', type=Path, default=Path('memory_system/data/prepared/personamem-v2-32k.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extra', type=int, nargs='+', default=[4, 8, 16, 24])
    args = p.parse_args()
    if any(x <= 0 for x in args.extra) or len(args.extra) != len(set(args.extra)):
        p.error('Extra candidate counts must be positive and unique')
    main(args)
