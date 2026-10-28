"""Offline paired fine selection on archived candidates and CE scores.

Reproduces the archived control before comparing optional topic propagation.
Gold stays in the scorer. No providers; no claims about final packet or accuracy.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
from unittest.mock import patch

from evaluate_recall_context import CSV, packet_coverage, records, reference_turns
from app import config, fine_evidence, retrieval_queries


def semantic_seats(coarse, baseline, trace, seats):
    """Diagnostic alternative only: share extra seats with nearby CE evidence.

    Core and its complete recorded dependencies come first. Select at most
    `seats` additional units by CE inside the existing total capacity window,
    then fill from the baseline's optional groups in their original order.
    This function is NOT used by the production search pipeline.
    """
    if not seats:
        return list(baseline)
    by_id = {c['id']: c for c in coarse}
    ordered = sorted(coarse, key=fine_evidence.key, reverse=True)
    visible = {c['id']: fine_evidence.selection.fragments(c) for c in coarse}
    capacity = trace['adaptive']['capacity']
    core = set(trace['adaptive']['core_ids'])
    selected = [by_id[mid] for mid in trace['adaptive']['core_ids']]

    def admit(candidate, max_new=None):
        ids = {c['id'] for c in selected}
        group, missing = fine_evidence.closure(candidate, by_id)
        new = group - ids
        if missing or not new or len(selected) + len(new) > capacity:
            return 0
        if max_new is not None and len(new) > max_new:
            return 0
        selected.extend(c for c in ordered if c['id'] in new)
        return len(new)

    for c in list(selected):
        group, missing = fine_evidence.closure(c, by_id)
        if missing or not group <= {r['id'] for r in baseline}:
            raise ValueError('Diagnostic requires complete baseline core dependencies')
        admit(c)
    used = 0
    for c in ordered[:capacity]:
        if used >= seats:
            break
        if c['id'] in core or c.get('_ce_score') is None or not visible[c['id']]:
            continue
        if fine_evidence.selection.represented_by(c, selected, visible):
            continue
        used += admit(c, seats-used)
    for a in trace['adaptive']['additions']:
        c = by_id[a['candidate_id']]
        if not fine_evidence.selection.represented_by(c, selected, visible):
            admit(c)
    for c in ordered[:capacity]:
        if c.get('_ce_score') is None or not visible[c['id']]:
            continue
        if not fine_evidence.selection.represented_by(c, selected, visible):
            admit(c)
    assert core <= {c['id'] for c in selected}
    assert len(selected) <= capacity
    return selected


def main(args):
    trace_path = args.run / 'search-debug.jsonl'
    traces = {t['search_id']: t for t in records(trace_path)}
    conversations = {d['conversation_id']: d for d in records(args.prepared)}
    csv.field_size_limit(32 * 1024 * 1024)
    with CSV.open() as source:
        gold = list(csv.DictReader(source))
    raw = {}
    with sqlite3.connect((args.run / 'memory.db').resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(row['user_id'], {})[row['request_id'], row['message_index']] = dict(row)
    rows = []
    for result in records(args.run / 'results.jsonl'):
        t = traces[result['search_id']]
        if t['cascade']['cross_encoder']['status'] != 'ok':
            raise ValueError('Partial CE needs a separate replay protocol')
        by_id = {c['id']: copy.deepcopy(c) for c in t['ranked']}
        scores = {c['id']: c['score'] for c in t['cascade']['fine']['selection']['decisions']}
        coarse = [by_id[mid] for mid in t['cascade']['coarse']['selected_ids']]
        for c in coarse:
            c['_ce_score'] = scores[c['id']]
        settings = {k: v for k, v in t['versions']['settings'].items() if hasattr(config, k)}
        with patch.multiple(config, **settings), patch('socket.socket.connect', side_effect=AssertionError('Offline only')):
            plan = copy.deepcopy(t['plan'])
            specs = retrieval_queries.build(t['query'], t['options'], plan)
            if specs != t['query_specs']:
                raise ValueError('Query reconstruction differs from archive')
            requirements = plan['_coverage_requirements']
            fused = {c['id'] for c in sorted(coarse, key=lambda c: -(c.get('_fused') or 0))[:config.CASCADE_FUSED_RESERVE]}
            kw = dict(fused_ids=fused, extra_limit=config.FINE_EXTRA_LIMIT)
            control, old = fine_evidence.propose(coarse, config.CASCADE_FINE_LIMIT, requirements, **kw)
            if [c['id'] for c in control] != t['cascade']['fine']['selected_ids']:
                raise ValueError(f'Control reproduction mismatch: {result["qa_id"]}')
            if args.semantic_seats is None:
                treatment, new = fine_evidence.propose(coarse, config.CASCADE_FINE_LIMIT, requirements,
                    context_queries=t['query_specs'], **kw)
                assert old['adaptive']['core_ids'] == new['adaptive']['core_ids']
                assert old['reservations'] == new['reservations']
            else:
                treatment = semantic_seats(coarse, control, old, args.semantic_seats)
                new = old
        conv = conversations[result['conversation_id']]
        q = next(q for q in conv['qa'] if q['id'] == result['qa_id'])
        assert t['query'] == q['question'] and t['options'] == q['options']
        turns = reference_turns(q, conv['source_history'], gold[int(result['qa_id'])])
        def coverage(candidates):
            return packet_coverage(turns, [c['_packet_item'] for c in candidates], raw[t['user_id']])
        rows.append(dict(qa_id=result['qa_id'], conversation_id=result['conversation_id'],
            control=coverage(control), treatment=coverage(treatment),
            control_ids=[c['id'] for c in control], treatment_ids=[c['id'] for c in treatment],
            context_facets=new['adaptive']['context_facets']))
    summary = dict(questions=len(rows), reproduced_controls=len(rows),
        reference_present=sum(r['control']['reference_present'] for r in rows),
        visible={a: sum(r[a]['visible'] for r in rows) for a in ('control', 'treatment')},
        full_turns={a: sum(r[a]['full_turns'] for r in rows) for a in ('control', 'treatment')},
        gained=[r['qa_id'] for r in rows if r['treatment']['visible'] and not r['control']['visible']],
        lost=[r['qa_id'] for r in rows if r['control']['visible'] and not r['treatment']['visible']],
        changed=sum(r['control_ids'] != r['treatment_ids'] for r in rows),
        candidate_set_changed=sum(set(r['control_ids']) != set(r['treatment_ids']) for r in rows),
        candidate_counts={a: sum(len(r[a+'_ids']) for r in rows) for a in ('control', 'treatment')})
    report = dict(summary=summary, rows=rows, limitation=__doc__,
        policy='topic_context' if args.semantic_seats is None else 'diagnostic_semantic_seats',
        semantic_seats=args.semantic_seats,
        trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        prepared_sha256=hashlib.sha256(args.prepared.read_bytes()).hexdigest())
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--prepared', type=Path, default=Path('memory_system/data/prepared/personamem-v2-32k.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--semantic-seats', type=int, choices=(0, 1, 2, 4, 8),
                   help='Diagnostic alternative: reserve extra seats for nearby CE sources; not production code')
    main(p.parse_args())
