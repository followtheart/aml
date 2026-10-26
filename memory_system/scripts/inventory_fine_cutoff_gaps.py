"""Inventory source-novel candidates skipped inside the effective fine CE cutoff.

Read-only diagnosis on a complete HTTPS arm. Eligibility is fixed before reading
reference labels. This does not replay a changed selector, admission or answers.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def inventory(args):
    run = args.run.resolve()
    root = run.parent
    plan = read(root / 'plan.json')
    final = read(root / 'final-evaluation.json')
    assert final['status'] == 'verified_complete_trial'
    snapshot = root / 'source/memory_system'
    for name, digest in plan['source_files'].items():
        assert sha(root / 'source' / name) == digest, name
    sys.path.insert(0, str(snapshot))
    sys.path.insert(0, str(snapshot / 'scripts'))
    from app import fine_evidence
    import audit_https_funnel as funnel
    from evaluate_recall_context import reference_turns

    manifest = read(run / 'isolated-manifest.json')
    assert manifest['client_exit_code'] == manifest['audit_exit_code'] == 0
    assert manifest['server_exit_code'] in (0, -15)
    prepared = Path(manifest['data'])
    assert sha(prepared) == manifest['data_sha256']
    inputs = [root / 'final-evaluation.json', root / 'plan.json',
              run / 'isolated-manifest.json', run / 'server-search-debug.jsonl',
              run / 'completed/data/results/personamem-v2-32k.jsonl',
              run / 'funnel/stage-transport.json', prepared, args.gold_csv,
              Path(__file__), Path(fine_evidence.__file__)]
    for path in inputs:
        if str(path) in final['inputs']:
            assert sha(path) == final['inputs'][str(path)], str(path)
    results = records(inputs[4])
    traces = {t['search_id']: t for t in records(inputs[3])}
    stages = {r['qa_id']: r for r in read(inputs[5])['rows']}
    assert len(results) == len(traces) == len(stages) == plan['questions']
    fixed = []
    for result in results:
        t = traces[result['search_id']]
        assert not result.get('error_stage') and result['packet_hash'] == t['packet_hash']
        assert result['user_id'] == t['user_id']
        lookup = {c['id']: c for c in t['ranked']}
        cascade = t['cascade']
        coarse = sorted([lookup[mid] for mid in cascade['coarse']['selected_ids']],
                        key=fine_evidence.key, reverse=True)
        fine = [lookup[mid] for mid in cascade['fine']['selected_ids']]
        adaptive = cascade['fine']['selection']['adaptive']
        capacity = adaptive['capacity']
        fine_ids = {c['id'] for c in fine}
        visible = {c['id']: fine_evidence.selection.fragments(c) for c in coarse}
        eligible = []
        for rank, candidate in enumerate(coarse[:capacity], 1):
            mid = candidate['id']
            score = candidate.get('_ce_score')
            if (mid in fine_ids or type(score) not in (int, float)
                    or not math.isfinite(score) or not visible[mid]):
                continue
            group, missing = fine_evidence.closure(candidate, lookup)
            if missing or len(group - fine_ids) != 1:
                continue
            if fine_evidence.selection.represented_by(candidate, fine, visible):
                continue
            eligible.append(dict(id=mid, ce_rank=rank, ce_score=score,
                in_core_pool=mid in adaptive['core_pool_ids']))
        fixed.append((result, t, coarse, fine, eligible))

    # Only attribution below uses reference labels; no selection is changed.
    csv.field_size_limit(32 * 1024 * 1024)
    with args.gold_csv.open() as stream:
        gold = list(csv.DictReader(stream))
    conversations = {c['conversation_id']: c for c in records(prepared)}
    raw = {}
    with sqlite3.connect((run / 'memory.db').as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for source in db.execute('SELECT * FROM source_messages'):
            source = dict(source)
            raw.setdefault(source['user_id'], {})[funnel.identity(source)] = source
    rows = []
    for result, t, coarse, fine, eligible in fixed:
        qa = result['qa_id']
        conversation = conversations[result['conversation_id']]
        question = next(q for q in conversation['qa'] if q['id'] == qa)
        assert question['question'] == t['query'] and question['options'] == t['options']
        turns = reference_turns(question, conversation['source_history'], gold[int(qa)])
        def coverage(candidates):
            cards = [dict(s, text=s.get('content', ''), id=f'{i}:{j}')
                     for i, c in enumerate(candidates)
                     for j, s in enumerate(c['_packet_item'].get('sources', []))]
            return [v['full'] for v in funnel.turn_coverage(turns, cards, raw[t['user_id']])]
        stage = stages[qa]
        original = coverage(fine)
        assert original == [v['full'] for v in stage['visible']['fine']['turn_coverage']]
        coarse_coverage = coverage(coarse)
        assert coarse_coverage == [v['full'] for v in stage['visible']['coarse']['turn_coverage']]
        lookup = {c['id']: c for c in coarse}
        for candidate in eligible:
            candidate['reference_turns'] = coverage([lookup[candidate['id']]])
        union = coverage(fine + [lookup[eligible[0]['id']]]) if eligible else original
        rows.append(dict(qa_id=qa, ce_status=t['cascade']['cross_encoder']['status'],
            fine_size=len(fine), fine_capacity=t['cascade']['fine']['selection']['adaptive']['capacity'],
            original_fine_reference_turns=original, coarse_reference_turns=coarse_coverage,
            eligible=eligible, first_candidate_union_reference_turns=union,
            catalog_reference_turns=[v['full'] for v in stage['visible']['catalog']['turn_coverage']]))
    summary = dict(questions=len(rows), provider_calls=0,
        eligible_questions=sum(bool(r['eligible']) for r in rows),
        fine_gap_questions=[r['qa_id'] for r in rows if any(r['coarse_reference_turns'])
                            and not any(r['original_fine_reference_turns'])],
        first_nominee_potential_new_fine_questions=[r['qa_id'] for r in rows
            if any(r['first_candidate_union_reference_turns']) and not any(r['original_fine_reference_turns'])],
        first_nominee_potential_new_fine_turn_questions=[r['qa_id'] for r in rows if any(
            a and not b for a, b in zip(r['first_candidate_union_reference_turns'], r['original_fine_reference_turns']))],
        any_eligible_reference_in_fine_gap=[r['qa_id'] for r in rows if not any(r['original_fine_reference_turns'])
            and any(any(c['reference_turns']) for c in r['eligible'])])
    return dict(summary=summary, rows=rows, limitation=__doc__ +
        '\nThe diagnostic union can exceed fine capacity: it only measures possible coverage, not a valid selector or adoption evidence.',
        inputs={str(p.resolve()): sha(p) for p in inputs})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/raw/personamem-v2.csv')
    args = parser.parse_args()
    with patch('socket.socket.connect', side_effect=AssertionError('Offline inventory')):
        report = inventory(args)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report['summary']))
