"""Audit paired frozen listwise results and literal original-turn transport."""
import argparse
import ast
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import audit_https_funnel as audit


def main(args):
    rows = list(audit.records(args.input))
    modes = args.modes.split(',')
    expected = {(str(i), mode) for i in range(26) for mode in modes}
    if len(rows) != len(expected) or {(r['qa_id'], r['protocol']) for r in rows} != expected:
        raise ValueError('Require all 26 questions in every requested arm')
    searches = {s['search_id']: s for s in audit.records(args.run / 'completed/logs/search-debug.jsonl')}
    memory = list(audit.records(args.run / 'completed/logs/memory-debug.jsonl'))
    csv.field_size_limit(32 * 1024 * 1024)
    with args.gold.open() as f:
        gold = list(csv.DictReader(f))
    prepared_path = Path(__file__).resolve().parents[1] / 'data/prepared/personamem-v2-32k.jsonl'
    prepared = {r['conversation_id']: r for r in audit.records(prepared_path)}
    results = list(audit.records(args.run / 'completed/data/results/personamem-v2-32k.jsonl'))
    audit.analyze(results, memory, list(searches.values()), prepared, gold)
    violations = []
    policy_only = {}
    for row in rows:
        s = searches[row['source_search_id']]
        raw = {audit.identity(z): z for event in memory if event['user_id'] == s['user_id']
               for z in event['source_messages']}
        snippet = gold[int(row['qa_id'])]['related_conversation_snippet']
        try:
            snippet = json.loads(snippet)
        except ValueError:
            snippet = ast.literal_eval(snippet)
        turns = [audit.norm(z['content']) for z in snippet if z['role'] == 'user']
        by_id = {c['id']: c for c in s['ranked']}
        for field in ('candidate_ids', 'selected_ids') + (('primary_selected_ids',) if 'primary_selected_ids' in row else ()):
            cards = [dict(z, id=f'{i}:{j}', text=z.get('content', ''))
                     for i, mid in enumerate(row[field])
                     for j, z in enumerate(by_id[mid].get('_packet_item', {}).get('sources', []))]
            row[field + '_full_turns'] = sum(t['full'] for t in audit.turn_coverage(turns, cards, raw))
        if row['qa_id'] not in policy_only:
            old = s['cascade']['listwise']
            retained = set(old['selected_ids']) | set(old['recovery'].get('unreviewed_ids', []))
            cards = [dict(z, id=f'{i}:{j}', text=z.get('content', ''))
                     for i, mid in enumerate(sorted(retained))
                     for j, z in enumerate(by_id[mid].get('_packet_item', {}).get('sources', []))]
            policy_only[row['qa_id']] = dict(
                added_unresolved=len(retained - set(old['selected_ids'])),
                selected_full_turns=sum(t['full'] for t in audit.turn_coverage(turns, cards, raw)))
        review = row['recovery']
        unresolved = set(review['unreviewed_ids'])
        if not unresolved <= set(row['selected_ids']):
            violations.append([row['qa_id'], row['protocol'], 'unresolved_deleted'])
        if not set(row['selected_ids']) <= set(row['candidate_ids']):
            violations.append([row['qa_id'], row['protocol'], 'invented_candidate'])
        row['unresolved_deleted'] = len(unresolved - set(row['selected_ids']))
        row['final_deleted'] = len(row['candidate_ids']) - len(row['selected_ids'])
    for qid in {r['qa_id'] for r in rows}:
        pair = [r for r in rows if r['qa_id'] == qid]
        if any(r['input_hash'] != pair[0]['input_hash'] or r['candidate_ids'] != pair[0]['candidate_ids'] for r in pair[1:]):
            violations.append([qid, 'candidate_input_mismatch'])
    def summarize(group):
        labels = Counter(v['relevance'] for r in group if r['primary_status'] in ('ok','recovered')
                         for v in r.get('accepted_payload', {}).get('candidates', {}).values())
        return dict(relevance_labels=dict(labels), questions=len(group), candidate_count=sum(len(r['candidate_ids']) for r in group),
            dispatched_questions=sum(any(a['dispatched'] for a in r['attempts']) for r in group),
            primary_status=dict(Counter(r['primary_status'] for r in group)),
            first_failed=sum(r['attempts'][0]['status'] != 'ok' for r in group),
            primary_irrelevant=sum(len(r['irrelevant_ids']) for r in group),
            final_deleted=sum(r['final_deleted'] for r in group),
            confirmed_negative=sum(len(r['recovery']['confirmed_irrelevant_ids']) for r in group),
            restored=sum(len(r['recovery']['restored_ids']) for r in group),
            unreviewed=sum(len(r['recovery']['unreviewed_ids']) for r in group),
            unresolved_deleted=sum(r['unresolved_deleted'] for r in group),
            review_error_questions=sum(bool(r['recovery']['errors']) for r in group),
            review_errors=dict(Counter(e['error'] for r in group for e in r['recovery']['errors'])),
            review_calls=sum(len(r['recovery']['batches']) for r in group),
            input_full_turns=sum(r['candidate_ids_full_turns'] for r in group),
            primary_selected_full_turns=sum(r.get('primary_selected_ids_full_turns', 0) for r in group),
            selected_full_turns=sum(r['selected_ids_full_turns'] for r in group),
            mean_seconds=statistics.mean(r['seconds'] for r in group) if group else None,
            total_tokens=sum(r['tokens'] for r in group))
    matched = {qid for qid in {r['qa_id'] for r in rows}
               if all(r['primary_status'] in ('ok', 'recovered') for r in rows if r['qa_id'] == qid)}
    report = dict(summary={mode: summarize([r for r in rows if r['protocol'] == mode])
                           for mode in modes},
        both_primary_valid={mode: summarize([r for r in rows if r['protocol'] == mode and r['qa_id'] in matched])
                            for mode in modes}, violations=violations, rows=rows, historical_policy_only=dict(
            description='Offline union of archived selected IDs and unresolved IDs; no new judgment or repacking.',
            selected_full_turns=sum(r['selected_full_turns'] for r in policy_only.values()),
            added_candidates=sum(r['added_unresolved'] for r in policy_only.values()), rows=policy_only),
        limitations=['Frozen original candidates, CE scores and restored query/graph context; live stochastic models.',
                      'All arms use the same repair/deletion policy. Consult experiment metadata for the protocol or instruction intervention.',
                      'Source coverage is literal complete original turns, not semantic correctness.',
                      'No fresh ingestion, CE execution, final packing or answer scoring in this comparison.',
                      'Review batch count is an upper bound on provider calls; budget skips may not dispatch.'])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ('summary', 'both_primary_valid', 'violations', 'historical_policy_only')}, indent=2))
    if violations:
        raise SystemExit(1)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modes', default='permutation,keyed')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gold', type=Path, default=Path(__file__).resolve().parents[1] / 'data/raw/personamem-v2.csv')
    main(p.parse_args())
