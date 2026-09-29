"""Summarize one archived experiment without calling models or changing logs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    if not path.exists():
        return [], None
    raw = path.read_bytes()
    return [json.loads(line) for line in raw.splitlines() if line.strip()], hashlib.sha256(raw).hexdigest()


def summarize(root, prepared):
    paths = ['logs/memory-debug.jsonl', 'logs/search-debug.jsonl',
             'data/results/personamem-v2-32k.jsonl']
    loaded = [read(root / p) for p in paths]
    memory, searches, results = [rows for rows, _ in loaded]
    questions = {(c['conversation_id'], q['id']): q
                 for c in read(prepared)[0] for q in c['qa']}
    search_by_id = {s['search_id']: s for s in searches}
    anomalies, exits, calls, statuses, rows = [], Counter(), Counter(), Counter(), []
    support_lengths, initial_support_lengths, support_failures = Counter(), Counter(), Counter()
    all_attempt_calls, adaptive_status = Counter(), Counter()
    group_status, packet_omissions = Counter(), Counter()
    recovery_status, recovery_counts = Counter(), Counter()
    scope_status, scope_counts = Counter(), Counter()
    result_search_ids = {r.get('search_id') for r in results}
    unmatched_searches = [dict(search_id=s['search_id'], query=s.get('query'), fake=s.get('fake'))
                         for s in searches if s['search_id'] not in result_search_ids]
    anomalies.extend(f'missing_input:{p}' for p, (_, digest) in zip(paths, loaded)
                     if digest is None)
    if len(search_by_id) != len(searches):
        anomalies.append('duplicate_search_ids')
    seen = set()
    for result in results:
        key = result['conversation_id'], result['qa_id']
        if key in seen:
            anomalies.append(f'duplicate_result:{key}')
        seen.add(key)
        qa = questions[key]
        gold = qa['gold_labels']
        predicted = result['prediction']
        adaptive_status[result.get('answer_adaptive_retry', {}).get('status', 'disabled')] += 1
        attempts = result.get('answer_attempts')
        all_calls = ([c for a in attempts for c in a['diagnostics'].get('answer_calls', [])]
                     if attempts else result.get('answer_calls', []))
        all_attempt_calls.update(f"{c['stage']}:{c.get('status', 'unknown')}" for c in all_calls)
        if attempts:
            selected = result['answer_adaptive_retry']['selected_attempt']
            if attempts[selected].get('prediction') != predicted:
                anomalies.append(f'adaptive_prediction_mismatch:{key}')
            if attempts[selected]['diagnostics'].get('choice_alignment') != result.get('choice_alignment'):
                anomalies.append(f'adaptive_alignment_mismatch:{key}')
        if result['score'] != float(predicted in gold):
            anomalies.append(f'score_mismatch:{key}')
        search = search_by_id.get(result.get('search_id'))
        if not search or search.get('packet_hash') != result.get('packet_hash'):
            anomalies.append(f'search_packet_missing_or_mismatch:{key}')
        elif search.get('query') != qa['question'] or search.get('options') != qa['options']:
            anomalies.append(f'search_question_mismatch:{key}')
        if search:
            manifest = search.get('coverage_manifest', {})
            included = set(manifest.get('included_ids', []))
            for group in manifest.get('requested_evidence_groups', []):
                retained = set(group) & included
                state = 'included' if retained == set(group) else 'omitted' if not retained else 'partial'
                group_status[state] += 1
                if state == 'partial':
                    anomalies.append(f'atomic_group_split:{key}:{group}')
            packet_omissions.update(o.get('reason', 'unknown') for o in manifest.get('omitted', []))
        eligible = result.get('answer_eligible_options', [])
        exit_kind = ('correct' if result['score'] else 'error' if result.get('error_type')
                     else 'abstained' if predicted in ('', 'ABSTAIN')
                     else 'gold_excluded' if not set(gold).intersection(eligible)
                     else 'selection_wrong')
        exits[exit_kind] += 1
        support = result.get('answer_support_validation', {})
        statuses[support.get('status', 'unrecorded')] += 1
        recovery = result.get('answer_citation_recovery', {})
        recovery_status[recovery.get('status', 'not_recorded')] += 1
        requested = recovery.get('requested', recovery.get('claim_ids', []))
        attached = recovery.get('attached', [])
        recovery_counts.update(requested_claims=len(requested), attached_claims=len(attached),
                               rejected_proposals=len(recovery.get('rejected', [])))
        recovered_claims = [(e['letter'], c) for e in result.get('choice_alignment', [])
                            for c in e.get('claims', []) if c.get('citation_recovery')]
        recovery_counts['verified_claims'] += sum(c['status'] == 'supported' for _, c in recovered_claims)
        recovery_counts['normalized_quotes'] += sum(bool(ref.get('quote_normalization'))
            for _, c in recovered_claims for ref in c.get('citations', []))
        gold_attached = any(cid.split(':')[0] in gold for cid in attached)
        gold_verified = any(letter in gold and c['status'] == 'supported' for letter, c in recovered_claims)
        recovery_counts['gold_attached_questions'] += gold_attached
        recovery_counts['gold_verified_questions'] += gold_verified
        recovery_counts['gold_verified_and_eligible_questions'] += gold_verified and bool(set(gold) & set(eligible))
        recovery_counts['gold_verified_and_selected_questions'] += gold_verified and bool(result['score'])
        scope = result.get('answer_premise_scope', {})
        scope_status[scope.get('status', 'not_recorded')] += 1
        scope_counts['requested_claims'] += len(scope.get('requested', []))
        scope_counts['removed_claims'] += len(scope.get('removed', []))
        changed = [e for e in result.get('choice_alignment', []) if e.get('scope_reclassification')]
        changed_letters = {e['letter'] for e in changed}
        scope_counts['changed_options'] += len(changed)
        scope_counts['changed_generic_options'] += sum(e['kind'] == 'generic' for e in changed)
        scope_counts['changed_verified_options'] += sum(e.get('entailment_verified') is True for e in changed)
        scope_counts['changed_eligible_options'] += len(changed_letters & set(eligible))
        scope_counts['changed_selected_questions'] += predicted in changed_letters
        scope_counts['gold_changed_questions'] += bool(changed_letters & set(gold))
        scope_counts['gold_changed_and_eligible_questions'] += bool(changed_letters & set(gold) & set(eligible))
        scope_counts['gold_changed_and_selected_questions'] += bool(changed_letters & set(gold)) and bool(result['score'])
        for e in changed:
            for removed in e.get('removed_claims', []):
                if removed.get('reason') != 'scope_not_personal':
                    continue
                original = removed.get('original_claim', {})
                if (original.get('status') != 'unsupported' or original.get('citations')
                        or original.get('dropped_citations') or original.get('validation_errors')
                        or original.get('reason') not in ('no_source', 'source_too_weak', 'none')):
                    anomalies.append(f'scope_removed_protected_claim:{key}:{e["letter"]}')
        for call in result.get('answer_calls', []):
            calls[f"{call['stage']}:{call.get('status', 'unknown')}"] += 1
            if call['stage'].startswith('eval.choice_support'):
                response = call.get('response')
                if isinstance(response, dict) and isinstance(response.get('options'), (list, dict)):
                    count = len(response['options'])
                else:
                    count = 'no_options_list'
                support_lengths[count] += 1
                if call['stage'] == 'eval.choice_support':
                    initial_support_lengths[count] += 1
                detail = str(call.get('error_detail', ''))
                for reason, marker in (
                    ('missing_or_duplicate_option', 'Expected this option exactly once'),
                    ('claim_not_in_option', 'claim_not_in_option'),
                    ('quote_not_in_source', 'quote_not_in_source'),
                    ('no_valid_user_support', 'no_valid_user_support')):
                    if marker in detail:
                        support_failures[reason] += 1
        rows.append(dict(qa_id=key[1], gold=gold, prediction=predicted,
                         score=result['score'], exit=exit_kind, eligible=eligible,
                         unresolved=support.get('unresolved_options', [])))
    memories = {m['id']: m for event in memory for m in event['memories']}
    versions = sorted({r['versions']['pipeline_version'] for r in results})
    if len(versions) > 1:
        anomalies.append('mixed_pipeline_versions')
    return dict(input_hashes={p: digest for p, (_, digest) in zip(paths, loaded)},
                prepared_sha256=read(prepared)[1], pipeline_versions=versions,
                add_requests=len(memory), source_messages=sum(len(e['source_messages']) for e in memory),
                memory_types=dict(Counter(m['type'] for m in memories.values())),
                searches=len(searches), results=len(results),
                matched_searches=sum(s['search_id'] in result_search_ids for s in searches),
                unmatched_searches=unmatched_searches,
                atomic_groups=dict(group_status), packet_omissions=dict(packet_omissions),
                score=sum(r['score'] for r in results),
                accuracy=sum(r['score'] for r in results) / len(results) if results else None,
                exits=dict(exits), support_status=dict(statuses), answer_calls=dict(calls),
                answer_calls_scope='chosen_attempt', all_attempt_calls=dict(all_attempt_calls),
                adaptive_retry_status=dict(adaptive_status),
                citation_recovery_status=dict(recovery_status), citation_recovery_counts=dict(recovery_counts),
                premise_scope_status=dict(scope_status), premise_scope_counts=dict(scope_counts),
                support_response_option_counts=dict(support_lengths),
                initial_support_response_option_counts=dict(initial_support_lengths),
                support_failure_call_counts=dict(support_failures),
                gold_eligible=sum(bool(set(q['gold']).intersection(q['eligible'])) for q in rows),
                anomalies=anomalies, questions=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--prepared', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/prepared/personamem-v2-32k.jsonl')
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.prepared), ensure_ascii=False, indent=2))
