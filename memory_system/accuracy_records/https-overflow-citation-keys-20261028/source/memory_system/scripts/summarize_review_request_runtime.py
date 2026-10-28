"""Audit packet identity and upstream drift in full-reader review experiments."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from replay_source_vector_route import source_route
from app import evidence_packet
from summarize_reader_replay import read_complete


def before_review(row):
    calls = row['diagnostics'].get('answer_calls', [])
    return [{k: c.get(k) for k in ('stage', 'prompt_sha256', 'response', 'status')}
            for c in calls if c['stage'] != 'eval.choice_select_review']


def before_selection(row):
    prefix = []
    for call in row['diagnostics'].get('answer_calls', []):
        if call['stage'] == 'eval.choice_select':
            break
        prefix.append({k: call.get(k) for k in ('stage', 'prompt_sha256', 'response', 'status')})
    return prefix


def first_upstream_difference(a, b):
    left, right = before_review(a), before_review(b)
    for index in range(max(len(left), len(right))):
        x = left[index] if index < len(left) else None
        y = right[index] if index < len(right) else None
        if x != y:
            return dict(index=index, control_stage=x['stage'] if x else None,
                        treatment_stage=y['stage'] if y else None,
                        same_prompt=bool(x and y and x['prompt_sha256'] == y['prompt_sha256']))
    return None


def main(args):
    plan = json.loads(args.replay.with_suffix('.plan.json').read_text())
    run = Path(plan['run'])
    baseline = read_complete(run/'completed/data/results/personamem-v2-32k.jsonl')
    if [r['qa_id'] for r in baseline] != plan['questions']:
        raise ValueError('Cohort differs from preregistered plan')
    traces = {t['search_id']: t for t in read_complete(run/'server-search-debug.jsonl')}
    probe = json.loads(Path(plan['probe']).read_text()) if plan['probe'] else None
    if probe:
        digest = hashlib.sha256(json.dumps([s['content'] for s in probe['sources']], ensure_ascii=False).encode()).hexdigest()
        if digest != probe['source_text_sha256']:
            raise ValueError('Source universe changed')
    routes = {q['qa_id']: q for q in probe['questions']} if probe else {}
    if probe and set(routes) != set(plan['questions']):
        raise ValueError('Incomplete route cohort')
    expected = {}
    for r in baseline:
        t = traces[r['search_id']]
        packet = t['returned']
        if evidence_packet.digest(packet) != r['packet_hash']:
            raise ValueError('Baseline packet changed')
        if probe:
            q = routes[r['qa_id']]
            if q['search_id'] != r['search_id'] or q['query_specs'] != t['query_specs'] or t['user_id'] != probe['user_id']:
                raise ValueError('Route/query/user mismatch')
            packet, _ = source_route(probe, q, packet, t['evidence_token_budget'])
        expected[r['qa_id']] = (r['conversation_id'], r['packet_hash'], evidence_packet.digest(packet))
    pairs = defaultdict(dict)
    metadata_mismatches = []
    errors = []
    call_errors = []
    for r in read_complete(args.replay, args.allow_partial):
        qid, rep, enabled = r['qa_id'], r['repetition'], r['enabled']
        if qid not in expected or type(enabled) is not bool or not 1 <= rep <= plan['repeat']:
            raise ValueError('Unexpected question, arm or repetition')
        if (r['conversation_id'], r['original_packet_hash'], r['packet_hash']) != expected[qid]:
            raise ValueError('Packet identity mismatch')
        if r['query_context'] != plan['query_context']:
            raise ValueError('Context assignment mismatch')
        if enabled in pairs[qid, rep]:
            raise ValueError('Duplicate arm')
        settings = r['versions']['settings']
        if settings[plan['setting']] is not enabled or settings['CHOICE_QUERY_CONTEXT'] != plan['query_context']:
            metadata_mismatches.append(dict(qa_id=qid, enabled=enabled, repetition=rep))
        if r['diagnostics'].get('answer_status') == 'error' or r['diagnostics'].get('error_stage'):
            errors.append(dict(qa_id=qid, enabled=enabled, repetition=rep))
        for c in r['diagnostics'].get('answer_calls', []):
            if c.get('status') == 'error':
                call_errors.append(dict(qa_id=qid, enabled=enabled, repetition=rep,
                                        stage=c['stage'], final_answer_status=r['diagnostics'].get('answer_status')))
        pairs[qid, rep][enabled] = r
    results = []
    for (qid, rep), arms in pairs.items():
        if set(arms) != {False, True}:
            continue
        a, b = arms[False], arms[True]
        normalization = {
            name: row['diagnostics'].get('answer_scope_normalization', {}).get('removed', [])
            for name, row in (('control', a), ('treatment', b))
        }
        results.append(dict(qa_id=qid, repetition=rep, control=a['score'], treatment=b['score'],
                            control_prediction=a['prediction'], treatment_prediction=b['prediction'],
                            pre_selection_identical=before_selection(a) == before_selection(b),
                            eligible_identical=a['diagnostics'].get('answer_eligible_options') == b['diagnostics'].get('answer_eligible_options'),
                            treatment_notes_present=bool(b['diagnostics'].get('answer_selection_notes')),
                            upstream_identical=before_review(a) == before_review(b),
                            first_upstream_difference=first_upstream_difference(a, b),
                            scope_normalization=normalization,
                            scope_normalization_changed=normalization['control'] != normalization['treatment'],
                            control_review=a['diagnostics'].get('answer_selection_review', {}).get('status'),
                            treatment_review=b['diagnostics'].get('answer_selection_review', {}).get('status')))
    count = sum(len(a) for a in pairs.values())
    partial = count != len(expected)*plan['repeat']*2
    if partial and not args.allow_partial:
        raise ValueError('Incomplete experiment')
    report = dict(partial=partial, rows=count, complete_pairs=len(results),
                  control=sum(r['control'] for r in results), treatment=sum(r['treatment'] for r in results),
                  wins=[r['qa_id'] for r in results if r['treatment'] > r['control']],
                  losses=[r['qa_id'] for r in results if r['treatment'] < r['control']],
                  scope_normalization_changed=[r['qa_id'] for r in results if r['scope_normalization_changed']],
                  metadata_mismatches=metadata_mismatches, errors=errors, call_errors=call_errors, pairs=results,
                  setting=plan['setting'],
                  caveat='Packet identities validated. All reader calls live. Upstream drift fields compare all calls except final review; for non-review interventions these may reflect the intervention itself. '
                         'Scope-normalization differences are observed removals, not causal attribution; changed upstream claims can also change removals. '
                         'Metadata mismatches are retained explicitly, never repaired in archived rows. Not fresh HTTPS accuracy.')
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs','metadata_mismatches','caveat')}))
    print(f'cached_metadata_mismatches={len(metadata_mismatches)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true')
    main(p.parse_args())
