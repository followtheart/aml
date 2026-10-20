"""Frozen-score capacity/policy experiments with exact archived baseline checks.

Never invokes providers. Expanded inputs cannot predict LLM decisions or answers.
Historical logs lack settled token headroom: byte-bounded counterfactual admission
is an upper bound, not a claim that the live budget can fund the request.
"""
import argparse
import ast
import copy
import csv
import json
import hashlib
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import cascade_rerank as ranking, config, evidence_selection, fine_evidence
import audit_https_funnel as audit

SOURCE_HASHES = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [Path(__file__), ROOT / 'app/fine_evidence.py',
                           ROOT / 'app/evidence_selection.py', ROOT / 'app/cascade_rerank.py']}


def coarse_rows(search):
    by_id = {r['id']: copy.deepcopy(r) for r in search['ranked']}
    scores = {r['id']: r['score'] for r in search['cascade']['fine']['selection']['decisions']}
    rows = [by_id[mid] for mid in search['cascade']['coarse']['selected_ids']]
    for row in rows:
        row['_ce_score'] = scores[row['id']]
    return rows


def replay(search, fine_limit, llm_limit, policy='archived'):
    trace = search['cascade']
    coarse = coarse_rows(search)
    requirements = [{k: r[k] for k in ('id', 'text', 'origin')}
                    for r in search['coverage_manifest']['coverage']['requirements']]
    selection_trace = trace['fine']['selection']
    reservations = (selection_trace['reservations'] +
                    selection_trace.get('soft_reservation_policy', {}).get('suppressed', []))
    def members(reason):
        return {r['candidate_id'] for r in reservations if reason in r['reasons']}
    if policy == 'archived':
        settings = search['versions']['settings']
        if settings.get('FINE_EVIDENCE_ADAPTIVE'):
            policy = 'adaptive_shared' if settings.get('LISTWISE_SHARED_SOURCES') else 'adaptive'
            fine_limit += settings['FINE_EXTRA_LIMIT']
            llm_limit += settings['FINE_EXTRA_LIMIT']
        else:
            policy = 'bounded' if 'soft_reservation_policy' in selection_trace else 'legacy'
    if policy not in ('legacy', 'bounded', 'bounded_unshared', 'no_window', 'adaptive', 'adaptive_shared'):
        raise ValueError(f'Unknown policy: {policy}')
    adaptive = policy.startswith('adaptive')
    selection = {}
    fine = evidence_selection.select(coarse, fine_limit, requirements,
        lambda c: (c['_ce_score'] if c['_ce_score'] is not None else -1e30, c.get('_fused', 0)),
        reserve_ids=members('unscored_rescue'), fused_ids=members('fused_head'), trace=selection,
        bound_soft_reservations=policy != 'legacy', soft_rank_window=policy != 'no_window')
    if adaptive:
        core_limit = min(fine_limit, search['versions']['settings']['CASCADE_FINE_LIMIT'])
        fine, selection = fine_evidence.propose(coarse, core_limit, requirements,
            rescue_ids=members('unscored_rescue'), fused_ids=members('fused_head'),
            extra_limit=max(0, fine_limit - core_limit))
    reservations = [r for r in selection['reservations'] if r['selected']]
    carried = {r['candidate_id'] for r in reservations}
    soft = {r['candidate_id'] for r in reservations if r['reasons'] == ['fused_head']}
    positions = {r['id']: len(fine) - i for i, r in enumerate(fine)}
    ordered = ranking.shortlist(fine, llm_limit, [],
        lambda c: (c['id'] in carried and c['id'] not in soft, positions[c['id']]), carry_ids=carried)
    plan = copy.deepcopy(search['plan'])
    plan.update(_coverage_requirements=requirements, _fusion_triples=search['fusion_triples'],
                _fusion_edges=search['fusion_edges'])
    if adaptive:
        plan['_fine_path_groups'] = selection['adaptive']['path_groups']
    req = SimpleNamespace(query=search['query'], options=search['options'])
    settings = dict(search['versions']['settings'])
    if policy in ('adaptive', 'bounded_unshared'):
        settings['LISTWISE_SHARED_SOURCES'] = False
    token_headroom = trace.get('admission_budget', {}).get('token_headroom')
    submitted, rejected = [], []
    with ExitStack() as stack:
        stack.enter_context(patch.object(config, 'LISTWISE_KEYED_OUTPUT',
                                         settings.get('LISTWISE_KEYED_OUTPUT', False)))
        for name in ('LISTWISE_SHARED_SOURCES', 'LISTWISE_SPAN_REFS'):
            if name in settings:
                stack.enter_context(patch.object(config, name, settings[name]))
        if policy == 'adaptive_shared':
            stack.enter_context(patch.object(config, 'LISTWISE_SHARED_SOURCES', True))
        for row in sorted(ordered, key=lambda c: c['id'] not in carried):
            if adaptive and row['id'] not in selection['adaptive']['core_ids']:
                continue
            request = ranking.listwise_request(req, plan, submitted + [row],
                                               shared_sources=None if policy == 'adaptive_shared' else False)
            if (len(request['prompt'].encode()) > settings['RERANK_MAX_PROMPT_BYTES'] or
                    token_headroom is not None and request['reservation'] > token_headroom):
                rejected.append(row['id'])
            else:
                submitted.append(row)
        if adaptive:
            submitted, admission = fine_evidence.admit_additions(submitted, fine,
                selection['adaptive']['additions'], count_limit=llm_limit,
                byte_limit=settings['RERANK_MAX_PROMPT_BYTES'], token_headroom=token_headroom,
                request_for=lambda rows: ranking.listwise_request(req, plan,
                    sorted(rows, key=lambda c: -positions[c['id']])))
            selection['adaptive']['admission'] = admission
            rejected.extend(mid for d in admission if not d['admitted'] for mid in d['group_ids'])
        if not adaptive and settings.get('LISTWISE_SHARED_SOURCES', False) and submitted:
            for row in sorted(ordered, key=lambda c: c['id'] not in carried):
                if row in submitted:
                    continue
                trial = sorted(submitted + [row], key=lambda c: -positions[c['id']])
                request = ranking.listwise_request(req, plan, trial)
                if (len(request['prompt'].encode()) <= settings['RERANK_MAX_PROMPT_BYTES'] and
                        (token_headroom is None or request['reservation'] <= token_headroom)):
                    submitted.append(row)
                    rejected.remove(row['id'])
        submitted.sort(key=lambda c: -positions[c['id']])
        request = ranking.listwise_request(req, plan, submitted)
    return dict(fine_ids=[c['id'] for c in fine], input_ids=[c['id'] for c in submitted],
                prompt_rejected=rejected, selection=selection,
                prompt_bytes=len(request['prompt'].encode()) if submitted else 0,
                request_token_bound=request['reservation'] if submitted else 0)


def experiment(run, policies, capacities):
    searches = {s['search_id']: s for s in audit.records(run / 'server-search-debug.jsonl')}
    results = list(audit.records(run / 'completed/data/results/personamem-v2-32k.jsonl'))
    memory = list(audit.records(run / 'server-memory-debug.jsonl'))
    csv.field_size_limit(32 * 1024 * 1024)
    with (ROOT / 'data/raw/personamem-v2.csv').open() as stream:
        gold = list(csv.DictReader(stream))
    prepared = {r['conversation_id']: r for r in audit.records(ROOT / 'data/prepared/personamem-v2-32k.jsonl')}
    audit.analyze(results, memory, list(searches.values()), prepared, gold)
    output = []
    for result in results:
        search = searches[result['search_id']]
        settings = search['versions']['settings']
        baseline = replay(search, settings['CASCADE_FINE_LIMIT'], settings['CASCADE_LLM_LIMIT'])
        for key, actual in (
            ('fine_ids', search['cascade']['fine']['selected_ids']),
            ('input_ids', search['cascade']['listwise']['candidate_ids']),
            ('prompt_bytes', search['cascade']['listwise']['input_bytes'])):
            if baseline[key] != actual:
                raise ValueError(f"Baseline mismatch qa={result['qa_id']} {key}; cannot claim a valid replay")
        # Labels are never provided to candidate selection.
        snippet = gold[int(result['qa_id'])]['related_conversation_snippet']
        try:
            snippet = json.loads(snippet)
        except ValueError:
            snippet = ast.literal_eval(snippet)
        turns = [audit.norm(s['content']) for s in snippet if s['role'] == 'user']
        raw = {audit.identity(s): s for e in memory if e['user_id'] == result['user_id']
               for s in e['source_messages']}
        by_id = {c['id']: c for c in search['ranked']}
        def measure(ids):
            cards = [dict(s, id=f'{i}:{j}', text=s.get('content', ''))
                     for i, mid in enumerate(ids)
                     for j, s in enumerate(by_id[mid].get('_packet_item', {}).get('sources', []))]
            return [c['full'] for c in audit.turn_coverage(turns, cards, raw)]
        variants = {}
        for policy in policies:
            for fine, llm in capacities:
                r = replay(search, fine, llm, policy)
                r.update(fine_coverage=measure(r['fine_ids']), input_coverage=measure(r['input_ids']))
                variants[f'{policy}:{fine}/{llm}'] = r
        output.append(dict(qa_id=result['qa_id'], baseline=baseline, variants=variants))
    summary = {}
    for key in output[0]['variants']:
        rows = [r['variants'][key] for r in output]
        summary[key] = dict(fine_full_turns=sum(sum(r['fine_coverage']) for r in rows),
            input_full_turns=sum(sum(r['input_coverage']) for r in rows),
            input_candidates=sum(len(r['input_ids']) for r in rows),
            prompt_bytes=sum(r['prompt_bytes'] for r in rows),
            max_request_token_bound=max(r['request_token_bound'] for r in rows))
    return dict(run=str(run), source_hashes=SOURCE_HASHES, questions=len(output),
        baseline_reproduced=True, summary=summary, rows=output,
        caveat='Candidate admission only. Baseline fine IDs, input IDs and prompt bytes exactly reproduced. '
               'Recorded token headroom is enforced when available; otherwise expanded admission is an upper bound. '
               'Counterfactual provider call and deadline consumption is not simulated. '
               'No counterfactual listwise decisions, final packet coverage or answer accuracy are inferred.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policies', default='legacy,bounded,no_window')
    parser.add_argument('--capacities', default='16/16,24/16,24/24,32/32')
    args = parser.parse_args()
    capacities = [tuple(map(int, v.split('/'))) for v in args.capacities.split(',')]
    if any(len(c) != 2 or min(c) < 1 for c in capacities):
        parser.error('Capacities must be positive fine/listwise pairs')
    out = experiment(args.run, args.policies.split(','), capacities)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(run=str(args.run), baseline_reproduced=True, summary=out['summary']), indent=2))
