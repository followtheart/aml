"""Pure selection replay using the frozen baseline admission algorithm.

Generated once from cascade_rerank's post-CE selection block; default runtime
keeps its original path. No network calls or global function monkeypatches.
"""
import time
from . import config, evidence_selection, fine_evidence

SELECTION_BLOCK_SHA256 = 'cb678cab172da046663c7f5a049e17d1474b233a0d3e3658e291763b8e1cd95d'


def select(req, plan, coarse, scores, limits, cascade, *, ranks=None):
    trace = plan['_cascade']
    requirements = plan.get('_coverage_requirements', [])
    coarse_selection = trace['coarse']['selection']
    shortlist = cascade.shortlist
    listwise_request = cascade.listwise_request
    FINE_NEW_COARSE_CORE_LIMIT = cascade.FINE_NEW_COARSE_CORE_LIMIT
    with fine_evidence.ordered_by(ranks):
        # No calibration between CE and graph scores: failed CE entries get a
        # reserved, bounded opportunity; the rest use CE order only.
        failed = [c for c in coarse if c['id'] not in scores]
        reserve_count = min(max(1, min(config.CASCADE_FINE_LIMIT, config.CASCADE_LLM_LIMIT) // 4), len(failed)) if scores else 0
        failed_ids = {c['id'] for c in evidence_selection.rescue(failed, reserve_count, req.query,
                                                               plan.get('_query_specs', []))}
        # Retrieval hits may reserve a seat only within the bounded CE rank window.
        fused_head = sorted(coarse, key=lambda c: -(c.get('_fused') or 0))[:max(0, config.CASCADE_FUSED_RESERVE)]
        fused_ids = {c['id'] for c in fused_head} - failed_ids if scores else set()
        fine_selection = {}
        fine = shortlist(coarse, min(config.CASCADE_FINE_LIMIT, len(coarse)), requirements,
            lambda c: (c['_ce_score'] if c.get('_ce_score') is not None else -1e30, c.get('_fused', 0)),
            reserve_ids=failed_ids, fused_ids=fused_ids, trace=fine_selection,
            bound_soft_reservations=True) if scores else shortlist(coarse, config.CASCADE_FINE_LIMIT,
                requirements, lambda c: (c.get('_fused', 0),), trace=fine_selection)
        if config.FINE_EVIDENCE_ADAPTIVE and scores:
            core_pool_ids = None
            core_failed_ids, core_fused_ids = failed_ids, fused_ids
            if config.COARSE_EVIDENCE_EXTEND and config.FINE_COARSE_CORE_QUOTA:
                baseline_pool_ids = set(trace['coarse']['baseline_ids'])
                original_pool = [c for c in coarse if c['id'] in baseline_pool_ids]
                new_scored = sorted([c for c in coarse if c['id'] not in baseline_pool_ids
                                     and c.get('_ce_score') is not None], key=fine_evidence.key, reverse=True)
                core_pool_ids = list(baseline_pool_ids | {c['id'] for c in new_scored[:FINE_NEW_COARSE_CORE_LIMIT]})
                original_failed = [c for c in original_pool if c['id'] not in scores]
                original_reserve = min(max(1, min(config.CASCADE_FINE_LIMIT, config.CASCADE_LLM_LIMIT) // 4),
                                       len(original_failed))
                core_failed_ids = {c['id'] for c in evidence_selection.rescue(
                    original_failed, original_reserve, req.query, plan.get('_query_specs', []))}
                core_fused_ids = {c['id'] for c in sorted(original_pool,
                    key=lambda c: -(c.get('_fused') or 0))[:max(0, config.CASCADE_FUSED_RESERVE)]} - core_failed_ids
            fine, fine_selection = fine_evidence.propose(coarse, min(config.CASCADE_FINE_LIMIT, len(coarse)),
                requirements, rescue_ids=core_failed_ids, fused_ids=core_fused_ids, extra_limit=config.FINE_EXTRA_LIMIT,
                core_pool_ids=core_pool_ids, context_queries=plan.get('_query_specs', ()) if config.FINE_TOPIC_CONTEXT else (),
                soft_promotion_limit=1 if config.FINE_WEAK_PROMOTION_ONE else None,
                semantic_reserve=config.FINE_SEMANTIC_RESERVE,
                new_coarse_insurance=config.FINE_NEW_COARSE_INSURANCE,
                new_coarse_missing_only=config.FINE_NEW_COARSE_MISSING_ONLY)
            if core_pool_ids is not None:
                fine_selection['adaptive']['new_coarse_core_limit'] = FINE_NEW_COARSE_CORE_LIMIT
            plan['_fine_path_groups'] = fine_selection['adaptive']['path_groups']
        fine_ids = {c['id'] for c in fine}
        reservations = [r for r in fine_selection['reservations'] if r['selected']]
        carried_ids = {r['candidate_id'] for r in reservations}
        # Fused-head seats are a soft guard against one bad CE score; witnessed
        # coverage and rescue reservations outrank them at every later cap.
        soft_ids = {r['candidate_id'] for r in reservations if r['reasons'] == ['fused_head']}
        trace['fine'] = dict(input_count=len(coarse), output_count=len(fine), selected_ids=[c['id'] for c in fine],
                             selection=fine_selection, reserved_ids=sorted(carried_ids),
                             fused_reserved_ids=sorted(fused_ids & fine_ids))
        for stage, selection in (('coarse', coarse_selection), ('fine', fine_selection)):
            for r in selection['reservations']:
                if not r['selected'] and r['reasons'] != ['fused_head']:
                    trace['reservation_missing'].extend(dict(requirement_id=rid, candidate_id=r['candidate_id'],
                        stage=stage, reason='capacity') for rid in r['requirement_ids'] or [None])
        trace['omitted'].extend(dict(id=c['id'], stage='fine', reason='fine_limit') for c in coarse if c['id'] not in fine_ids)
        # Preserve the fine rank and its coverage reservations at the listwise cap.
        positions = {c['id']: len(fine) - i for i, c in enumerate(fine)}
        admission_selection = {}
        ordered = shortlist(fine, config.CASCADE_LLM_LIMIT, [],
                            lambda c: (c['id'] in carried_ids and c['id'] not in soft_ids, positions[c['id']]),
                            carry_ids=carried_ids, trace=admission_selection)
        submitted, token_limited = [], False
        mandatory = carried_ids
        for c in sorted(ordered, key=lambda c: c['id'] not in mandatory):
            request = listwise_request(req, plan, submitted + [c],
                                       shared_sources=None if config.FINE_EVIDENCE_ADAPTIVE else False)
            if len(request['prompt'].encode('utf-8')) > config.RERANK_MAX_PROMPT_BYTES:
                trace['omitted'].append(dict(id=c['id'], stage='listwise', reason='prompt_budget'))
            elif limits and request['reservation'] > limits.max_tokens - limits.tokens - limits.reserved_tokens:
                token_limited = True
                trace['omitted'].append(dict(id=c['id'], stage='listwise', reason='token_budget'))
            else:
                submitted.append(c)
        baseline_ids = {c['id'] for c in submitted}
        admission_headroom = (max(0, limits.max_tokens - limits.tokens - limits.reserved_tokens)
                              if limits else None)
        trace['admission_budget'] = dict(token_headroom=admission_headroom,
            call_headroom=limits.max_calls - limits.calls - limits.reserved_calls if limits else None,
            deadline_seconds=max(0, limits.deadline - time.monotonic()) if limits else None,
            prompt_limit_bytes=config.RERANK_MAX_PROMPT_BYTES)
        if config.LISTWISE_SHARED_SOURCES and submitted:
            # Preserve the original admission before spending any compression gain.
            # CE scheduling above stays original; full request tokens still apply.
            for c in sorted(ordered, key=lambda c: c['id'] not in mandatory):
                if c['id'] in baseline_ids:
                    continue
                trial = sorted(submitted + [c], key=lambda c: -positions[c['id']])
                request = listwise_request(req, plan, trial)
                if len(request['prompt'].encode('utf-8')) > config.RERANK_MAX_PROMPT_BYTES:
                    continue
                if limits and request['reservation'] > limits.max_tokens - limits.tokens - limits.reserved_tokens:
                    continue
                submitted.append(c)
                trace['omitted'] = [o for o in trace['omitted']
                                    if not (o['id'] == c['id'] and o['stage'] == 'listwise')]
        if config.FINE_EVIDENCE_ADAPTIVE and 'adaptive' in fine_selection:
            core_admitted = [c['id'] for c in submitted]
            submitted, decisions = fine_evidence.admit_additions(submitted, fine,
                fine_selection['adaptive']['additions'],
                count_limit=config.CASCADE_LLM_LIMIT + config.FINE_EXTRA_LIMIT,
                byte_limit=config.RERANK_MAX_PROMPT_BYTES, token_headroom=admission_headroom,
                request_for=lambda rows: listwise_request(req, plan,
                    sorted(rows, key=lambda c: -positions[c['id']])))
            trace['adaptive_admission'] = dict(core_ids=core_admitted, decisions=decisions)
        submitted.sort(key=lambda c: -positions[c['id']])
        submitted_ids = {c['id'] for c in submitted}
        # An adaptive group may re-admit a core item rejected by the first pass.
        # Never leave a stale omission, or label a byte/token rejection as a count cap.
        trace['omitted'] = [o for o in trace['omitted']
                            if not (o['stage'] == 'listwise' and o['id'] in submitted_ids)]
        adaptive_reasons = {mid: d['reason'] for d in trace.get('adaptive_admission', {}).get('decisions', [])
                            if not d['admitted'] for mid in d['group_ids']}
        trace['omitted'].extend(dict(id=c['id'], stage='listwise',
                                    reason=adaptive_reasons.get(c['id'], 'listwise_limit'))
                                for c in fine if c['id'] not in submitted_ids and c not in ordered)
        for r in reservations:
            if r['candidate_id'] not in submitted_ids and r['reasons'] != ['fused_head']:
                omission = next((o for o in reversed(trace['omitted']) if o['id'] == r['candidate_id']), {})
                reason = omission.get('reason', 'capacity')
                if reason == 'listwise_limit':
                    reason = 'capacity'
                trace['reservation_missing'].extend(dict(requirement_id=rid,candidate_id=r['candidate_id'],
                    stage='listwise',reason=reason) for rid in r['requirement_ids'] or [None])
    return dict(plan=plan, fine=fine, submitted=submitted, ordered=ordered,
                token_limited=token_limited, carried_ids=carried_ids,
                baseline_ids=baseline_ids, admission_selection=admission_selection)
