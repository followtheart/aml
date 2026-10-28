"""Pure preparation and complete-group admission for the isolated experiment.

No provider calls, label access, text edits, or process-wide patches. Ordering
overrides are arguments, so concurrent baseline requests retain their policy.
"""
import copy
import time
from . import answer_choice, config, cross_encoder, evidence_selection, fine_evidence
from . import joint_prompt_selection, rank_constraint_catalog, user_source_order


def prepare(req, plan, coarse, original_main, headroom, cascade):
    # Retain the frozen experiment's original CE pin set and request bodies.
    # This legacy pin set is not the guard: actual stage constraints are
    # separately protected and checked using rank_constraint_catalog below.
    pinned = [c['id'] for c in coarse if answer_choice.build_catalog([c['_packet_item']])[1]]
    source = user_source_order.views(coarse, req.user_id, protected_ids=pinned,
                                     document_limit=config.CE_MAX_DOCUMENT_BYTES)
    query = cascade.query_text(req, plan)
    batches, batch = [], []
    for key, view in source['views'].items():
        if (len(query.encode()) > config.CE_MAX_DOCUMENT_BYTES or
                cross_encoder.request_size(query, [view['text']]) > config.CE_MAX_REQUEST_BYTES):
            raise ValueError('Source query/document exceeds CE request budget')
        trial = batch + [key]
        if batch and (len(trial) > config.CE_BATCH_SIZE or cross_encoder.request_size(
                query, [source['views'][k]['text'] for k in trial]) > config.CE_MAX_REQUEST_BYTES):
            batches.append(batch)
            batch = []
        batch.append(key)
    if batch:
        batches.append(batch)
    bounds = [cross_encoder.request_size(query, [source['views'][k]['text'] for k in b])
              for b in batches]
    reservation = cascade.listwise_request(req, plan, original_main)['reservation']
    seconds = min(config.CE_DEADLINE_SECONDS,
                  max(0, headroom['deadline_seconds'] - config.RERANK_DEADLINE_SECONDS))
    reason = ('no_comparable_user_sources' if len(source['members']) < 2 else
              'token_headroom' if sum(bounds) + reservation > headroom['token_headroom'] else
              'call_headroom' if len(batches) + 1 > headroom['call_headroom'] else
              'deadline_headroom' if seconds <= 0 else None)
    return dict(source=source, protected_ids=pinned, ce_query=query, batches=batches,
                batch_bounds=bounds, main_reservation=reservation,
                scheduling=dict(eligible=reason is None, reason=reason, ce_deadline_seconds=seconds))


def alternative_fine(req, plan, coarse, coarse_trace, order, cascade):
    """Reproduce the frozen fine proposal using a request-local ordering key."""
    rank_key = user_source_order.ordering_key(order)
    scores = {c['id']: c['_ce_score'] for c in coarse if c.get('_ce_score') is not None}
    if not config.FINE_EVIDENCE_ADAPTIVE or not scores:
        raise ValueError('Joint selection requires the scored adaptive fine policy')
    failed = [c for c in coarse if c['id'] not in scores]
    reserve_count = min(max(1, min(config.CASCADE_FINE_LIMIT, config.CASCADE_LLM_LIMIT) // 4), len(failed))
    failed_ids = {c['id'] for c in evidence_selection.rescue(failed, reserve_count, req.query,
                                                           plan.get('_query_specs', []))}
    fused_head = sorted(coarse, key=lambda c: -(c.get('_fused') or 0))[:max(0, config.CASCADE_FUSED_RESERVE)]
    fused_ids = {c['id'] for c in fused_head} - failed_ids
    core_pool_ids = None
    if config.COARSE_EVIDENCE_EXTEND and config.FINE_COARSE_CORE_QUOTA:
        baseline_pool_ids = set(coarse_trace['baseline_ids'])
        original_pool = [c for c in coarse if c['id'] in baseline_pool_ids]
        new_scored = sorted([c for c in coarse if c['id'] not in baseline_pool_ids
                             and c.get('_ce_score') is not None], key=rank_key, reverse=True)
        core_pool_ids = list(baseline_pool_ids | {c['id'] for c in new_scored[:cascade.FINE_NEW_COARSE_CORE_LIMIT]})
        original_failed = [c for c in original_pool if c['id'] not in scores]
        reserve_count = min(max(1, min(config.CASCADE_FINE_LIMIT, config.CASCADE_LLM_LIMIT) // 4),
                            len(original_failed))
        failed_ids = {c['id'] for c in evidence_selection.rescue(original_failed, reserve_count, req.query,
                                                               plan.get('_query_specs', []))}
        fused_ids = {c['id'] for c in sorted(original_pool, key=lambda c: -(c.get('_fused') or 0))[
                     :max(0, config.CASCADE_FUSED_RESERVE)]} - failed_ids
    fine, selection = fine_evidence.propose(coarse, min(config.CASCADE_FINE_LIMIT, len(coarse)),
        plan.get('_coverage_requirements', []), rescue_ids=failed_ids, fused_ids=fused_ids,
        extra_limit=config.FINE_EXTRA_LIMIT, core_pool_ids=core_pool_ids,
        context_queries=plan.get('_query_specs', ()) if config.FINE_TOPIC_CONTEXT else (),
        soft_promotion_limit=1 if config.FINE_WEAK_PROMOTION_ONE else None,
        semantic_reserve=config.FINE_SEMANTIC_RESERVE,
        new_coarse_insurance=config.FINE_NEW_COARSE_INSURANCE,
        new_coarse_missing_only=config.FINE_NEW_COARSE_MISSING_ONLY,
        ordering_key=rank_key)
    return fine, selection['adaptive']['path_groups']


def select(req, plan, coarse, original_fine, original_main, coarse_trace, carried_ids,
           job, source_scores, headroom, cascade, *, started_at=None):
    """Return a proposal; callers commit it only after live budget validation."""
    started = time.monotonic() if started_at is None else started_at
    order = user_source_order.fused_order(coarse, job['source'], source_scores,
                                          protected_ids=job['protected_ids'])
    if order['status'] != 'ordered':
        return dict(status='fallback', reason=order['reason'])
    raw_fine, paths = alternative_fine(req, plan, coarse, coarse_trace, order, cascade)
    pool = list(dict.fromkeys(c['id'] for c in original_fine + raw_fine))
    catalog = rank_constraint_catalog.catalog
    protected_fine = [c['id'] for c in original_fine if catalog([c])]
    mandatory = set(carried_ids) & {c['id'] for c in original_main}
    mandatory.update(c['id'] for c in original_main if catalog([c]))
    working_plan = copy.deepcopy(plan)
    paths = plan.get('_fine_path_groups', []) + paths
    working_plan['_fine_path_groups'] = [list(p) for p in sorted({tuple(sorted(g)) for g in paths})]
    result = joint_prompt_selection.choose(coarse, pool, original_fine, rank_ids=order['ordered_ids'],
        protected_fine_ids=protected_fine, mandatory_main_ids=mandatory,
        render=lambda rows: cascade.listwise_request(req, working_plan, rows),
        graph=lambda rows: cascade.listwise_graph(working_plan, rows),
        token_headroom=headroom['token_headroom'], byte_limit=config.RERANK_MAX_PROMPT_BYTES,
        count_limit=config.CASCADE_LLM_LIMIT + config.FINE_EXTRA_LIMIT,
        fine_limit=config.CASCADE_FINE_LIMIT + config.FINE_EXTRA_LIMIT,
        deadline=started + max(0, headroom['deadline_seconds'] - config.RERANK_DEADLINE_SECONDS))
    if result['status'] != 'selected':
        return result
    guard = {}
    for stage, before, after in [('fine', original_fine, result['fine']),
                                  ('main', original_main, result['submitted'])]:
        a, b = catalog(before), catalog(after)
        guard[stage] = dict(original=a, proposed=b, missing=rank_constraint_catalog.missing(a, b))
    if any(g['missing'] for g in guard.values()):
        return dict(status='fallback', reason='constraint_catalog_loss', guard=guard)
    return dict(result, plan=working_plan, proposal=order, guard=guard,
                selection_seconds=time.monotonic() - started)
