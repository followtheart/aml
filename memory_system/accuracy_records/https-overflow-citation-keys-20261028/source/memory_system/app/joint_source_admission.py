"""Default-off, request-scoped source scoring and complete-group admission."""
import asyncio
import copy
import hashlib
import json
import time
from . import budget, config, cross_encoder, joint_source_selection


def remaining(limits):
    return dict(token_headroom=limits.max_tokens-limits.tokens-limits.reserved_tokens,
                call_headroom=limits.max_calls-limits.calls-limits.reserved_calls,
                deadline_seconds=max(0, limits.deadline-time.monotonic()))


def body_hash(query, texts):
    if config.CE_API_FORMAT == 'cohere':
        body = dict(model=config.CE_MODEL, query=query, documents=texts, top_n=len(texts))
    elif config.CE_API_FORMAT == 'dashscope':
        body = dict(model=config.CE_MODEL, input=dict(query=query, documents=texts),
                    parameters=dict(top_n=len(texts), return_documents=False))
    else:
        raise ValueError('Unsupported CE format')
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


async def propose(req, plan, coarse, fine, main, carried_ids, cascade):
    details = dict(status='disabled', batches=[], original_fine_ids=[c['id'] for c in fine],
                   original_main_ids=[c['id'] for c in main])
    if not config.JOINT_SOURCE_RANK_ADMISSION:
        return None, details
    limits = budget.current.get()
    if (limits is None or not config.FINE_EVIDENCE_ADAPTIVE or config.LISTWISE_STAGED
            or config.SOURCE_CE_LATE_ADMISSION or not main):
        details.update(status='skipped', reason='unsupported_policy_or_missing_budget')
        return None, details
    started = time.monotonic()
    try:
        headroom = remaining(limits)
        details['original_headroom'] = headroom
        job = joint_source_selection.prepare(req, plan, coarse, main, headroom, cascade)
        details.update(query=job['ce_query'], protected_score_ids=job['protected_ids'],
                       source_view_ids=list(job['source']['views']),
                       source_members=job['source']['members'], scheduling=job['scheduling'],
                       main_reservation=job['main_reservation'])
        if not job['scheduling']['eligible']:
            details.update(status='skipped', reason=job['scheduling']['reason'])
            return None, details
        available = remaining(limits)
        seconds = min(config.CE_DEADLINE_SECONDS,
                      max(0, available['deadline_seconds']-config.RERANK_DEADLINE_SECONDS))
        if (sum(job['batch_bounds'])+job['main_reservation'] > available['token_headroom']
                or len(job['batches'])+1 > available['call_headroom'] or seconds <= 0):
            raise budget.BudgetExceeded('Main budget unavailable after preparation')
        details['ce_deadline_seconds'] = seconds
        tokens_reserved = calls_reserved = False
        scores = {}
        try:
            limits.reserve_calls(1)
            calls_reserved = True
            limits.reserve_tokens(job['main_reservation'])
            tokens_reserved = True
            async with asyncio.timeout(seconds):
                for index, keys in enumerate(job['batches']):
                    texts = [job['source']['views'][key]['text'] for key in keys]
                    entry = dict(index=index, view_ids=keys, status='started',
                                 body_sha256=body_hash(job['ce_query'], texts), bound=job['batch_bounds'][index])
                    details['batches'].append(entry)
                    try:
                        values = await cross_encoder.rerank(job['ce_query'], texts,
                            stage='search.joint_source_rerank', rate_limit_retries=0)
                        if len(values) != len(keys):
                            raise ValueError('Incomplete source batch')
                        scores.update(zip(keys, values, strict=True))
                        entry.update(status='ok', scores=values)
                    except BaseException as exc:
                        entry.update(status='error', error=type(exc).__name__)
                        raise
        finally:
            if tokens_reserved:
                limits.release_tokens(job['main_reservation'])
            if calls_reserved:
                limits.release_calls(1)
        headroom = remaining(limits)
        details['selection_headroom'] = headroom
        result = joint_source_selection.select(req, plan, coarse, fine, main,
            plan['_cascade']['coarse'], carried_ids, job, scores, headroom, cascade)
        details.update(status=result['status'], reason=result.get('reason'))
        if result['status'] != 'selected':
            return None, details
        available = remaining(limits)
        request = result['request']
        if (not result['submitted'] or request['reservation'] > available['token_headroom']
                or available['call_headroom'] < 1
                or available['deadline_seconds'] < config.RERANK_DEADLINE_SECONDS
                or len(request['prompt'].encode()) > config.RERANK_MAX_PROMPT_BYTES):
            raise budget.BudgetExceeded('Main budget unavailable after selection')
        details.update(proposal=result['proposal'], guard=result['guard'],
            protected_fine_ids=result['protected_fine_ids'], mandatory_main_ids=result['mandatory_main_ids'],
            decisions=result['decisions'], rejections=result['rejections'],
            proposed_fine_ids=[c['id'] for c in result['fine']],
            proposed_main_ids=[c['id'] for c in result['submitted']],
            proposed_main_bytes=len(request['prompt'].encode()),
            proposed_main_reservation=request['reservation'], selection_seconds=result['selection_seconds'])
        return result, details
    except Exception as exc:
        details.update(status='fallback', reason=type(exc).__name__)
        return None, details
    finally:
        details.update(elapsed_seconds=time.monotonic()-started, remaining_budget=remaining(limits))


async def apply(req, plan, coarse, fine, main, carried_ids, baseline_ids, admission, cascade):
    """Commit only validated proposals, including actual stage diagnostics."""
    result, details = await propose(req, plan, coarse, fine, main, carried_ids, cascade)
    trace = plan['_cascade']
    trace['joint_source_admission'] = details
    if result is None:
        return fine, main, carried_ids, baseline_ids, admission
    old_fine = trace['fine']
    details['original_stages'] = copy.deepcopy(dict(fine=old_fine,
        adaptive_admission=trace.get('adaptive_admission'), omitted=trace['omitted'],
        reservation_missing=trace['reservation_missing'], admission_selection=admission,
        baseline_candidate_ids=sorted(baseline_ids), carried_ids=sorted(carried_ids)))
    fine, main = result['fine'], result['submitted']
    fine_ids, main_ids = {c['id'] for c in fine}, {c['id'] for c in main}
    new_carried = carried_ids & fine_ids
    reservations = [dict(r, selected=r['candidate_id'] in fine_ids)
                    for r in old_fine['selection']['reservations']]
    fine_decisions = [dict(id=c['id'], selected=c['id'] in fine_ids,
        reason='joint_relevance_admission' if c['id'] in fine_ids else 'joint_fine_limit',
        represented_by=[]) for c in coarse]
    trace['fine'] = dict(input_count=len(coarse), output_count=len(fine),
        selected_ids=[c['id'] for c in fine], reserved_ids=sorted(new_carried),
        fused_reserved_ids=sorted(set(old_fine.get('fused_reserved_ids', [])) & fine_ids),
        selection=dict(policy='joint_source_rank', reservations=reservations, decisions=fine_decisions))
    reasons = {r['id']: r['reason'] for r in result['rejections']}
    trace['omitted'] = [r for r in trace['omitted'] if r['stage'] not in ('fine', 'listwise')]
    trace['omitted'].extend(dict(id=c['id'], stage='fine', reason='joint_fine_limit')
                           for c in coarse if c['id'] not in fine_ids)
    trace['omitted'].extend(dict(id=c['id'], stage='listwise', reason=reasons.get(c['id'], 'joint_group_not_admitted'))
                           for c in fine if c['id'] not in main_ids)
    trace['reservation_missing'] = [r for r in trace['reservation_missing'] if r['stage'] not in ('fine', 'listwise')]
    for r in reservations:
        if r['reasons'] == ['fused_head']:
            continue
        for stage, ids in [('fine', fine_ids), ('listwise', main_ids)]:
            if r['candidate_id'] not in ids:
                trace['reservation_missing'].extend(dict(requirement_id=rid, candidate_id=r['candidate_id'],
                    stage=stage, reason='joint_group_not_admitted') for rid in r['requirement_ids'] or [None])
    trace['adaptive_admission'] = dict(policy='joint_source_rank',
        mandatory_ids=result['mandatory_main_ids'], decisions=result['decisions'])
    admission = dict(policy='joint_source_rank', decisions=[dict(id=c['id'], selected=c['id'] in main_ids,
        reason='joint_relevance_admission' if c['id'] in main_ids else reasons.get(c['id'], 'joint_group_not_admitted'))
        for c in fine])
    # The live plan owns the original trace object; never replace it with the
    # deep-copied plan used to evaluate hypothetical source/graph combinations.
    plan['_fine_path_groups'] = result['plan']['_fine_path_groups']
    details['baseline_candidate_ids_semantics'] = 'joint_main_all_ids_without_core_extra_split'
    return fine, main, new_carried, main_ids, admission
