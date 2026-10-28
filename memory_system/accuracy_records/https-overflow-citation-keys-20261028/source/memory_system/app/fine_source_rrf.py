"""Fixed source/candidate rank fusion before fine selection, opt-in only."""
import asyncio
import copy
import hashlib
import json
import time

from . import answer_choice, budget, config, cross_encoder, response_observer
from . import fine_source_rrf_selection, user_source_order, rank_visible_constraints


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def remaining(limits):
    return dict(token_headroom=limits.max_tokens-limits.tokens-limits.reserved_tokens,
                call_headroom=limits.max_calls-limits.calls-limits.reserved_calls,
                deadline_seconds=max(0, limits.deadline-time.monotonic()))


def wire_body(query, documents):
    if config.CE_API_FORMAT == 'cohere':
        return dict(model=config.CE_MODEL, query=query, documents=documents, top_n=len(documents))
    if config.CE_API_FORMAT == 'dashscope':
        return dict(model=config.CE_MODEL, input=dict(query=query, documents=documents),
                    parameters=dict(top_n=len(documents), return_documents=False))
    raise ValueError('Unsupported source rerank format')


def constraints(rows):
    return list(answer_choice.build_catalog([c['_packet_item'] for c in rows])[1].values())


def missing_constraints(before, after):
    def identity(card):
        return tuple(card.get(key) for key in ('request_id', 'message_index', 'source_event_id', 'role'))
    return [card for card in before if not any(identity(card) == identity(other)
            and card['text'] in other['text'] for other in after)]


def prepare(req, plan, coarse, submitted, cascade, headroom):
    protected = [c['id'] for c in coarse if constraints([c])]
    source = user_source_order.views(coarse, req.user_id, protected_ids=protected,
                                    document_limit=config.CE_MAX_DOCUMENT_BYTES)
    query = plan['_cascade']['cross_encoder']['query']
    batches, batch = [], []
    for key, view in source['views'].items():
        if (len(query.encode()) > config.CE_MAX_DOCUMENT_BYTES
                or cross_encoder.request_size(query, [view['text']]) > config.CE_MAX_REQUEST_BYTES):
            raise ValueError('Source query/document budget')
        trial = batch + [key]
        if batch and (len(trial) > config.CE_BATCH_SIZE or cross_encoder.request_size(
                query, [source['views'][k]['text'] for k in trial]) > config.CE_MAX_REQUEST_BYTES):
            batches.append(batch)
            batch = []
        batch.append(key)
    if batch:
        batches.append(batch)
    main_reservation = cascade.listwise_request(req, plan, submitted)['reservation']
    bound = sum(cross_encoder.request_size(query, [source['views'][k]['text'] for k in b]) for b in batches)
    seconds = min(config.CE_DEADLINE_SECONDS, max(0, headroom['deadline_seconds']-config.RERANK_DEADLINE_SECONDS-2))
    reason = ('no_comparable_user_sources' if len(source['members']) < 2 else
              'token_headroom' if bound + main_reservation > headroom['token_headroom'] else
              'call_headroom' if len(batches) + 1 > headroom['call_headroom'] else
              'deadline_headroom' if seconds <= 0 else None)
    return dict(source=source, protected_ids=protected, query=query, batches=batches,
                main_reservation=main_reservation, conservative_bound=bound, seconds=seconds, reason=reason)


def propose(req, plan, coarse, baseline, source, scores, protected, limits, cascade):
    ordering = user_source_order.fused_order(coarse, source, scores, protected_ids=protected)
    if ordering['status'] != 'ordered':
        return None, ordering, {}, ordering['reason']
    trial_plan = copy.deepcopy(plan)
    trial_plan.pop('_fine_path_groups', None)
    trace = trial_plan['_cascade']
    trace['fine'], trace['listwise'], trace['reservation_missing'] = {}, {}, []
    trace['omitted'] = [o for o in trace['omitted'] if o['stage'] in ('rules', 'coarse')]
    for name in ('adaptive_admission', 'admission_budget', 'fine_source_rrf'):
        trace.pop(name, None)
    ranks = {mid:len(ordering['ordered_ids'])-i for i,mid in enumerate(ordering['ordered_ids'])}
    candidate_scores = {c['id']:c['_ce_score'] for c in coarse if c.get('_ce_score') is not None}
    selected = fine_source_rrf_selection.select(req, trial_plan, copy.deepcopy(coarse), candidate_scores,
                                               limits, cascade, ranks=ranks)
    guard = {}
    for stage in ('fine', 'submitted'):
        old, new = rank_visible_constraints.catalog(baseline[stage]), rank_visible_constraints.catalog(selected[stage])
        guard[stage] = dict(original=old, proposed=new, missing=missing_constraints(old, new))
    if any(g['missing'] for g in guard.values()):
        return None, ordering, guard, 'constraint_catalog_loss'
    original = {c['id']:c for c in coarse}
    for c in selected['fine'] + selected['submitted']:
        if c != original[c['id']]:
            raise ValueError('Candidate content, source or score mutated')
    if len(selected['fine']) > 24 or len(selected['submitted']) > 24:
        raise ValueError('Fine/main candidate capacity')
    request = cascade.listwise_request(req, trial_plan, selected['submitted'])
    if (len(request['prompt'].encode()) > min(24000, config.RERANK_MAX_PROMPT_BYTES)
            or request['reservation'] > remaining(limits)['token_headroom']):
        return None, ordering, guard, 'settled_main_budget'
    return selected, ordering, guard, None


async def apply(req, plan, coarse, baseline, cascade):
    details = plan['_cascade']['fine_source_rrf'] = dict(status='skipped', batches=[], responses=[], guard_policy='actual_rank_visible_sources_v1')
    limits = budget.current.get()
    if limits is None:
        details['reason'] = 'missing_request_budget'
        return None
    if any((config.OPTION_MARGIN_SOURCE_ADMISSION, config.JOINT_SOURCE_RANK_ADMISSION,
            config.SOURCE_CE_LATE_ADMISSION)):
        details['reason'] = 'incompatible_source_policy'
        return None
    if config.FAKE or config.CE_MODEL != 'qwen3-rerank' or config.LLM_MODEL != 'gpt-4o-mini':
        details['reason'] = 'incompatible_models'
        return None
    started = time.monotonic()
    initial_calls, initial_tokens = limits.calls, limits.tokens
    held_calls = held_tokens = 0
    try:
        budget.check()
        headroom = remaining(limits)
        job = prepare(req, plan, coarse, baseline['submitted'], cascade, headroom)
        details.update(original_headroom=headroom, source_view_ids=list(job['source']['views']),
                       source_members=job['source']['members'], protected_ids=job['protected_ids'],
                       main_reservation=job['main_reservation'], conservative_bound=job['conservative_bound'],
                       phase_seconds=job['seconds'], original_fine_ids=[c['id'] for c in baseline['fine']],
                       original_main_ids=[c['id'] for c in baseline['submitted']])
        if job['reason']:
            details['reason'] = job['reason']
            return None
        scores = {}
        limits.reserve_calls(1)
        held_calls = 1
        limits.reserve_tokens(job['main_reservation'])
        held_tokens = job['main_reservation']
        try:
            async with asyncio.timeout(job['seconds']):
                for i, keys in enumerate(job['batches']):
                    budget.check()
                    texts = [job['source']['views'][key]['text'] for key in keys]
                    bound = cross_encoder.request_size(job['query'], texts)
                    room = remaining(limits)
                    if bound > room['token_headroom'] or room['call_headroom'] < 1:
                        raise budget.BudgetExceeded('Source batch cannot preserve main reservation')
                    entry = dict(index=i, view_ids=keys, request_sha256=digest(wire_body(job['query'], texts)),
                                 input_bound=bound, status='dispatched', calls_before=limits.calls,
                                 tokens_before=limits.tokens, headroom_before=room)
                    details['batches'].append(entry)
                    try:
                        with response_observer.observe(lambda event: details['responses'].append(dict(event, batch=i))):
                            values = await cross_encoder.rerank(job['query'], texts, stage='search.fine_source_rrf',
                                                                rate_limit_retries=0)
                    finally:
                        entry.update(calls_after=limits.calls, tokens_after=limits.tokens)
                    if len(values) != len(keys):
                        raise ValueError('Incomplete source score batch')
                    scores.update(zip(keys, values, strict=True))
                    entry.update(status='ok', scores=values)
        finally:
            limits.release_tokens(held_tokens)
            limits.release_calls(held_calls)
            held_tokens = held_calls = 0
        budget.check()
        if remaining(limits)['deadline_seconds'] < config.RERANK_DEADLINE_SECONDS + 2:
            details.update(status='fallback', reason='settled_deadline')
            return None
        selected, ordering, guard, reason = propose(req, plan, coarse, baseline, job['source'], scores,
                                                    job['protected_ids'], limits, cascade)
        details.update(ordering=ordering, guard=guard, status='fallback' if reason else 'selected', reason=reason)
        if selected is not None:
            details.update(proposed_fine_ids=[c['id'] for c in selected['fine']],
                           proposed_main_ids=[c['id'] for c in selected['submitted']])
            selected['plan']['_cascade']['fine_source_rrf'] = details
        return selected
    except asyncio.CancelledError:
        details.update(status='cancelled', reason='cancelled')
        raise
    except Exception as exc:
        details.update(status='fallback', reason=type(exc).__name__)
        return None
    finally:
        if held_tokens:
            limits.release_tokens(held_tokens)
        if held_calls:
            limits.release_calls(held_calls)
        details.update(elapsed_seconds=time.monotonic()-started, remaining_budget=remaining(limits),
                       charged_calls=limits.calls-initial_calls, charged_tokens=limits.tokens-initial_tokens)
