"""Bounded coarse selection -> provider Cross-Encoder -> LLM listwise selection."""
import asyncio
import json
import math
import re
import time
import httpx
from . import budget, config, cross_encoder, evidence_selection, llm, profile, prompts, rerank_query


def protected(item):
    return bool(item.get('_user_rule')) or profile.is_forget_rule(item)


def shortlist(rows, count, requirements, key, reserve_ids=(), *, trace=None, carry_ids=()):
    return evidence_selection.select(rows, count, requirements, key, reserve_ids,
                                     trace=trace, carry_ids=carry_ids)


def query_text(req, plan):
    return rerank_query.build(req, plan)


def listwise_prompt(req, plan, rows):
    ids = {c['id'] for c in rows}
    triples = [t for t in plan.get('_fusion_triples', []) if t.get('amu_id') in ids]
    edges = [e for e in plan.get('_fusion_edges', []) if e['source'] in ids and e['target'] in ids]
    metadata = dict(triples=[{k: t.get(k) for k in ('amu_id', 'subject', 'relation', 'object', 'valid_from', 'valid_to')}
                            for t in triples[:40]], associations=edges[:40])
    return prompts.render('05c_listwise_rerank.txt', query=req.query,
        options='\n'.join(req.options or []) or '(none)',
        time_scope=json.dumps(plan.get('time_scope') or {}, ensure_ascii=False),
        requirements=json.dumps(plan.get('_coverage_requirements', []), ensure_ascii=False),
        relations=json.dumps(metadata, ensure_ascii=False), last_index=len(rows) - 1,
        candidates='\n'.join(f"{i}: [id: {c['id']}] " + re.sub(r'\s+', ' ', c.get('_rank_text', c['content'])).strip()
                             for i, c in enumerate(rows)))


def schema(count):
    index = dict(type='integer', minimum=0, maximum=count - 1)
    return dict(type='object', additionalProperties=False, required=['ranking', 'irrelevant', 'groups'], properties={
        'ranking': dict(type='array', description='All candidate indices exactly once, ordered by usefulness for answering the question; include irrelevant indices last.',
                        minItems=count, maxItems=count, uniqueItems=True, items=index),
        'irrelevant': dict(type='array', description='Indices with no useful evidence for this question. They still occur in ranking but will not be returned.',
                           maxItems=count, uniqueItems=True, items=index),
        'groups': dict(type='array', description='Usually empty []. Only group two to four complementary candidates that MUST be read together. Never copy ranking here, never group the whole list or redundant/irrelevant items.',
                       maxItems=count // 2, items=dict(type='array', minItems=2,
            maxItems=min(4, count), uniqueItems=True, items=index))})


def validate(result, count):
    if not isinstance(result, dict):
        raise ValueError('Listwise response must be an object')
    def indices(value):
        return (isinstance(value, list) and all(type(i) is int and 0 <= i < count for i in value)
                and len(set(value)) == len(value))
    ranking, irrelevant, groups = (result.get(k) for k in ('ranking', 'irrelevant', 'groups'))
    if not indices(ranking) or len(ranking) != count or not indices(irrelevant) or not isinstance(groups, list):
        raise ValueError('Invalid listwise permutation or irrelevant indices')
    grouped = set()
    for group in groups:
        if (not indices(group) or not 2 <= len(group) <= 4 or set(group) & set(irrelevant)
                or set(group) & grouped):
            raise ValueError('Invalid or overlapping evidence group')
        grouped.update(group)
    return ranking, set(irrelevant), groups


def listwise_request(req, plan, rows, *, repair=False):
    """Use the same full request bound for scheduling and provider dispatch."""
    prompt = listwise_prompt(req, plan, rows)
    output_limit = max(256, 24 * len(rows))
    output_schema = schema(len(rows))
    system = ('Rank the supplied evidence for the question and call emit_json_result. '
              'Return every input index once in ranking. Mark unrelated items in irrelevant. '
              'Default groups to []; create a group only for 2 to 4 complementary items '
              'that must be read together. Never copy the ranking list into groups.')
    if repair:
        system += ' Repair the invalid structure: groups must be disjoint, exclude irrelevant indices and contain at most 4 members each.'
    reservation = (len(prompt.encode('utf-8')) + len(json.dumps(output_schema).encode('utf-8'))
                   + len(system.encode('utf-8')) + 1024 + output_limit)
    return dict(prompt=prompt, schema=output_schema, max_tokens=output_limit,
                system=system, reservation=reservation)


async def rank(req, plan, candidates):
    trace = plan['_cascade'] = dict(coarse={}, cross_encoder={}, fine={}, listwise={}, omitted=[],
                                   reservation_missing=[])
    plan['_rerank'], plan['_evidence_groups'] = [], []
    if not candidates:
        plan['_rerank_status'] = 'not_run'
        return []
    rules = [c for c in candidates if protected(c)]
    ordinary = [c for c in candidates if not protected(c)]
    requirements = plan.get('_coverage_requirements', [])
    coarse_selection = {}
    coarse = shortlist(ordinary, config.CASCADE_COARSE_LIMIT, requirements,
                       lambda c: (c.get('_fused', 0),), trace=coarse_selection)
    trace['coarse'] = dict(input_count=len(ordinary), output_count=len(coarse),
                          limit=config.CASCADE_COARSE_LIMIT, selected_ids=[c['id'] for c in coarse],
                          selection=coarse_selection)
    coarse_ids = {c['id'] for c in coarse}
    trace['omitted'].extend(dict(id=c['id'], stage='coarse', reason='coarse_limit')
                            for c in ordinary if c['id'] not in coarse_ids)
    plan['_rerank_pool_size'] = len(coarse)
    query = query_text(req, plan)
    pending, rejected = [], []
    limits = budget.current.get()
    sizes = {}
    for c in coarse:
        text = c.get('_rank_text', c['content'])
        sizes[c['id']] = cross_encoder.request_size(query, [text])
        if (len(text.encode('utf-8')) > config.CE_MAX_DOCUMENT_BYTES or
                len(query.encode('utf-8')) > config.CE_MAX_DOCUMENT_BYTES or
                sizes[c['id']] > config.CE_MAX_REQUEST_BYTES):
            rejected.append(dict(id=c['id'], error='input_budget'))
            continue
        pending.append(c)
    # Keep enough for one affordable whole-candidate listwise request. This is
    # scheduling headroom, not a second reservation of CE's provider input.
    # The full listwise pool is admitted again against the settled budget below.
    headroom = 0
    if limits:
        available = max(0, limits.max_tokens - limits.tokens - limits.reserved_tokens)
        smallest_ce = min((sizes[c['id']] for c in pending), default=0)
        for c in coarse:
            budget.check()
            request = listwise_request(req, plan, [c])
            if (len(request['prompt'].encode('utf-8')) <= config.RERANK_MAX_PROMPT_BYTES
                    and request['reservation'] + smallest_ce <= available):
                headroom = max(headroom, request['reservation'])
    held_calls = int(headroom > 0 and limits.calls + limits.reserved_calls < limits.max_calls)
    batch_traces, scores, waves = [], {}, []
    ce_seconds = min(config.CE_DEADLINE_SECONDS, max(.01, limits.deadline - time.monotonic() - 5)) if limits else config.CE_DEADLINE_SECONDS
    ce_deadline = time.monotonic() + ce_seconds
    async def score(entry, rows):
        try:
            entry['status'] = 'running'
            values = await cross_encoder.rerank(query, [c.get('_rank_text', c['content']) for c in rows],
                                                 stage=entry.get('stage', f"search.cross_encoder.batch_{entry['batch']}"))
            if len(values) != len(rows) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError('Cross-Encoder returned invalid score mapping')
            scores.update((c['id'], float(v)) for c, v in zip(rows, values))
            entry['status'] = 'ok'
        except asyncio.CancelledError:
            entry['status'] = 'cancelled'
            raise
        except Exception as exc:
            entry.update(status='error', error=type(exc).__name__,
                         retryable_timeout=isinstance(exc, (TimeoutError, httpx.TimeoutException)))
    if held_calls:
        limits.reserve_calls(held_calls)
    try:
        async with asyncio.timeout(ce_seconds):
            while pending:
                budget.check()
                available = (max(0, limits.max_tokens - limits.tokens - limits.reserved_tokens - headroom)
                             if limits else config.CE_MAX_REQUEST_BYTES * config.RERANK_CONCURRENCY)
                calls = max(0, limits.max_calls - limits.calls - limits.reserved_calls) if limits else config.RERANK_CONCURRENCY
                if not calls:
                    rejected.extend(dict(id=c['id'], error='BudgetExceeded', reason='call_budget') for c in pending)
                    pending.clear()
                    break
                # No CE from an earlier wave is still holding a reservation.
                # A document that exceeds the *whole* remainder is genuinely
                # unaffordable; smaller later documents may still fit.
                while pending and sizes[pending[0]['id']] > available:
                    rejected.append(dict(id=pending.pop(0)['id'], error='BudgetExceeded', reason='token_budget'))
                if not pending:
                    break
                slots = min(config.RERANK_CONCURRENCY, calls, len(pending))
                cap = min(config.CE_MAX_REQUEST_BYTES, available // slots)
                if sizes[pending[0]['id']] > cap:
                    # Sharing is optional: do not reject a document merely
                    # because it is larger than half of the available budget.
                    slots, cap = 1, min(config.CE_MAX_REQUEST_BYTES, available)
                wave = []
                for _ in range(slots):
                    rows, size = [], 0
                    while (pending and len(rows) < config.CE_BATCH_SIZE
                           and size + sizes[pending[0]['id']] <= cap):
                        c = pending.pop(0)
                        rows.append(c)
                        size += sizes[c['id']]
                    if not rows:
                        break
                    entry = dict(batch=len(batch_traces) + 1, wave=len(waves) + 1,
                        candidate_ids=[c['id'] for c in rows], candidate_count=len(rows),
                        input_bound=size, status='queued')
                    batch_traces.append(entry)
                    wave.append((entry, rows))
                waves.append(dict(wave=len(waves) + 1, available_tokens=available,
                                  batches=[entry['batch'] for entry, _ in wave]))
                # Settlement includes rerank's finally/release_tokens. Only then
                # recompute batch sizes; never fail a queued batch on in-flight
                # reservation pressure or retry a request already sent to HTTP.
                await asyncio.gather(*(score(entry, rows) for entry, rows in wave))
            # Retry only request timeouts, after every initial wave has settled.
            # Keep the original deadline, shared budget and listwise reservation.
            timed_out = {mid for b in batch_traces if b.get('retryable_timeout')
                         for mid in b['candidate_ids'] if mid not in scores}
            retry_rows = evidence_selection.rescue([c for c in coarse if c['id'] in timed_out],
                config.CE_RETRY_BATCHES * min(config.CE_RETRY_BATCH_SIZE, config.CE_BATCH_SIZE),
                req.query, plan.get('_query_specs', []))
            for attempt in range(config.CE_RETRY_BATCHES):
                budget.check()
                if not retry_rows or ce_deadline - time.monotonic() < min(1., config.CE_TIMEOUT_SECONDS / 4):
                    break
                available = (max(0, limits.max_tokens - limits.tokens - limits.reserved_tokens - headroom)
                             if limits else config.CE_MAX_REQUEST_BYTES)
                if limits and limits.calls + limits.reserved_calls >= limits.max_calls:
                    break
                retry_batch, size = [], 0
                for c in list(retry_rows):
                    if size + sizes[c['id']] <= min(available, config.CE_MAX_REQUEST_BYTES):
                        retry_batch.append(c)
                        retry_rows.remove(c)
                        size += sizes[c['id']]
                    if len(retry_batch) >= min(config.CE_RETRY_BATCH_SIZE, config.CE_BATCH_SIZE):
                        break
                if not retry_batch:
                    break
                entry = dict(batch=len(batch_traces) + 1, wave=len(waves) + 1,
                    stage=f'search.cross_encoder.retry_{attempt + 1}', retry=True,
                    candidate_ids=[c['id'] for c in retry_batch], candidate_count=len(retry_batch),
                    input_bound=size, status='queued')
                batch_traces.append(entry)
                waves.append(dict(wave=len(waves) + 1, available_tokens=available, batches=[entry['batch']], retry=True))
                await score(entry, retry_batch)
    except TimeoutError:
        rejected.extend(dict(id=c['id'], error='TimeoutError', reason='stage_deadline') for c in pending)
    finally:
        if held_calls:
            limits.release_calls(held_calls)
    ce_errors = rejected + [dict(batch=b['batch'], error=b.get('error', b['status']))
                            for b in batch_traces if b['status'] != 'ok'
                            and any(mid not in scores for mid in b['candidate_ids'])]
    for c in coarse:
        c['_ce_score'] = scores.get(c['id'])
        if c['id'] in scores:
            c['_final'], c['_score_kind'] = scores[c['id']], 'cross_encoder'
        else:
            c.pop('_final', None)
            c['_score_kind'] = 'unscored'
    trace['cross_encoder'] = dict(status='ok' if len(scores) == len(coarse) else 'partial' if scores else 'fallback',
        input_count=len(coarse), scored_count=len(scores), batches=sorted(batch_traces, key=lambda b: b['batch']), errors=ce_errors,
        model='fake-cross-encoder' if config.FAKE else config.CE_MODEL,
        scheduler='settled_waves', waves=waves, listwise_token_headroom=headroom, listwise_call_headroom=held_calls,
        query=query, recovered_ids=sorted({mid for b in batch_traces if b['status'] != 'ok'
            for mid in b['candidate_ids'] if mid in scores}))
    # No calibration between CE and graph scores: failed CE entries get a
    # reserved, bounded opportunity; the rest use CE order only.
    failed = [c for c in coarse if c['id'] not in scores]
    reserve_count = min(max(1, min(config.CASCADE_FINE_LIMIT, config.CASCADE_LLM_LIMIT) // 4), len(failed)) if scores else 0
    failed_ids = {c['id'] for c in evidence_selection.rescue(failed, reserve_count, req.query,
                                                           plan.get('_query_specs', []))}
    fine_selection = {}
    fine = shortlist(coarse, min(config.CASCADE_FINE_LIMIT, len(coarse)), requirements,
        lambda c: (c['_ce_score'] if c.get('_ce_score') is not None else -1e30, c.get('_fused', 0)),
        reserve_ids=failed_ids, trace=fine_selection) if scores else shortlist(coarse, config.CASCADE_FINE_LIMIT,
            requirements, lambda c: (c.get('_fused', 0),), trace=fine_selection)
    fine_ids = {c['id'] for c in fine}
    reservations = [r for r in fine_selection['reservations'] if r['selected']]
    carried_ids = {r['candidate_id'] for r in reservations}
    trace['fine'] = dict(input_count=len(coarse), output_count=len(fine), selected_ids=[c['id'] for c in fine],
                         selection=fine_selection, reserved_ids=sorted(carried_ids))
    for stage, selection in (('coarse', coarse_selection), ('fine', fine_selection)):
        for r in selection['reservations']:
            if not r['selected']:
                trace['reservation_missing'].extend(dict(requirement_id=rid, candidate_id=r['candidate_id'],
                    stage=stage, reason='capacity') for rid in r['requirement_ids'] or [None])
    trace['omitted'].extend(dict(id=c['id'], stage='fine', reason='fine_limit') for c in coarse if c['id'] not in fine_ids)
    # Preserve the fine rank and its coverage reservations at the listwise cap.
    positions = {c['id']: len(fine) - i for i, c in enumerate(fine)}
    admission_selection = {}
    ordered = shortlist(fine, config.CASCADE_LLM_LIMIT, [], lambda c: (positions[c['id']],),
                        carry_ids=carried_ids, trace=admission_selection)
    submitted, token_limited = [], False
    mandatory = carried_ids
    for c in sorted(ordered, key=lambda c: c['id'] not in mandatory):
        request = listwise_request(req, plan, submitted + [c])
        if len(request['prompt'].encode('utf-8')) > config.RERANK_MAX_PROMPT_BYTES:
            trace['omitted'].append(dict(id=c['id'], stage='listwise', reason='prompt_budget'))
        elif limits and request['reservation'] > limits.max_tokens - limits.tokens - limits.reserved_tokens:
            token_limited = True
            trace['omitted'].append(dict(id=c['id'], stage='listwise', reason='token_budget'))
        else:
            submitted.append(c)
    submitted.sort(key=lambda c: -positions[c['id']])
    submitted_ids = {c['id'] for c in submitted}
    trace['omitted'].extend(dict(id=c['id'], stage='listwise', reason='listwise_limit')
                            for c in fine if c['id'] not in submitted_ids and c not in ordered)
    for r in reservations:
        if r['candidate_id'] not in submitted_ids:
            omission = next((o for o in reversed(trace['omitted']) if o['id'] == r['candidate_id']), {})
            reason = omission.get('reason', 'capacity')
            if reason == 'listwise_limit':
                reason = 'capacity'
            trace['reservation_missing'].extend(dict(requirement_id=rid,candidate_id=r['candidate_id'],
                stage='listwise',reason=reason) for rid in r['requirement_ids'] or [None])
    selected, irrelevant, status, errors = list(submitted), set(), 'not_run', []
    attempts = []
    if submitted:
        seconds = min(config.RERANK_DEADLINE_SECONDS, max(.01, limits.deadline - time.monotonic() - 2)) if limits else config.RERANK_DEADLINE_SECONDS
        try:
            async with asyncio.timeout(seconds):
                for attempt in range(1 + min(1, config.RERANK_REPAIR_MAX_CALLS)):
                    stage = 'search.listwise' + ('.repair' if attempt else '')
                    try:
                        request = listwise_request(req, plan, submitted, repair=bool(attempt))
                        reservation = request.pop('reservation')
                        prompt = request.pop('prompt')
                        if limits:
                            limits.reserve_tokens(reservation)
                        try:
                            result = await llm.complete_json(prompt, **request,
                                stage=stage, attempts=1, timeout=min(20, seconds))
                        finally:
                            if limits:
                                limits.release_tokens(reservation)
                        permutation, irrelevant, groups = validate(result, len(submitted))
                        selected = [submitted[i] for i in permutation if i not in irrelevant]
                        plan['_evidence_groups'] = [[submitted[i]['id'] for i in group] for group in groups]
                        attempts.append(dict(stage=stage, status='ok'))
                        status = 'recovered' if attempt else 'ok'
                        break
                    except Exception as exc:
                        attempts.append(dict(stage=stage, status='error', error=type(exc).__name__))
                        cause, seen = exc, set()
                        while cause.__cause__ is not None and id(cause) not in seen:
                            seen.add(id(cause))
                            cause = cause.__cause__
                        if not isinstance(cause, (ValueError, TypeError, KeyError)) or attempt == min(1, config.RERANK_REPAIR_MAX_CALLS):
                            raise
        except Exception as exc:
            status = 'fallback'
            errors.append(dict(stage='listwise', error=type(exc).__name__))
    elif fine:
        status = 'fallback'
        errors.append(dict(stage='listwise', error='BudgetExceeded' if token_limited else 'prompt_budget'))
        if token_limited:
            # The budget cannot fund even one complete listwise input. Preserve
            # CE's ordering and the existing bounded fallback, rather than lose
            # all successfully scored evidence solely on LLM admission.
            selected = list(ordered)
    trace['listwise'] = dict(status=status, input_count=len(submitted), output_count=len(selected),
        candidate_ids=[c['id'] for c in submitted], selected_ids=[c['id'] for c in selected],
        input_bytes=len(listwise_prompt(req, plan, submitted).encode('utf-8')) if submitted else 0,
        attempts=attempts, errors=errors, selection=admission_selection, carried_ids=sorted(carried_ids))
    for i in irrelevant:
        trace['omitted'].append(dict(id=submitted[i]['id'], stage='listwise', reason='irrelevant'))
    if status == 'fallback':
        unknown = [c for c in selected if c.get('_ce_score') is None]
        excess = {c['id'] for c in unknown[config.EVIDENCE_FALLBACK_ITEMS:]}
        selected = [c for c in selected if c['id'] not in excess]
        trace['omitted'].extend(dict(id=mid, stage='fallback', reason='fallback_limit') for mid in sorted(excess))
        trace['listwise']['output_count'] = len(selected)
        trace['listwise']['selected_ids'] = [c['id'] for c in selected]
    selected = rules + selected
    selected_ids = {c['id'] for c in selected}
    for index, c in enumerate(selected):
        c['_cascade_selected'], c['_listwise_rank'] = True, index
        if protected(c):
            c.pop('_final', None)
            c['_score_kind'] = 'constraint'
        elif c.get('_final') is None:
            c['_score_kind'] = 'listwise' if status in ('ok', 'recovered') else 'graph_fallback'
        plan['_rerank'].append(dict(id=c['id'], reason='kept', score=c.get('_final'), keep=True))
    for c in candidates:
        if c['id'] not in selected_ids:
            c['_cascade_selected'] = False
    plan['_rerank_errors'] = [dict(stage='cross_encoder', **error) for error in ce_errors] + errors
    plan['_rerank_status'] = ('not_run' if not ordinary else 'partial' if plan['_rerank_errors'] and selected
        else 'fallback' if plan['_rerank_errors'] else 'recovered' if status == 'recovered' else 'ok')
    return selected + [c for c in candidates if c['id'] not in selected_ids]
