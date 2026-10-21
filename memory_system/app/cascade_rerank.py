"""Bounded coarse selection -> provider Cross-Encoder -> LLM listwise selection."""
import asyncio
import json
import math
import re
import time
import httpx
from . import budget, config, cross_encoder, evidence_selection, fine_evidence, listwise_contract, listwise_recovery, llm, personal_evidence, profile, prompts, rerank_query

# Wording shared by every forget/instruction rule; it must not count as topical overlap.
_RULE_BOILERPLATE = set('user asked assistant forget forgot forgotten requested request information detail details '
                        'never mention mentioned remember delete remove erase told said should always please '
                        'recently anymore longer previously earlier'.split())


def protected(item):
    return bool(item.get('_user_rule')) or profile.is_forget_rule(item)


def rule_overlap(item, needles):
    """Topical terms a rule shares with the question and its options."""
    text = item.get('_rank_text', item.get('content', ''))
    return len((personal_evidence.terms(text) - _RULE_BOILERPLATE) & needles)


def cap_rules(req, plan, rules):
    """Keep only the rules that can bear on this question.

    Every protected rule used to bypass ranking and ride into the packet, so a
    user with many forget requests spent most of the evidence budget on
    constraints unrelated to the question. Rules are ordered by topical
    overlap with the query/options, then by fused retrieval score, and cut at
    PACKET_RULE_LIMIT. Dropped rules are recorded in the cascade trace.
    """
    limit = config.PACKET_RULE_LIMIT
    needles = personal_evidence.terms(' '.join([req.query] + list(req.options or []))) - _RULE_BOILERPLATE
    overlaps = {c['id']: rule_overlap(c, needles) for c in rules}
    ordered = sorted(rules, key=lambda c: (-overlaps[c['id']], -(c.get('_fused') or 0), c['id']))
    kept = ordered if limit <= 0 else ordered[:limit]
    dropped = ordered[len(kept):]
    plan['_cascade']['rule_cap'] = dict(limit=limit, input_count=len(rules), kept_ids=[c['id'] for c in kept],
                                       dropped=[dict(id=c['id'], overlap=overlaps[c['id']]) for c in dropped])
    return kept, dropped


def shortlist(rows, count, requirements, key, reserve_ids=(), *, trace=None, carry_ids=(), fused_ids=(),
              bound_soft_reservations=False):
    return evidence_selection.select(rows, count, requirements, key, reserve_ids,
                                     trace=trace, carry_ids=carry_ids, fused_ids=fused_ids,
                                     bound_soft_reservations=bound_soft_reservations)


def query_text(req, plan):
    return rerank_query.build(req, plan)


def listwise_prompt(req, plan, rows):
    ids = {c['id'] for c in rows}
    triples = [t for t in plan.get('_fusion_triples', []) if t.get('amu_id') in ids]
    edges = [e for e in plan.get('_fusion_edges', []) if e['source'] in ids and e['target'] in ids]
    metadata = dict(triples=[{k: t.get(k) for k in ('amu_id', 'subject', 'relation', 'object', 'valid_from', 'valid_to')}
                            for t in triples[:40]], associations=edges[:40])
    incomplete = {c['id']: c['_rank_context']['reasons'] for c in rows
                  if c.get('_rank_context', {}).get('context_incomplete')}
    if incomplete:
        metadata['incomplete_candidate_context'] = incomplete
    if plan.get('_fine_path_groups'):
        metadata['recorded_paths_for_joint_review'] = [
            g for g in plan['_fine_path_groups'] if set(g) <= ids]
        metadata['path_scope'] = 'Recorded retrieval relations, not proof. Inspect these sources together.'
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
    """Compatibility validator for the historical permutation protocol."""
    state = listwise_contract.State(count)
    state.accept(result)
    return state.require_valid()


def listwise_retry_kind(exc):
    """Classify the full exception chain without retrying permanent failures."""
    seen, chain = set(), []
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        chain.append(exc)
        exc = exc.__cause__
    if any(isinstance(e, budget.BudgetExceeded) for e in chain):
        return None
    if any(getattr(e, 'status_code', None) in (400, 401, 403, 404, 422)
           or getattr(getattr(e, 'response', None), 'status_code', None) in (400, 401, 403, 404, 422)
           for e in chain):
        return None
    if any(getattr(e, 'validation_code', None) or isinstance(e, (ValueError, TypeError, KeyError))
           for e in chain):
        return 'structure'
    if any(isinstance(e, (TimeoutError, httpx.TimeoutException, httpx.NetworkError))
           or getattr(e, 'status_code', None) in (500, 502, 503, 504)
           or getattr(getattr(e, 'response', None), 'status_code', None) in (500, 502, 503, 504)
           for e in chain):
        return 'transport'
    return None


def listwise_request(req, plan, rows, *, repair=False, shared_sources=None, feedback=None,
                     repair_field=None, locked=None, keyed=None, candidate_keys=None):
    """Use the same full request bound for scheduling and provider dispatch."""
    prompt = listwise_prompt(req, plan, rows)
    if rows and (config.LISTWISE_SHARED_SOURCES if shared_sources is None else shared_sources):
        from .listwise_sources import prompt_with_shared
        compact = prompt_with_shared(listwise_prompt)(req, plan, rows)
        if len(compact.encode('utf-8')) < len(prompt.encode('utf-8')):
            prompt = compact
    keyed = config.LISTWISE_KEYED_OUTPUT if keyed is None else keyed
    output_limit = max(256, (48 if keyed else 24) * len(rows))
    group_keys = ([k for k, v in (locked or {}).get('candidates', {}).items()
                   if v.get('relevance') != 'irrelevant'] if keyed and repair_field == 'groups' else None)
    output_schema = (listwise_contract.schema(len(rows), keyed=keyed, field=repair_field,
                                            candidate_keys=candidate_keys, group_keys=group_keys)
                     if keyed or repair_field else schema(len(rows)))
    system = ('Rank the supplied evidence for the question and call emit_json_result. '
              'Return every input index once in ranking. Mark unrelated items in irrelevant. '
              'Default groups to []; create a group only for 2 to 4 complementary items '
              'that must be read together. Never copy the ranking list into groups.')
    if keyed or repair_field:
        prompt = prompt.rsplit('\nReturn ONLY JSON', 1)[0]
        fields = list(output_schema['properties'])
        prompt += '\nListwise output fields: ' + json.dumps(fields)
        prompt += '\nCandidate count: ' + str(len(rows))
        if keyed:
            prompt = prompt.replace('Return candidate indices, never source indices.',
                                    'Return candidate output keys, never source indices.', 1)
            prompt += ('\nCandidate output keys: ' + ', '.join(f'c{i}' for i in range(len(rows))) +
                       '\nKey cN identifies candidate index N, never a source reference. '
                       'Groups use candidate keys such as c0, never source IDs.')
            if 'candidates' in fields:
                prompt += (' Compare ALL candidates together. For each requested candidate key return priority '
                           '(integer 0..100, lower is better; ties allowed) and relevance '
                           '("useful", "uncertain", or "irrelevant"). Use uncertain when unsure; '
                           'do not label potentially useful evidence irrelevant.')
        if candidate_keys is not None:
            prompt += '\nRequested candidate keys: ' + json.dumps(candidate_keys)
        prompt += ('\nReturn ONLY the requested JSON fields, no others. '
                   'Default groups to []; groups must be disjoint, contain 2 to 4 '
                   'complementary candidates, and exclude irrelevant candidates.')
        system = ('Compare the supplied evidence together and call emit_json_result. '
                  'Follow the requested output fields exactly. Candidate priorities are '
                  'relative ordering judgments, not probabilities or factual support.')
        if not keyed:
            system += ' Ranking must contain every candidate index exactly once, including irrelevant items.'
        if repair_field:
            context = locked or {}
            if repair_field == 'candidates' and keyed:
                context = {'immutable_candidate_keys': list((locked or {}).get('candidates', {}))}
            if repair_field == 'groups':
                # Group validation needs identities and exclusions, not priorities.
                excluded = ([k for k, v in context.get('candidates', {}).items()
                             if v.get('relevance') == 'irrelevant'] if keyed
                            else context.get('irrelevant', []))
                context = {'excluded_from_groups': excluded}
            prompt += '\nValidated constraints (immutable): ' + json.dumps(context, separators=(',', ':'))
            system += f' Repair ONLY {repair_field}. Do not emit or change the validated fields.'
    if repair:
        system += (' Regenerate only the requested fields from the same candidate indices. '
                   'Include every candidate required by the output schema. '
                   'Groups must be disjoint, exclude irrelevant indices and contain 2 to 4 members.')
        if feedback:
            system += ' Validator feedback: ' + json.dumps(feedback, sort_keys=True)
    reservation = (len(prompt.encode('utf-8')) + len(json.dumps(output_schema).encode('utf-8'))
                   + len(system.encode('utf-8')) + 1024 + output_limit)
    return dict(prompt=prompt, schema=output_schema, max_tokens=output_limit,
                system=system, reservation=reservation)


async def rank(req, plan, candidates):
    trace = plan['_cascade'] = dict(coarse={}, cross_encoder={}, fine={}, listwise={}, omitted=[],
                                   reservation_missing=[])
    plan['_rerank'], plan['_evidence_groups'] = [], []
    plan.pop('_fine_path_groups', None)
    if not candidates:
        plan['_rerank_status'] = 'not_run'
        return []
    rules, dropped_rules = cap_rules(req, plan, [c for c in candidates if protected(c)])
    trace['omitted'].extend(dict(id=c['id'], stage='rules', reason='rule_limit') for c in dropped_rules)
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
            request = listwise_request(req, plan, [c], shared_sources=False)
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
                         retryable_timeout=isinstance(exc, (TimeoutError, httpx.TimeoutException)),
                         retryable_transport=isinstance(exc, (TimeoutError, httpx.TimeoutException,
                                                               httpx.RemoteProtocolError, httpx.ConnectError)))
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
            # Retry transient transport failures after every initial wave has settled.
            # Keep the original deadline, shared budget and listwise reservation.
            timed_out = {mid for b in batch_traces if b.get('retryable_transport')
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
        fine, fine_selection = fine_evidence.propose(coarse, min(config.CASCADE_FINE_LIMIT, len(coarse)),
            requirements, rescue_ids=failed_ids, fused_ids=fused_ids, extra_limit=config.FINE_EXTRA_LIMIT)
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
    if config.LISTWISE_STAGED and submitted:
        from . import listwise_staged
        seconds = min(config.RERANK_DEADLINE_SECONDS, max(.01, limits.deadline - time.monotonic() - 2)) if limits else config.RERANK_DEADLINE_SECONDS
        selected, irrelevant, staged = await listwise_staged.select(
            req, plan, submitted, deadline=time.monotonic() + seconds)
        trace['listwise'] = staged
        trace['listwise'].update(baseline_candidate_ids=sorted(baseline_ids),
                                 selection=admission_selection, carried_ids=sorted(carried_ids))
        status, errors, recovery = staged['status'], staged['errors'], staged['recovery']
    else:
        selected, irrelevant, status, errors = list(submitted), set(), 'not_run', []
        attempts = []
        feedback, retry_kind, retained_ranking = None, None, None
        keyed = config.LISTWISE_KEYED_OUTPUT
        contract = listwise_contract.State(len(submitted), keyed=keyed,
            tie_order=sorted(range(len(submitted)),
                             key=lambda i: fine_evidence.key(submitted[i]), reverse=True))
        repair_field, repairs, transport_retries = None, 0, 0
        repair_limit = min(2, config.RERANK_REPAIR_MAX_CALLS)
        listwise_deadline = None
        if submitted:
            seconds = min(config.RERANK_DEADLINE_SECONDS, max(.01, limits.deadline - time.monotonic() - 2)) if limits else config.RERANK_DEADLINE_SECONDS
            listwise_deadline = time.monotonic() + seconds
            try:
                async with asyncio.timeout(seconds):
                    for attempt in range(1 + repair_limit + int(repair_limit > 0)):
                        stage = 'search.listwise' + (('.retry' if retry_kind == 'transport' else '.repair') if attempt else '')
                        dispatched = False
                        try:
                            remaining = listwise_deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError('Listwise deadline exhausted')
                            if limits and limits.calls + limits.reserved_calls >= limits.max_calls:
                                raise budget.BudgetExceeded('No listwise call slot remaining')
                            request = listwise_request(req, plan, submitted,
                                repair=repair_field is not None or bool(attempt and retry_kind == 'structure'),
                                feedback=feedback, repair_field=repair_field, keyed=keyed,
                                candidate_keys=contract.pending_candidate_keys() if keyed and repair_field == 'candidates' else None,
                                locked=dict({k: v for k, v in contract.payload.items()
                                             if k != repair_field and k not in contract.errors},
                                            **({'candidates': contract.accepted_candidates} if keyed else {})))
                            reservation = request.pop('reservation')
                            prompt = request.pop('prompt')
                            if len(prompt.encode('utf-8')) > config.RERANK_MAX_PROMPT_BYTES:
                                raise budget.BudgetExceeded('Listwise prompt exceeds byte budget')
                            if limits:
                                limits.reserve_tokens(reservation)
                            try:
                                dispatched = True
                                result = await llm.complete_json(prompt, **request,
                                    stage=stage, attempts=1, timeout=min(20, remaining))
                            finally:
                                if limits:
                                    limits.release_tokens(reservation)
                            contract.accept(result, repair_field)
                            permutation, irrelevant, groups = contract.require_valid()
                            selected = [submitted[i] for i in permutation if i not in irrelevant]
                            plan['_evidence_groups'] = [[submitted[i]['id'] for i in group] for group in groups]
                            attempts.append(dict(stage=stage, status='ok', repair_field=repair_field, dispatched=dispatched))
                            status = 'recovered' if attempt else 'ok'
                            break
                        except Exception as exc:
                            attempts.append(dict(stage=stage, status='error', error=type(exc).__name__, dispatched=dispatched))
                            attempts[-1]['repair_field'] = repair_field
                            if getattr(exc, 'validation_code', None):
                                attempts[-1]['validation_code'] = exc.validation_code
                                feedback = dict(getattr(exc, 'validation_details', {}))
                                feedback['code'] = exc.validation_code
                                attempts[-1]['validation_details'] = feedback
                            if getattr(exc, 'valid_ranking', None) is not None:
                                retained_ranking = exc.valid_ranking
                            if contract.ranking is not None:
                                retained_ranking = list(contract.ranking)
                            retry_kind = listwise_retry_kind(exc)
                            if retry_kind is None:
                                raise
                            if retry_kind == 'transport':
                                if transport_retries >= 1 or repair_limit == 0:
                                    raise
                                transport_retries += 1
                                await asyncio.sleep(.1)
                            else:
                                if repairs >= repair_limit:
                                    raise
                                repairs += 1
                                repair_field = contract.next_field()
            except Exception as exc:
                groups_only = (contract.ranking is not None and contract.irrelevant is not None
                               and set(contract.errors) == {'groups'})
                status = 'groups_partial' if groups_only else 'ranking_only' if retained_ranking is not None else 'fallback'
                if groups_only:
                    # Group metadata is optional. Keep validated ranking/relevance and
                    # still send every proposed deletion through source-grounded review.
                    irrelevant = set(contract.irrelevant)
                    selected = [submitted[i] for i in contract.ranking if i not in irrelevant]
                    plan['_evidence_groups'] = []
                elif retained_ranking is not None:
                    selected = [submitted[i] for i in retained_ranking]
                    irrelevant = set()
                    plan['_evidence_groups'] = []
                errors.append(dict(stage='listwise_groups' if groups_only else 'listwise', error=type(exc).__name__))
        elif fine:
            status = 'fallback'
            errors.append(dict(stage='listwise', error='BudgetExceeded' if token_limited else 'prompt_budget'))
            if token_limited:
                # The budget cannot fund even one complete listwise input. Preserve
                # CE's ordering and the existing bounded fallback, rather than lose
                # all successfully scored evidence solely on LLM admission.
                selected = list(ordered)
        primary_selected_ids = [c['id'] for c in selected]
        recovery = dict(status='not_needed', reviewed_ids=[], restored_ids=[], unreviewed_ids=[], judgments=[], errors=[])
        if status in ('ok', 'recovered', 'groups_partial') and irrelevant:
            selected, recovery = await listwise_recovery.recover(req, submitted, selected, irrelevant,
                                                                deadline=listwise_deadline)
            errors.extend(dict(stage='listwise_recovery', **error) for error in recovery['errors'])
            if recovery['status'] == 'recovered' and status != 'groups_partial':
                status = 'recovered'
            # Review owns deletion authorization. Unresolved candidates are retained
            # below accepted useful evidence, bounded by the admitted input set.
            # Final packet capacity is enforced separately and is not irrelevance.
        trace['listwise'] = dict(status=status, input_count=len(submitted), output_count=len(selected),
            output_protocol='keyed_candidates_v1' if keyed else 'permutation_v1',
            candidate_contract_status='valid' if contract.ranking is not None and contract.irrelevant is not None else 'invalid',
            groups_status='valid' if contract.groups is not None else 'invalid',
            accepted_candidate_keys=sorted(contract.accepted_candidates),
            ignored_candidate_keys=sorted(contract.ignored_candidate_keys),
            repair_calls=sum(a['dispatched'] and a['stage'] == 'search.listwise.repair' for a in attempts),
            transport_retries=sum(a['dispatched'] and a['stage'] == 'search.listwise.retry' for a in attempts),
            candidate_key_map={f'c{i}': c['id'] for i, c in enumerate(submitted)} if keyed else {},
            candidate_ids=[c['id'] for c in submitted], selected_ids=[c['id'] for c in selected],
            input_bytes=len(listwise_request(req, plan, submitted)['prompt'].encode('utf-8')) if submitted else 0,
            shared_sources=config.LISTWISE_SHARED_SOURCES,
            baseline_candidate_ids=sorted(baseline_ids),
            attempts=attempts, errors=errors, selection=admission_selection, carried_ids=sorted(carried_ids),
            primary_selected_ids=primary_selected_ids, primary_irrelevant_ids=[submitted[i]['id'] for i in sorted(irrelevant)],
            recovery=recovery)
    for i in irrelevant:
        if submitted[i]['id'] not in {c['id'] for c in selected}:
            reason = ('represented' if any(r['candidate_id'] == submitted[i]['id']
                       for r in recovery.get('represented', [])) else 'irrelevant_confirmed')
            trace['omitted'].append(dict(id=submitted[i]['id'], stage='listwise', reason=reason))
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
            c['_score_kind'] = 'listwise' if status in ('ok', 'recovered', 'ranking_only', 'groups_partial') or (
                config.LISTWISE_STAGED and trace['listwise'].get('ranking_status') == 'ok') else 'graph_fallback'
        plan['_rerank'].append(dict(id=c['id'], reason='kept', score=c.get('_final'), keep=True))
    for c in candidates:
        if c['id'] not in selected_ids:
            c['_cascade_selected'] = False
    plan['_rerank_errors'] = [dict(stage='cross_encoder', **error) for error in ce_errors] + errors
    plan['_rerank_status'] = ('not_run' if not ordinary else 'partial' if plan['_rerank_errors'] and selected
        else 'fallback' if plan['_rerank_errors'] else 'recovered' if status == 'recovered' else 'ok')
    return selected + [c for c in candidates if c['id'] not in selected_ids]
