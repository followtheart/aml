"""Default-off source projection, complete option scoring and one semantic gate.

Only already filtered, hydrated coarse evidence is eligible. No DB expansion,
answer labels, question IDs or partial-matrix fallback ranking are used here.
"""
import asyncio
import copy
import hashlib
import json
import time

from . import (answer_context, budget, config, cross_encoder, fine_evidence,
               llm, local_candidate_display as display, option_margin_policy as policy,
               option_margin_review as review, rank_constraint_catalog,
               source_projection_coverage as coverage, response_observer)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def identity(source):
    return tuple(source.get(k) for k in
                 ('user_id', 'request_id', 'message_index', 'source_event_id', 'role'))


def project_visible(parent, source, user_id):
    """Project a visible source; upstream snapshot owns DB/lifecycle validation."""
    item = parent.get('_packet_item', {})
    if parent.get('view_status') == 'stale' or parent.get('_user_rule') or item.get('is_constraint'):
        return None
    if source.get('role') != 'user' or source.get('user_id') != user_id:
        return None
    if (not source.get('request_id') or not source.get('source_event_id')
            or type(source.get('message_index')) is not int):
        return None
    body = source.get('content')
    if (not isinstance(body, str) or not body.strip()
            or source.get('content_omitted') or source.get('content_excerpted')):
        return None
    span = source.get('content_span')
    if span is not None and (span.get('start'), span.get('end'), span.get('original_length')) != (0, len(body), len(body)):
        return None
    if body not in parent.get('_rank_text', '') or body not in item.get('content', ''):
        return None
    mid = 'source_user_' + digest(source)
    sources = [copy.deepcopy(source)]
    content = answer_context.with_evidence('Conversation excerpt:', sources)
    packet = dict(id=mid, content=content, memory_type='episode', personal_evidence=False,
                  is_constraint=False, coverage_ids=[], equivalent_ids=[], sources=sources,
                  source_count=1, packet_hash_version=4, created_at=item.get('created_at'))
    return dict(id=mid, content=content, type='episode', _rank_text=content,
                _packet_item=packet, _rank_context=copy.deepcopy(parent.get('_rank_context', {})))


def safe_slot(plan, coarse, fine, main, carried_ids, cascade):
    pool = {c['id']: c for c in coarse}
    pinned = (set(plan['_cascade']['fine']['selection'].get('adaptive', {}).get('core_ids', []))
              | set(carried_ids) | {c['id'] for c in main})
    atoms = {c['id']: coverage.visible_atoms(c) for c in fine}
    original_graph = cascade.listwise_graph(plan, fine)
    for candidate in reversed(fine):
        mid = candidate['id']
        if mid in pinned or rank_constraint_catalog.catalog([candidate]):
            continue
        remaining = [c for c in fine if c['id'] != mid]
        ids = {c['id'] for c in remaining}
        if any(missing or not group <= ids for group, missing in
               (fine_evidence.closure(c, pool) for c in remaining)):
            continue
        if display.missing_graph(original_graph, cascade.listwise_graph(plan, remaining)):
            continue
        available = [atom for c in remaining for atom in (atoms[c['id']] or [])]
        if coverage.covered(atoms[mid], available):
            return mid
    return None


def prepare(req, plan, coarse, fine, main, carried_ids, headroom, cascade):
    slot = safe_slot(plan, coarse, fine, main, carried_ids, cascade)
    job = dict(slot=slot, nominees=[], views={}, queries=[], batches=[], main_reservation=0)
    if slot is None:
        return dict(job, reason='no_source_preserving_slot')
    if len(fine) > 24 or len(main) >= 24:
        return dict(job, reason='capacity')
    pool = {c['id']: c for c in coarse}
    known = [s for c in main for s in c['_packet_item']['sources'] if s.get('content')]
    graphs = [cascade.listwise_graph(plan, rows) for rows in (fine, main)]
    dedup = {}
    for parent in coarse:
        budget.check()
        group, missing = fine_evidence.closure(parent, pool)
        if missing or group != {parent['id']} or rank_constraint_catalog.catalog([parent]):
            continue
        for source in parent['_packet_item'].get('sources', []):
            unit = project_visible(parent, source, req.user_id)
            if unit is None or any(identity(s) == identity(source) and source['content'] in s['content'] for s in known):
                continue
            key = unit['id'], json.dumps(unit['_rank_context'], sort_keys=True)
            if key in dedup:
                dedup[key]['parent_ids'].append(parent['id'])
                continue
            trial_fine = [unit if c['id'] == slot else c for c in fine]
            trial_main = main + [unit]
            if not {c['id'] for c in trial_main} <= {c['id'] for c in trial_fine}:
                continue
            if any(display.missing_graph(before, cascade.listwise_graph(plan, rows))
                   for before, rows in zip(graphs, (trial_fine, trial_main))):
                continue
            request, audit = display.request(req, plan, trial_main, cascade)
            size = len(request['prompt'].encode())
            if size > min(24000, config.RERANK_MAX_PROMPT_BYTES) or request['reservation'] > headroom['token_headroom']:
                continue
            text = policy.PREFIX + source['content']
            if len(text.encode()) > config.CE_MAX_DOCUMENT_BYTES:
                continue
            view_id = digest(dict(identity=dict(zip(
                ('user_id', 'request_id', 'message_index', 'source_event_id', 'role'), identity(source))), text=source['content']))
            nominee = dict(id=unit['id'], unit=unit, parent_ids=[parent['id']],
                           replaced_fine_id=slot, trial_main_bytes=size, reservation=request['reservation'],
                           local_display_used=audit['used'], source_view_id=view_id,
                           parent_context_incomplete=bool(unit['_rank_context'].get('context_incomplete')))
            job['nominees'].append(nominee)
            dedup[key] = nominee
            job['views'][view_id] = text
    job['queries'] = [policy.query(req.query, option) for option in req.options]
    invalid = any(len(q.encode()) > config.CE_MAX_DOCUMENT_BYTES for q in job['queries'])
    for option, query in enumerate(job['queries']):
        current = []
        for key, text in job['views'].items():
            trial = current + [key]
            if current and (len(trial) > config.CE_BATCH_SIZE or
                            cross_encoder.request_size(query, [job['views'][k] for k in trial]) > config.CE_MAX_REQUEST_BYTES):
                job['batches'].append(dict(option=option, view_ids=current))
                current = []
            if cross_encoder.request_size(query, [text]) > config.CE_MAX_REQUEST_BYTES:
                invalid = True
            current.append(key)
        if current:
            job['batches'].append(dict(option=option, view_ids=current))
    for batch in job['batches']:
        batch['bound'] = cross_encoder.request_size(job['queries'][batch['option']],
                                                   [job['views'][k] for k in batch['view_ids']])
    job['main_reservation'] = max((n['reservation'] for n in job['nominees']), default=0)
    reason = ('no_complete_projection' if not job['nominees'] else 'option_count' if len(job['queries']) < 2
              else 'query_or_document_bytes' if invalid else 'initial_batch_headroom'
              if max((b['bound'] for b in job['batches']), default=0) + job['main_reservation'] > headroom['token_headroom']
              else 'call_headroom' if len(job['batches']) + 2 > headroom['call_headroom'] else None)
    return dict(job, reason=reason)


def remaining(limits):
    return dict(token_headroom=limits.max_tokens-limits.tokens-limits.reserved_tokens,
                call_headroom=limits.max_calls-limits.calls-limits.reserved_calls,
                deadline_seconds=max(0, limits.deadline-time.monotonic()))


async def propose(req, plan, coarse, fine, main, carried_ids, cascade):
    details = dict(status='disabled', batches=[], responses=[], display_ids=False)
    if not config.OPTION_MARGIN_SOURCE_ADMISSION:
        return None, details
    limits = budget.current.get()
    if (limits is None or not main or not config.FINE_EVIDENCE_ADAPTIVE or config.LISTWISE_STAGED
            or config.SOURCE_CE_LATE_ADMISSION or config.JOINT_SOURCE_RANK_ADMISSION):
        return None, dict(details, status='skipped', reason='unsupported_policy_or_missing_budget')
    started = time.monotonic()
    try:
        details['original_headroom'] = remaining(limits)
        job = prepare(req, plan, coarse, fine, main, carried_ids, remaining(limits), cascade)
        details.update(slot=job['slot'], source_view_ids=list(job['views']),
                       nominee_ids=[n['id'] for n in job['nominees']])
        if job['reason']:
            return None, dict(details, status='skipped', reason=job['reason'])
        seconds = min(45, max(0, remaining(limits)['deadline_seconds'] - config.RERANK_DEADLINE_SECONDS - 2))
        if seconds <= 0:
            raise budget.BudgetExceeded('No source phase time')
        details.update(phase_seconds=seconds, main_reservation=job['main_reservation'])
        scores = {i: {} for i in range(len(job['queries']))}
        reserved_tokens = reserved_calls = 0
        try:
            limits.reserve_calls(1)
            reserved_calls = 1
            limits.reserve_tokens(job['main_reservation'])
            reserved_tokens = job['main_reservation']
            with response_observer.observe(details['responses'].append):
                async with asyncio.timeout(seconds):
                    for batch in job['batches']:
                        if batch['bound'] > remaining(limits)['token_headroom'] or remaining(limits)['call_headroom'] < 2:
                            raise budget.BudgetExceeded('Protect main and gate')
                        entry = dict(batch, status='started', tokens_before=limits.tokens, calls_before=limits.calls)
                        details['batches'].append(entry)
                        values = await cross_encoder.rerank(job['queries'][batch['option']],
                            [job['views'][k] for k in batch['view_ids']], stage='search.option_margin_source', rate_limit_retries=0)
                        if len(values) != len(batch['view_ids']):
                            raise ValueError('Incomplete source batch')
                        scores[batch['option']].update(zip(batch['view_ids'], values, strict=True))
                        entry.update(status='ok', scores=values, tokens_after=limits.tokens, calls_after=limits.calls)
                    winner = policy.rank(job['nominees'], scores, len(job['queries']))[0]
                    if winner['option_score_range'] <= 0:
                        details.update(status='skipped', reason='zero_score_range')
                        return None, details
                    details['winner'] = {k: v for k, v in winner.items() if k != 'unit'}
                    # The nomination phase protected the worst-case main request.
                    # Once the winner is fixed, release only its unused surplus.
                    limits.release_tokens(reserved_tokens-winner['reservation'])
                    reserved_tokens = winner['reservation']
                    source = winner['unit']['_packet_item']['sources'][0]
                    gate = review.build(req.query, req.options, [dict(id=winner['source_view_id'], text=source['content'])])
                    if len(gate['prompt'].encode()) > review.MAX_PROMPT_BYTES:
                        raise ValueError('Gate prompt exceeds bound')
                    details['gate_request_sha256'] = digest(gate)
                    limits.reserve_tokens(gate['reservation'])
                    try:
                        payload = await llm.complete_json(gate['prompt'], system=gate['system'], schema=gate['schema'],
                            max_tokens=gate['max_tokens'], stage='search.option_margin_gate',
                            timeout=max(.01, seconds-(time.monotonic()-started)), attempts=1, rate_limit_retries=0)
                    finally:
                        limits.release_tokens(gate['reservation'])
                    details['gate_payload'] = payload
                    judgments = review.validate(payload, gate)
                    details['judgments'] = judgments
                    if not judgments[0]['accepted']:
                        details.update(status='rejected', reason=judgments[0]['status'])
                        return None, details
                    if remaining(limits)['token_headroom'] < 0:
                        raise budget.BudgetExceeded('Protected main tokens consumed')
        finally:
            if reserved_tokens:
                limits.release_tokens(reserved_tokens)
            if reserved_calls:
                limits.release_calls(reserved_calls)
        trial_fine = [winner['unit'] if c['id'] == job['slot'] else c for c in fine]
        trial_main = main + [winner['unit']]
        request, _ = display.request(req, plan, trial_main, cascade)
        if (safe_slot(plan, coarse, fine, main, carried_ids, cascade) != job['slot']
                or len(trial_fine) > 24 or len(trial_main) > 24
                or len(request['prompt'].encode()) > min(24000, config.RERANK_MAX_PROMPT_BYTES)
                or request['reservation'] > remaining(limits)['token_headroom']
                or remaining(limits)['call_headroom'] < 1
                or remaining(limits)['deadline_seconds'] < config.RERANK_DEADLINE_SECONDS):
            raise budget.BudgetExceeded('Final admission guard')
        details.update(status='selected', display_ids=True, proposed_main_bytes=len(request['prompt'].encode()),
                       proposed_fine_ids=[c['id'] for c in trial_fine], proposed_main_ids=[c['id'] for c in trial_main])
        return dict(fine=trial_fine, main=trial_main, unit=winner['unit']), details
    except Exception as exc:
        details.update(status='fallback', reason=type(exc).__name__)
        return None, details
    finally:
        details.update(elapsed_seconds=time.monotonic()-started, remaining_budget=remaining(limits))


async def apply(req, plan, coarse, fine, main, carried_ids, candidates, cascade):
    result, details = await propose(req, plan, coarse, fine, main, carried_ids, cascade)
    trace = plan['_cascade']
    trace['option_margin_source'] = details
    if result is None:
        return fine, main, candidates, False
    details['original_stages'] = copy.deepcopy(dict(fine=trace['fine'], omitted=trace['omitted']))
    unit, removed = result['unit'], details['slot']
    fine, main = result['fine'], result['main']
    old = trace['fine']
    # The original adaptive decisions remain a historical stage, explicitly
    # nested, rather than pretending a synthetic unit was a coarse candidate.
    trace['fine'] = dict(input_count=len(coarse), output_count=len(fine),
        selected_ids=[c['id'] for c in fine], reserved_ids=old['reserved_ids'],
        fused_reserved_ids=old.get('fused_reserved_ids', []),
        selection=dict(policy='option_margin_source', original_selection=copy.deepcopy(old['selection']),
                       reservations=copy.deepcopy(old['selection']['reservations']),
                       projection=dict(id=unit['id'], replaced_id=removed, parent_ids=details['winner']['parent_ids'])))
    trace['omitted'] = [o for o in trace['omitted'] if o['id'] != removed or o['stage'] not in ('fine', 'listwise')]
    trace['omitted'].append(dict(id=removed, stage='fine', reason='source_preserved_replacement'))
    # Keep the synthetic candidate available to final fine-context packing even
    # when the main model does not select it as primary evidence.
    return fine, main, candidates + [unit], True
