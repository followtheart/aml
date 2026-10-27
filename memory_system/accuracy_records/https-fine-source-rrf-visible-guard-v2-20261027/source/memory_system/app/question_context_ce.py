"""One optional whole-source seat ranked against the original question alone.

The existing lexical packet is immutable. This stage neither promotes facts
nor expands visible sources, and shares the original 5 KiB context allowance.
"""
import copy
import hashlib
import math
import time

from . import budget, config, cross_encoder, evidence_packet, local_work
from .fine_context_lexical import POLICY as PREFIX_POLICY
from .fine_context_reserve import propose as reserve

POLICY = 'question_ce_residual_context_v1'
STAGE = 'search.question_context_ce'


def prepare(packed, fine, *, user_id, top_k, byte_budget, reserve_bytes, query):
    """Score only complete units which individually fit after the fixed prefix."""
    candidates, documents, omitted = [], [], []
    for candidate in fine:
        trial, details = reserve(packed, [candidate], user_id=user_id, top_k=top_k,
            byte_budget=byte_budget, reserve_bytes=reserve_bytes, limit=1)
        if not details['appended_ids']:
            continue
        document = trial[-1]['content']
        if len(document.encode()) > config.CE_MAX_DOCUMENT_BYTES:
            omitted.append(dict(candidate_id=candidate['id'], reason='document_bytes'))
            continue
        candidates.append(candidate)
        documents.append(document)
    bound = cross_encoder.request_size(query, documents)
    status = ('not_needed' if not candidates else 'request_bounds' if
        len(candidates) > config.CE_BATCH_SIZE or bound > config.CE_MAX_REQUEST_BYTES or
        not query.strip() or len(query.encode()) > config.CE_MAX_DOCUMENT_BYTES else 'ready')
    trace = dict(candidate_ids=[c['id'] for c in candidates],
        query_sha256=hashlib.sha256(query.encode()).hexdigest(),
        document_sha256=[hashlib.sha256(d.encode()).hexdigest() for d in documents],
        request_bound=bound, omitted=omitted, status=status,
        reserve_bytes=reserve_bytes, scores=[])
    return candidates, documents, trace


def _budget_status(limits, request_bound):
    if limits is None:
        return 'missing_budget', None
    state = dict(calls=limits.calls, max_calls=limits.max_calls,
        reserved_calls=limits.reserved_calls, tokens=limits.tokens,
        max_tokens=limits.max_tokens, reserved_tokens=limits.reserved_tokens,
        seconds_remaining=limits.deadline-time.monotonic(), cancelled=limits.cancelled.is_set())
    return ('ready' if _has_budget(state, request_bound) else 'budget_skipped'), state


def _has_budget(state, bound):
    return (not state['cancelled'] and state['calls']+state['reserved_calls'] < state['max_calls'] and
        state['tokens']+state['reserved_tokens']+bound <= state['max_tokens'] and
        state['seconds_remaining'] >= config.CE_TIMEOUT_SECONDS+1)


def finish(packed, packet_hash, manifest, candidates, trace, *, user_id, top_k):
    """Rebuild hashes and shared accounting without changing prefix content."""
    if evidence_packet.digest(packed) != packet_hash:
        raise ValueError('Question CE prefix digest mismatch')
    context = manifest['conversation_context']
    if context['policy'] != PREFIX_POLICY or context['reserve_bytes'] != 5120:
        raise ValueError('Question CE requires the original 5 KiB lexical prefix')
    if trace['status'] == 'scored':
        scores = trace['scores']
        if (len(scores) != len(candidates) or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in scores)):
            raise ValueError('Question CE must score every candidate exactly once')
        ordered = [c for _, c in sorted(zip(scores, candidates), key=lambda v: -v[0])]
    else:
        if trace['scores']:
            raise ValueError('Unscored question CE trace contains scores')
        ordered = []
    result, extra = reserve(packed, ordered, user_id=user_id, top_k=top_k,
        byte_budget=manifest['token_budget'], reserve_bytes=5120-context['added_bytes'], limit=1)
    if result[:len(packed)] != packed:
        raise ValueError('Question CE changed the accepted prefix')
    trace = dict(trace, proposal=extra)
    combined = dict(context, policy=POLICY, limit=4,
        prefix_context=copy.deepcopy(context), prefix_packet_hash=packet_hash,
        prefix_packet_ids=[p['id'] for p in packed], question_ce=trace,
        proposed=context['proposed']+extra['proposed'],
        appended_ids=context['appended_ids']+extra['appended_ids'],
        added_bytes=context['added_bytes']+extra['added_bytes'],
        result_count=len(result), total_bytes=extra['total_bytes'])
    combined['status'] = 'appended' if combined['appended_ids'] else 'not_needed'
    updated = dict(manifest, conversation_context=combined)
    if extra['appended_ids']:
        packet_hash = evidence_packet.digest(result)
        result = [dict(p, packet_hash=packet_hash) for p in result]
        updated.update(included_ids=[p['id'] for p in result], token_upper_bound=extra['total_bytes'],
            evidence_count=manifest['evidence_count']+len(extra['appended_ids']))
    return result, packet_hash, updated


async def attach(packed, packet_hash, manifest, fine, *, user_id, top_k, query):
    context = manifest['conversation_context']
    candidates, documents, trace = await local_work.run(prepare, packed, fine, user_id=user_id,
        top_k=top_k, byte_budget=manifest['token_budget'],
        reserve_bytes=5120-context['added_bytes'], query=query)
    trace['budget_before'] = None
    if trace['status'] == 'ready':
        trace['status'], trace['budget_before'] = _budget_status(budget.current.get(), trace['request_bound'])
        if trace['status'] == 'ready':
            try:
                trace['scores'] = await cross_encoder.rerank(query, documents, stage=STAGE)
                trace['status'] = 'scored'
            except Exception as exc:
                # CancelledError remains a BaseException and is never swallowed.
                trace.update(status='error', error_type=type(exc).__name__)
    return await local_work.run(finish, packed, packet_hash, manifest, candidates, trace,
        user_id=user_id, top_k=top_k)


def replay(packed, packet_hash, manifest, fine, trace, *, user_id, top_k, query):
    """Offline reproduction validates the full request before using saved scores."""
    candidates, _, expected = prepare(packed, fine, user_id=user_id, top_k=top_k,
        byte_budget=manifest['token_budget'], reserve_bytes=5120-manifest['conversation_context']['added_bytes'],
        query=query)
    for key in ('candidate_ids', 'query_sha256', 'document_sha256', 'request_bound', 'omitted', 'reserve_bytes'):
        if trace[key] != expected[key]:
            raise ValueError('Question CE replay request mismatch: '+key)
    if expected['status'] != 'ready':
        if trace['status'] != expected['status'] or trace['budget_before'] is not None:
            raise ValueError('Question CE dispatched an ineligible request')
    elif trace['status'] == 'missing_budget':
        if trace['budget_before'] is not None:
            raise ValueError('Question CE missing budget mismatch')
    elif trace['status'] in ('scored', 'error', 'budget_skipped'):
        state = trace['budget_before']
        if not state or (_has_budget(state, expected['request_bound']) != (trace['status'] != 'budget_skipped')):
            raise ValueError('Question CE replay budget mismatch')
        if trace['status'] == 'error' and not trace.get('error_type'):
            raise ValueError('Question CE error is missing its type')
    else:
        raise ValueError('Unknown question CE result status')
    result = finish(packed, packet_hash, manifest, candidates, trace, user_id=user_id, top_k=top_k)
    if result[2]['conversation_context']['question_ce'] != trace:
        raise ValueError('Question CE proposal does not reproduce')
    return result
