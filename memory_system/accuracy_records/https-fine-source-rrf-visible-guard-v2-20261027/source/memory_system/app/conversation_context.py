"""Bounded original user context for a retained assistant reply.

Only an eligible fine supplier with identical assistant fragments and exactly
one preceding complete user turn may supply context. No inferred facts, new
coverage labels, retrieval calls, or displacement of the existing packet.
"""
import copy
import json
from . import answer_context, budget, evidence_packet, schemas

POLICY = 'adjacent_user_context_v1'


def source_key(source):
    return json.dumps({key: source.get(key) for key in (
        'request_id', 'message_index', 'source_event_id', 'user_id', 'role', 'content_span', 'content')},
        sort_keys=True, ensure_ascii=False)


def usable(source, user_id):
    return (source.get('user_id') == user_id and bool(source.get('source_event_id'))
            and bool(source.get('request_id')) and type(source.get('message_index')) is int
            and source['message_index'] >= 0
            and isinstance(source.get('content'), str) and bool(source['content'].strip()))


def propose(packed, fine, *, user_id, top_k, byte_budget, limit=2):
    """Preserve the original prefix and fit context within the same byte/item caps."""
    baseline = copy.deepcopy(packed)
    result = list(packed)
    used = sum(len(item['content'].encode()) for item in result) + max(0, len(result)-1)
    if len(result) > top_k or used > byte_budget:
        raise ValueError('Baseline packet already exceeds its limits')
    trace = dict(proposed=[], appended_ids=[], added_bytes=0)
    known = {source_key(s) for item in result for s in item.get('sources', []) if usable(s, user_id)}
    kept = {item['id'] for item in result}
    for anchor in packed:
        budget.check()
        sources = anchor.get('sources', [])
        if not sources or any(s.get('role') != 'assistant' or not usable(s, user_id) for s in sources):
            continue
        assistant_keys = {source_key(s) for s in sources}
        for candidate in fine:
            budget.check()
            mid = candidate['id']
            item = candidate.get('_packet_item', {})
            if mid in kept or candidate.get('view_status') == 'stale' or item.get('is_constraint'):
                continue
            all_sources = item.get('sources', [])
            if not all_sources or any(not usable(s, user_id) for s in all_sources):
                continue
            own = {source_key(s): s for s in all_sources}
            if not assistant_keys < own.keys():
                continue
            extra = own.keys() - assistant_keys
            if len(extra) != 1:
                continue
            key = next(iter(extra))
            parent = own[key]
            if key in known or parent['role'] != 'user':
                continue
            span = parent.get('content_span')
            if span and (span.get('start') != 0 or span.get('end') != span.get('original_length')
                         or span.get('end') != len(parent['content'])):
                continue
            if not all(s['request_id'] == parent['request_id'] and s['message_index'] == parent['message_index']+1
                       for s in sources):
                continue
            # Source-only context: do not inherit a supplier's factual or
            # requirement-support marks, generated summary, or equivalent IDs.
            addition = dict(id=mid, content=answer_context.with_evidence('Conversation excerpt:', [parent]),
                created_at=item.get('created_at'), memory_type='episode', score_kind='conversation_context',
                personal_evidence=False, coverage_ids=[], is_constraint=False, equivalent_ids=[],
                sources=[copy.deepcopy(parent)], source_count=1, packet_hash_version=4)
            addition = schemas.SearchItem.model_validate(addition).model_dump()
            cost = len(addition['content'].encode()) + bool(result)
            reason = ('context_limit' if len(trace['appended_ids']) >= limit else
                      'top_k' if len(result) >= top_k else 'token_budget' if used+cost > byte_budget else 'admitted')
            trace['proposed'].append(dict(anchor_id=anchor['id'], candidate_id=mid,
                parent_source_event_id=parent['source_event_id'], reason=reason, added_bytes=cost))
            if reason != 'admitted':
                continue
            result.append(addition)
            trace['appended_ids'].append(mid)
            trace['added_bytes'] += cost
            known.add(key)
            kept.add(mid)
            used += cost
    if packed != baseline or result[:len(packed)] != baseline:
        raise ValueError('Original packet changed')
    trace.update(total_bytes=used, original_count=len(packed), result_count=len(result))
    return result, trace



def attach(packed, packet_hash, manifest, fine, *, user_id, top_k, enabled, reserve_enabled=False,
           reserve_extended=False,lexical_enabled=False,fill_enabled=False,query='',options=(),
           source_query_enabled=False,source_champions=(),source_nomination=None):
    """Finalize hashes/accounting; accepted content and support labels stay fixed."""
    if evidence_packet.digest(packed) != packet_hash:
        raise ValueError('Baseline packet hash mismatch')
    byte_budget = manifest['token_budget']
    active = enabled or reserve_enabled
    proposer, policy = propose, POLICY
    if reserve_enabled:
        from . import fine_context_reserve
        proposer, policy = fine_context_reserve.propose, fine_context_reserve.POLICY
        if reserve_extended:
            policy = 'fine_original_context_5k_v1'
        if lexical_enabled:
            from . import fine_context_lexical
            proposer,policy=fine_context_lexical.propose,fine_context_lexical.POLICY
            if fill_enabled:
                from . import fine_context_fill
                proposer,policy=fine_context_fill.propose,fine_context_fill.POLICY
    source_active = (reserve_enabled and reserve_extended and lexical_enabled and
                     source_query_enabled and not fill_enabled)
    if source_active:
        from . import source_query_context
        proposer, policy = source_query_context.propose, source_query_context.POLICY
    trace = dict(policy=policy, enabled=active, base_packet_ids=[p['id'] for p in packed],
                 base_packet_hash=packet_hash, base_token_upper_bound=manifest['token_upper_bound'],
                 eligible_candidate_ids=[c['id'] for c in fine] if active else [])
    if active:
        result, details = proposer(packed, fine, user_id=user_id, top_k=top_k, byte_budget=byte_budget,
                                  **({'reserve_bytes':5120} if reserve_enabled and reserve_extended else {}),
                                  **({'query':query,'options':options or ()} if reserve_enabled and lexical_enabled else {}),
                                  **({'champions':source_champions} if source_active else {}))
    else:
        result = packed
        details = dict(proposed=[], appended_ids=[], added_bytes=0, total_bytes=manifest['token_upper_bound'],
                       original_count=len(packed), result_count=len(packed))
    trace.update(details, status='appended' if details['appended_ids'] else 'not_needed' if active else 'disabled')
    trace['policy'] = policy
    if source_active:
        trace['source_nomination'] = source_nomination
    updated = dict(manifest)
    if details['appended_ids']:
        packet_hash = evidence_packet.digest(result)
        result = [dict(item, packet_hash=packet_hash) for item in result]
        updated.update(included_ids=[p['id'] for p in result], token_upper_bound=details['total_bytes'],
                       evidence_count=manifest['evidence_count']+len(details['appended_ids']))
    updated['conversation_context'] = trace
    return result, packet_hash, updated, trace
