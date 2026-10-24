"""One existing whole-source carrier for an unseen rare retrieval term.

Preserve the original packet and three lexical context seats. The best scored
carrier orders each exact user source; smaller existing complete carriers of
that same source are tried first. No source clipping, new facts or model calls.
"""
import copy
import hashlib

from . import budget, config, evidence_packet, fine_evidence, fine_sparse_reserve
from .conversation_context import source_key, usable
from .fine_context_lexical import POLICY as PREFIX_POLICY
from .fine_context_reserve import full_user, propose as reserve

POLICY = 'sparse_source_representative_context_v1'


def enabled():
    return (config.PACKET_SPARSE_SOURCE_CONTEXT and config.PACKET_FINE_CONTEXT_RESERVE
            and config.PACKET_FINE_CONTEXT_5K and config.PACKET_FINE_CONTEXT_LEXICAL
            and not any((config.PACKET_FINE_CONTEXT_FILL, config.PACKET_SOURCE_QUERY_CONTEXT,
                config.PACKET_QUESTION_CONTEXT_CE, config.PACKET_SOURCE_FIRST_RESIDUAL,
                config.PACKET_MARGINAL_QUERY_CONTEXT, config.PACKET_QUERY_CONTEXT_COMBINED,
                config.PACKET_CORE_GAP_CONTEXT)))


def nominate(pool, packed, *, coarse_ids, fine_ids, requirements, user_id, top_k,
             byte_budget, remaining_bytes):
    coarse = [pool[mid] for mid in coarse_ids if mid in pool]
    kept = [dict(id=p['id'], _rank_text=p['content'], _packet_item=p) for p in packed]
    visible = {c['id']: fine_evidence.selection.fragments(c) for c in coarse}
    visible.update({c['id']: fine_evidence.selection.fragments(c) for c in kept})
    nominations = fine_sparse_reserve.nominate(coarse, kept, requirements, visible)
    kept_ids = {p['id'] for p in packed}
    excluded = kept_ids | set(fine_ids)
    eligible, inventory = [], []
    for c in sorted(coarse, key=fine_evidence.key, reverse=True):
        budget.check()
        if c['id'] in excluded or c['id'] not in nominations:
            continue
        group, missing = fine_evidence.closure(c, pool)
        reason = 'missing_dependency' if missing or not group <= kept_ids | {c['id']} else 'nominated'
        inventory.append(dict(candidate_id=c['id'], **nominations[c['id']], reason=reason))
        if reason == 'nominated':
            eligible.append(c)
    identities, costs = {}, {}
    for c in eligible:
        budget.check()
        _, trial = reserve(packed, [c], user_id=user_id, top_k=top_k, byte_budget=byte_budget,
                           reserve_bytes=remaining_bytes, limit=1)
        if not trial['proposed']:
            continue
        costs[c['id']] = trial['proposed'][0]['added_bytes']
        hits = set(nominations[c['id']]['terms'])
        identities[c['id']] = {source_key(s) for s in c['_packet_item']['sources']
            if usable(s, user_id) and full_user(s)
            and hits & fine_evidence.selection._topic_concepts(s['content'])}
    source_order, seen_sources = [], set()
    for c in eligible:
        for key in sorted(identities.get(c['id'], ())):
            if key not in seen_sources:
                source_order.append(key)
                seen_sources.add(key)
    representatives, seen_ids = [], set()
    for key in source_order:
        suppliers = sorted([c for c in eligible if key in identities.get(c['id'], ())],
            key=lambda c: (costs[c['id']], -c['_ce_score'], c['id']))
        for c in suppliers:
            if c['id'] not in seen_ids:
                representatives.append(c)
                seen_ids.add(c['id'])
    digest = lambda key: hashlib.sha256(key.encode()).hexdigest()
    for row in inventory:
        row.update(source_hashes=sorted(digest(k) for k in identities.get(row['candidate_id'], ())),
                   cost=costs.get(row['candidate_id']))
    return representatives, dict(nominations=inventory, source_order=[digest(k) for k in source_order],
                                  ordered_ids=[c['id'] for c in representatives])


def attach(packed, packet_hash, manifest, champions, nomination, *, user_id, top_k):
    if evidence_packet.digest(packed) != packet_hash:
        raise ValueError('Sparse source prefix hash mismatch')
    prefix = manifest['conversation_context']
    if prefix['policy'] != PREFIX_POLICY or prefix['reserve_bytes'] != 5120:
        raise ValueError('Sparse source context requires original 5 KiB lexical prefix')
    result, extra = reserve(packed, champions, user_id=user_id, top_k=top_k,
        byte_budget=manifest['token_budget'], reserve_bytes=5120-prefix['added_bytes'], limit=1)
    if result[:len(packed)] != packed:
        raise ValueError('Sparse source changed original packet')
    extra = dict(extra, nomination=copy.deepcopy(nomination),
                 status='appended' if extra['appended_ids'] else 'not_needed')
    context = dict(prefix, policy=POLICY, limit=4, prefix_context=copy.deepcopy(prefix),
        prefix_packet_ids=[p['id'] for p in packed], prefix_packet_hash=packet_hash, sparse_source=extra,
        proposed=prefix['proposed']+extra['proposed'], appended_ids=prefix['appended_ids']+extra['appended_ids'],
        added_bytes=prefix['added_bytes']+extra['added_bytes'], total_bytes=extra['total_bytes'], result_count=len(result))
    context['status'] = 'appended' if context['appended_ids'] else 'not_needed'
    updated = dict(manifest, conversation_context=context)
    if extra['appended_ids']:
        packet_hash = evidence_packet.digest(result)
        result = [dict(p, packet_hash=packet_hash) for p in result]
        updated.update(included_ids=[p['id'] for p in result], token_upper_bound=extra['total_bytes'],
            evidence_count=manifest['evidence_count']+len(extra['appended_ids']))
    return result, packet_hash, updated
