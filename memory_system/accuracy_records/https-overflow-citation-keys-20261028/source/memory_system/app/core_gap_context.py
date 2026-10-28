"""One complete source unit from the CE core that did not reach fine.

Keep the accepted packet and original 5 KiB context prefix. The optional extra
seat has its own 4 KiB ceiling within the existing request byte/item limits.
This is unverified retrieval context, never a useful-review or fact override.
"""
import copy
import math

from . import config, evidence_packet, fine_evidence
from .fine_context_lexical import POLICY as PREFIX_POLICY
from .fine_context_reserve import propose as reserve

POLICY = 'ce_core_gap_context_v1'
EXTRA_BYTES = 4096
INCOMPATIBLE = ('PACKET_FINE_CONTEXT_FILL', 'PACKET_SOURCE_QUERY_CONTEXT',
    'PACKET_QUESTION_CONTEXT_CE', 'PACKET_SOURCE_FIRST_RESIDUAL',
    'PACKET_MARGINAL_QUERY_CONTEXT', 'PACKET_QUERY_CONTEXT_COMBINED')


def enabled():
    return (config.PACKET_CORE_GAP_CONTEXT and config.PACKET_FINE_CONTEXT_RESERVE
            and config.PACKET_FINE_CONTEXT_5K and config.PACKET_FINE_CONTEXT_LEXICAL
            and not any(getattr(config, key) for key in INCOMPATIBLE))


def nominate(pool, packed, *, fine_ids, coarse_ids, selection, count):
    kept = {p['id'] for p in packed}
    excluded = set(fine_ids) | kept
    allowed = set(coarse_ids) & set(pool)
    ranks = {d['id']: d.get('rank') for d in selection.get('decisions', [])}
    eligible, inventory = [], []
    for mid, rank in sorted(ranks.items(), key=lambda p: p[1] if type(p[1]) is int else math.inf):
        if mid in excluded or mid not in allowed or type(rank) is not int or not 1 <= rank <= count:
            continue
        candidate = pool[mid]
        value = candidate.get('_ce_score')
        group, missing = fine_evidence.closure(candidate, pool)
        reason = ('unscored' if type(value) not in (int, float) or not math.isfinite(value)
                  else 'missing_dependency' if missing or not group <= kept | {mid}
                  else 'nominated')
        inventory.append(dict(candidate_id=mid, core_rank=rank, ce_score=value,
            dependency_ids=sorted(group - {mid}), missing_ids=sorted(missing), reason=reason))
        if reason == 'nominated':
            eligible.append(candidate)
    return eligible, inventory



def attach(packed, packet_hash, manifest, candidates, nomination, *, user_id, top_k):
    if evidence_packet.digest(packed) != packet_hash:
        raise ValueError('Core gap prefix hash mismatch')
    prefix = manifest['conversation_context']
    if prefix['policy'] != PREFIX_POLICY or prefix['reserve_bytes'] != 5120:
        raise ValueError('Core gap requires the original 5 KiB lexical prefix')
    result, extra = reserve(packed, candidates, user_id=user_id, top_k=top_k,
        byte_budget=manifest['token_budget'], reserve_bytes=EXTRA_BYTES, limit=1)
    if result[:len(packed)] != packed:
        raise ValueError('Core gap changed existing evidence')
    extra = dict(extra, nomination=copy.deepcopy(nomination),
        status='appended' if extra['appended_ids'] else 'not_needed')
    context = dict(prefix, policy=POLICY, limit=4, reserve_bytes=5120+EXTRA_BYTES,
        prefix_context=copy.deepcopy(prefix), prefix_packet_hash=packet_hash,
        prefix_packet_ids=[p['id'] for p in packed], core_gap=extra,
        proposed=prefix['proposed']+extra['proposed'],
        appended_ids=prefix['appended_ids']+extra['appended_ids'],
        added_bytes=prefix['added_bytes']+extra['added_bytes'],
        total_bytes=extra['total_bytes'], result_count=len(result))
    context['status'] = 'appended' if context['appended_ids'] else 'not_needed'
    updated = dict(manifest, conversation_context=context)
    if extra['appended_ids']:
        packet_hash = evidence_packet.digest(result)
        result = [dict(p, packet_hash=packet_hash) for p in result]
        updated.update(included_ids=[p['id'] for p in result], token_upper_bound=extra['total_bytes'],
            evidence_count=manifest['evidence_count']+len(extra['appended_ids']))
    return result, packet_hash, updated
