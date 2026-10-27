"""One residual complete-source first hit of the exact original question.

Keep every original lexical seat. Only an already scored coarse outsider may
add context within the same 5 KiB allowance; retrieval is not fact validation.
"""
import copy

from . import budget,evidence_packet
from .fine_context_lexical import POLICY as PREFIX_POLICY
from .fine_context_reserve import propose as reserve

POLICY='source_first_residual_v1'


def nominate(pool,*,routes,query_specs,query,coarse_ids,fine_ids,selected_ids):
    initial=[r for r in routes if r.get('channel')=='source_text' and r.get('round')==1]
    specs=[s for s in query_specs if s.get('origin')!='grounded_followup']
    indices=[str(i) for i,s in enumerate(specs) if s.get('origin')=='question' and s.get('text')==query]
    trace=dict(query_indices=indices,nominations=[],status='evaluated')
    if len(initial)!=1 or not indices:
        trace['status']='missing_original_question_or_source_route'
        return [],trace
    allowed=set(coarse_ids)-set(fine_ids)-set(selected_ids)
    seen=set()
    for position,candidate in enumerate(initial[0].get('candidates',[])):
        budget.check()
        mid=candidate['id']
        if mid not in allowed or mid not in pool or mid in seen:continue
        seen.add(mid)
        ranks=candidate.get('_query_ranks',{})
        hits=[key for key in indices if type(ranks.get(key)) is int and ranks[key]==1]
        if hits:trace['nominations'].append(dict(candidate_id=mid,query_indices=hits,source_route_rank=position+1))
    trace['nominations'].sort(key=lambda r:(-len(r['query_indices']),r['source_route_rank']))
    return [pool[r['candidate_id']] for r in trace['nominations']],trace


def attach(packed,packet_hash,manifest,champions,nomination,*,user_id,top_k):
    if evidence_packet.digest(packed)!=packet_hash:raise ValueError('Residual source prefix hash mismatch')
    prefix=manifest['conversation_context']
    if prefix['policy']!=PREFIX_POLICY or prefix['reserve_bytes']!=5120:
        raise ValueError('Residual source first hit requires the unchanged 5 KiB lexical prefix')
    result,extra=reserve(packed,champions,user_id=user_id,top_k=top_k,
        byte_budget=manifest['token_budget'],reserve_bytes=5120-prefix['added_bytes'],limit=1)
    if result[:len(packed)]!=packed:raise ValueError('Residual source changed original packet')
    extra=dict(extra,nomination=copy.deepcopy(nomination),status='appended' if extra['appended_ids'] else 'not_needed')
    context=dict(prefix,policy=POLICY,limit=4,prefix_context=copy.deepcopy(prefix),
        prefix_packet_ids=[p['id'] for p in packed],prefix_packet_hash=packet_hash,source_first=extra,
        proposed=prefix['proposed']+extra['proposed'],appended_ids=prefix['appended_ids']+extra['appended_ids'],
        added_bytes=prefix['added_bytes']+extra['added_bytes'],total_bytes=extra['total_bytes'],result_count=len(result))
    context['status']='appended' if context['appended_ids'] else 'not_needed'
    updated=dict(manifest,conversation_context=context)
    if extra['appended_ids']:
        packet_hash=evidence_packet.digest(result)
        result=[dict(p,packet_hash=packet_hash) for p in result]
        updated.update(included_ids=[p['id'] for p in result],token_upper_bound=extra['total_bytes'],
            evidence_count=manifest['evidence_count']+len(extra['appended_ids']))
    return result,packet_hash,updated
