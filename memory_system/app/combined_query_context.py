"""Compose a near-cutoff source seat and marginal query context within 5 KiB.

Both rankings use existing source-query evidence. They never establish facts,
read hidden sources, expand the packet budget, or make provider calls.
"""
import copy

from . import config, conversation_context, evidence_packet, marginal_query_context, source_query_context
from .fine_context_reserve import propose as reserve

POLICY='combined_query_context_v1'


def enabled():
    return (config.PACKET_QUERY_CONTEXT_COMBINED and config.PACKET_FINE_CONTEXT_RESERVE
        and config.PACKET_FINE_CONTEXT_5K and config.PACKET_FINE_CONTEXT_LEXICAL
        and not any((config.PACKET_FINE_CONTEXT_FILL,config.PACKET_SOURCE_QUERY_CONTEXT,
            config.PACKET_QUESTION_CONTEXT_CE,config.PACKET_SOURCE_FIRST_RESIDUAL,
            config.PACKET_MARGINAL_QUERY_CONTEXT)))


def attach(packed,packet_hash,manifest,fine,pool,*,user_id,top_k,routes,query_specs,
           query,options,coarse_ids,fine_ids,selected_ids,fine_selection):
    common=dict(routes=routes,query_specs=query_specs,query=query,options=options,
                coarse_ids=coarse_ids,selected_ids=selected_ids)
    champions,nomination=source_query_context.nominate(pool,**common,fine_ids=fine_ids,
        fine_selection=fine_selection)
    prefix,prefix_hash,prefix_manifest,prefix_trace=conversation_context.attach(
        packed,packet_hash,manifest,fine,user_id=user_id,top_k=top_k,enabled=False,
        reserve_enabled=True,reserve_extended=True,lexical_enabled=True,query=query,options=options,
        source_query_enabled=True,source_champions=champions,source_nomination=nomination)
    marginal,marginal_trace=marginal_query_context.nominate(pool,prefix,**common,user_id=user_id)
    result,extra=reserve(prefix,marginal,user_id=user_id,top_k=top_k,
        byte_budget=manifest['token_budget'],reserve_bytes=5120-prefix_trace['added_bytes'],limit=1)
    if result[:len(prefix)]!=prefix:
        raise ValueError('Combined query context displaced its source prefix')
    extra=dict(extra,nomination=marginal_trace,status='appended' if extra['appended_ids'] else 'not_needed')
    context=dict(prefix_trace,policy=POLICY,limit=5,prefix_context=copy.deepcopy(prefix_trace),
        prefix_packet_ids=[p['id'] for p in prefix],prefix_packet_hash=prefix_hash,combined_residual=extra,
        proposed=prefix_trace['proposed']+extra['proposed'],
        appended_ids=prefix_trace['appended_ids']+extra['appended_ids'],
        added_bytes=prefix_trace['added_bytes']+extra['added_bytes'],
        total_bytes=extra['total_bytes'],result_count=len(result))
    context['status']='appended' if context['appended_ids'] else 'not_needed'
    updated=dict(prefix_manifest,conversation_context=context)
    digest=prefix_hash
    if extra['appended_ids']:
        digest=evidence_packet.digest(result)
        result=[dict(p,packet_hash=digest) for p in result]
        updated.update(included_ids=[p['id'] for p in result],token_upper_bound=extra['total_bytes'],
            evidence_count=prefix_manifest['evidence_count']+len(extra['appended_ids']))
    return result,digest,updated,context
