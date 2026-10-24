"""One residual source ranked by literal-query improvement over the prefix.

Query ranks are retrieval signals only. Keep every original lexical seat and
source/byte constraint; no new provider call or hidden source read is needed.
"""
from . import budget,retrieval_queries,source_first_residual
from .conversation_context import source_key,usable

POLICY='marginal_literal_query_context_v1'


def nominate(pool,packed,*,routes,query_specs,query,options,coarse_ids,selected_ids,user_id):
    specs=[s for s in query_specs if s.get('origin')!='grounded_followup']
    direct={};groups={}
    for index,spec in enumerate(specs):
        key=str(index);oi=spec.get('option_index')
        question=spec.get('origin')=='question' and spec.get('text')==query
        topic=(spec.get('origin')=='option_topic' and type(oi) is int and 0<=oi<len(options)
               and spec.get('text')==retrieval_queries.topic_anchor(options[oi]))
        if question or topic:
            direct[key]=dict(spec)
            groups[key]='option:'+str(oi) if type(oi) is int else key
    initial=[r for r in routes if r.get('channel')=='source_text' and r.get('round')==1]
    trace=dict(direct_queries=direct,covered={},nominations=[],status='evaluated')
    if len(initial)!=1 or not direct:
        trace['status']='missing_literal_query_or_source_route'
        return [],trace
    known={source_key(s) for p in packed for s in p.get('sources',[]) if usable(s,user_id)}
    included={p['id'] for p in packed}
    covered={key:0.0 for key in direct}
    hits=initial[0].get('candidates',[])
    for hit in hits:
        budget.check()
        item=pool.get(hit['id'],{}).get('_packet_item',{})
        sources=[s for s in item.get('sources',[]) if s.get('content')]
        represented=hit['id'] in included or (sources and all(source_key(s) in known for s in sources))
        if represented:
            for key,rank in hit.get('_query_ranks',{}).items():
                if key in covered and type(rank) is int and rank>0:
                    covered[key]=max(covered[key],1/rank)
    allowed=set(coarse_ids)-set(selected_ids)
    for position,hit in enumerate(hits):
        budget.check()
        mid=hit['id']
        if mid not in allowed or mid not in pool:continue
        grouped={};gains={}
        for key,rank in hit.get('_query_ranks',{}).items():
            if key not in direct or type(rank) is not int or rank<1:continue
            value=max(0,1/rank-covered[key]);group=groups[key]
            grouped[group]=max(grouped.get(group,0),value)
            if value:gains[key]=dict(rank=rank,covered=covered[key],gain=value)
        value=sum(grouped.values())
        if value:
            trace['nominations'].append(dict(candidate_id=mid,score=value,
                source_route_rank=position+1,gains=gains))
    trace['covered']=covered
    trace['nominations'].sort(key=lambda r:(-r['score'],r['source_route_rank']))
    return [pool[r['candidate_id']] for r in trace['nominations']],trace


def attach(packed,packet_hash,manifest,champions,nomination,*,user_id,top_k):
    result,digest,updated=source_first_residual.attach(packed,packet_hash,manifest,champions,nomination,
        user_id=user_id,top_k=top_k)
    context=dict(updated['conversation_context'])
    context['marginal_query']=context.pop('source_first')
    context['policy']=POLICY
    return result,digest,dict(updated,conversation_context=context)
