"""Offline marginal source-query coverage within the original residual budget.

Rank scored coarse sources by improvement over already delivered query hits.
Source identity includes exact visible content/spans; ranking never establishes
personal facts. Original three context seats remain intact, at most one whole
source item may use the remaining 5 KiB allowance. All proposals precede gold.
"""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import main as score
from app import evidence_packet,retrieval_queries,search_pipeline
from app.conversation_context import source_key,usable
from app.fine_context_lexical import propose as lexical
from app.fine_context_reserve import propose as reserve

MODES=('literal_rank','literal_marginal','initial_rank','initial_marginal',
       'literal_first_marginal','initial_first_marginal')


def prepare(t):
    ctx=t['cascade']['conversation_context']
    assert evidence_packet.digest(t['returned'])==t['packet_hash']
    base=[dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
    assert evidence_packet.digest(base)==ctx['base_packet_hash']
    pool={c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
        if c.get('view_status')!='stale' and '_packet_item' in c}
    selected=set(t['selection']['selected_ids'])
    fine=[pool[mid] for mid in t['cascade']['fine']['selected_ids'] if mid in pool and mid not in selected]
    kw=dict(user_id=t['user_id'],top_k=t['top_k'],byte_budget=t['coverage_manifest']['token_budget'])
    prefix,old=lexical(base,fine,**kw,reserve_bytes=5120,query=t['query'],options=t['options'])
    digest=evidence_packet.digest(prefix)
    if ctx['policy']=='source_first_residual_v1':
        assert digest==ctx['prefix_packet_hash']
        assert [p['id'] for p in prefix]==ctx['prefix_packet_ids']
        assert [dict(p,packet_hash=digest) for p in prefix]==[
            dict(p,packet_hash=digest) for p in t['returned'][:len(prefix)]]
    else:
        assert ctx['policy']=='fine_user_lexical_context_v1'
        assert [dict(p,packet_hash=digest) for p in prefix]==t['returned']
    remaining=5120-old['added_bytes']
    specs=[s for s in t['query_specs'] if s['origin']!='grounded_followup']
    groups={};literal=set()
    for i,s in enumerate(specs):
        key=str(i);oi=s.get('option_index')
        groups[key]='option:'+str(oi) if type(oi) is int else key
        if s['origin']=='question' and s['text']==t['query']:literal.add(key)
        if s['origin']=='option_topic':
            if type(oi) is not int or not 0<=oi<len(t['options']) or s['text']!=retrieval_queries.topic_anchor(t['options'][oi]):
                raise ValueError('Not an exact literal option topic')
            literal.add(key)
    routes=[r for r in t['routes'] if r['channel']=='source_text' and r['round']==1]
    if len(routes)!=1:raise ValueError('Require one original source route')
    known={source_key(s) for p in prefix for s in p.get('sources',[]) if usable(s,t['user_id'])}
    included={p['id'] for p in prefix}
    covered={key:0.0 for key in groups}
    hits=routes[0]['candidates']
    for hit in hits:
        ranks=hit.get('_query_ranks',{})
        if any(key not in groups or type(rank) is not int or rank<1 for key,rank in ranks.items()):
            raise ValueError('Invalid query rank')
        item=pool.get(hit['id'],{}).get('_packet_item',{})
        sources=[s for s in item.get('sources',[]) if s.get('content')]
        represented=(hit['id'] in included or (sources and all(source_key(s) in known for s in sources)))
        if represented:
            for key,rank in ranks.items():covered[key]=max(covered[key],1/rank)
    allowed=set(t['cascade']['coarse']['selected_ids'])-selected
    inventory=[]
    for position,hit in enumerate(hits):
        mid=hit['id']
        if mid not in pool or mid not in allowed:continue
        _,proposal=reserve(prefix,[pool[mid]],**kw,reserve_bytes=remaining,limit=1)
        if not proposal['appended_ids']:continue
        ranks=hit.get('_query_ranks',{});scores={};gains={}
        for mode in MODES:
            grouped={};detail={}
            for key,rank in ranks.items():
                if mode.startswith('literal') and key not in literal:continue
                if '_first_' in mode and rank!=1:continue
                value=max(0,1/rank-covered[key]) if mode.endswith('marginal') else 1/rank
                group=groups[key]
                grouped[group]=max(grouped.get(group,0),value)
                if value:detail[key]=dict(rank=rank,covered=covered[key],gain=value)
            scores[mode]=sum(grouped.values());gains[mode]=detail
        inventory.append(dict(candidate_id=mid,source_route_rank=position+1,ce_score=pool[mid].get('_ce_score'),
            query_ranks=ranks,scores=scores,gains=gains,cost=proposal['added_bytes']))
    packets={'baseline':prefix};proposals={}
    for mode in MODES:
        hits=sorted([r for r in inventory if r['scores'][mode]>0],key=lambda r:(-r['scores'][mode],r['source_route_rank']))
        packets[mode],proposals[mode]=reserve(prefix,[pool[r['candidate_id']] for r in hits],
            **kw,reserve_bytes=remaining,limit=1)
        assert packets[mode][:len(prefix)]==prefix
    return dict(packets=packets,proposals=proposals,inventory=inventory,remaining_bytes=remaining)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    with patch('socket.socket.connect',side_effect=AssertionError('Offline only')):
        score(p.parse_args(),preparer=prepare,modes=MODES,limitation=__doc__,script_path=Path(__file__))
