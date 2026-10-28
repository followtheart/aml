"""Offline question-source residual ordering, including unselected fine sources.

Keep the original lexical prefix and 5 KiB limit; add at most one whole-source
item. Compare literal rank-one with the first deliverable original-question
hit, both outside fine and across the scored coarse pool. Reference labels
are loaded only after all proposals are fixed. No new provider/source reads.
Treatment inputs reconstruct and verify their original lexical prefixes.
"""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import main as score
from app import evidence_packet,search_pipeline
from app.fine_context_lexical import propose as lexical
from app.fine_context_reserve import propose as reserve

MODES=('coarse_first','coarse_best','all_first','all_best')


def prepare(t):
    ctx=t['cascade']['conversation_context']
    assert evidence_packet.digest(t['returned'])==t['packet_hash']
    base=[dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
    assert evidence_packet.digest(base)==ctx['base_packet_hash']
    pool={c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
        if c.get('view_status')!='stale' and '_packet_item' in c}
    fine_ids=set(t['cascade']['fine']['selected_ids'])
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
    initial=[s for s in t['query_specs'] if s['origin']!='grounded_followup']
    indices={str(i) for i,s in enumerate(initial) if s['origin']=='question' and s['text']==t['query']}
    if not indices:raise ValueError('Original question absent')
    routes=[r for r in t['routes'] if r['channel']=='source_text' and r['round']==1]
    if len(routes)!=1:raise ValueError('Require one initial source route')
    allowed=set(t['cascade']['coarse']['selected_ids'])-selected
    inventory=[]
    for position,hit in enumerate(routes[0]['candidates']):
        mid=hit['id']
        if mid not in pool or mid not in allowed:continue
        ranks=[rank for key,rank in hit.get('_query_ranks',{}).items() if key in indices]
        if not ranks:continue
        if any(type(rank) is not int or rank<1 for rank in ranks):raise ValueError('Invalid source query rank')
        _,proposal=reserve(prefix,[pool[mid]],**kw,reserve_bytes=remaining,limit=1)
        inventory.append(dict(candidate_id=mid,in_fine=mid in fine_ids,question_rank=min(ranks),
            source_route_rank=position+1,ce_score=pool[mid].get('_ce_score'),
            deliverable=bool(proposal['appended_ids']),proposal=proposal))
    packets={'baseline':prefix};proposals={}
    for mode in MODES:
        hits=[r for r in inventory if r['deliverable'] and
            (not mode.startswith('coarse') or not r['in_fine']) and
            (not mode.endswith('first') or r['question_rank']==1)]
        hits.sort(key=lambda r:(r['question_rank'],r['source_route_rank']))
        packets[mode],proposals[mode]=reserve(prefix,[pool[r['candidate_id']] for r in hits],
            **kw,reserve_bytes=remaining,limit=1)
        assert packets[mode][:len(prefix)]==prefix
    if ctx['policy']=='source_first_residual_v1':
        actual=packets['coarse_first'];actual_digest=evidence_packet.digest(actual)
        assert [dict(p,packet_hash=actual_digest) for p in actual]==t['returned']
    return dict(packets=packets,proposals=proposals,inventory=inventory,remaining_bytes=remaining)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    with patch('socket.socket.connect',side_effect=AssertionError('Offline only')):
        score(p.parse_args(),preparer=prepare,modes=MODES,limitation=__doc__,script_path=Path(__file__))
