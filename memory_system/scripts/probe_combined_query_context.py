"""Compose two existing source proposals under the same 5 KiB context cap.

No provider calls. Fix choices before loading reference labels. Compare both
individual rules and their composition on every question, not selected wins.
"""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import main as score
from app import evidence_packet, marginal_query_context, search_pipeline, source_query_context
from app.fine_context_lexical import propose as lexical
from app.fine_context_reserve import propose as reserve

MODES=('source_only','marginal_only','combined')


def prepare(t):
    ctx=t['cascade']['conversation_context']
    assert ctx['policy']=='fine_user_lexical_context_v1'
    base=[dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
    assert evidence_packet.digest(base)==ctx['base_packet_hash']
    pool={c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
          if c.get('view_status')!='stale' and '_packet_item' in c}
    selected=t['selection']['selected_ids'];fine_ids=t['cascade']['fine']['selected_ids']
    fine=[pool[mid] for mid in fine_ids if mid in pool and mid not in selected]
    kw=dict(user_id=t['user_id'],top_k=t['top_k'],byte_budget=t['coverage_manifest']['token_budget'])
    prefix,old=lexical(base,fine,**kw,reserve_bytes=5120,query=t['query'],options=t['options'])
    digest=evidence_packet.digest(prefix)
    assert [dict(p,packet_hash=digest) for p in prefix]==t['returned']
    common=dict(routes=t['routes'],query_specs=t['query_specs'],query=t['query'],options=t['options'],
                coarse_ids=t['cascade']['coarse']['selected_ids'],selected_ids=selected)
    champions,source_nomination=source_query_context.nominate(pool,**common,fine_ids=fine_ids,
        fine_selection=t['cascade']['fine']['selection'])
    source,source_trace=source_query_context.propose(base,fine,champions=champions,**kw,
        reserve_bytes=5120,query=t['query'],options=t['options'])
    packets=dict(baseline=prefix,source_only=source);proposals=dict(source_only=source_trace)
    inventory=dict(source_nomination=source_nomination)
    for mode,start,trace in [('marginal_only',prefix,old),('combined',source,source_trace)]:
        nominated,nomination=marginal_query_context.nominate(pool,start,**common,user_id=t['user_id'])
        result,extra=reserve(start,nominated,**kw,reserve_bytes=5120-trace['added_bytes'],limit=1)
        assert result[:len(start)]==start
        assert sum(len(p['content'].encode()) for p in result)+max(0,len(result)-1)<=kw['byte_budget']
        assert trace['added_bytes']+extra['added_bytes']<=5120
        packets[mode]=result;proposals[mode]=extra;inventory[mode]=nomination
    return dict(packets=packets,proposals=proposals,inventory=inventory,remaining_bytes=5120-old['added_bytes'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    with patch('socket.socket.connect',side_effect=AssertionError('Offline only')):
        score(p.parse_args(),preparer=prepare,modes=MODES,limitation=__doc__,script_path=Path(__file__))
