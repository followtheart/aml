"""Offline residual complete-source units connected to already delivered quotes.

An exact shared source and original message indices establish only conversation
structure, never user preference or factual support. No source reads/expansion,
provider calls, discarded prefix or larger byte allowance. Gold is scored last.
"""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import main,prepare as source_prepare
from app import search_pipeline
from app.conversation_context import source_key,usable
from app.fine_context_reserve import propose as reserve,full_user

MODES=('fine_exact_adjacent','coarse_exact_adjacent','coarse_shared_source',
       'fine_identity_adjacent','coarse_identity_adjacent')


def observation(source):
    return tuple(source.get(k) for k in ('user_id','request_id','message_index','source_event_id','role'))


def original_length(source):
    return (source.get('content_span') or {}).get('original_length',len(source['content']))


def prepare(t):
    original=source_prepare(t)
    prefix=original['packets']['baseline'];remaining=original['remaining_bytes']
    known={source_key(s) for p in prefix for s in p['sources'] if usable(s,t['user_id'])}
    represented={(observation(s),original_length(s)) for p in prefix for s in p['sources'] if usable(s,t['user_id'])}
    pool={c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
        if c.get('view_status')!='stale' and '_packet_item' in c}
    fine=set(t['cascade']['fine']['selected_ids']);selected=set(t['selection']['selected_ids'])
    kwargs=dict(user_id=t['user_id'],top_k=t['top_k'],byte_budget=t['coverage_manifest']['token_budget'])
    candidates={};inventory=[]
    for mid in t['cascade']['coarse']['selected_ids']:
        if mid not in pool or mid in selected:continue
        c=pool[mid]
        _,trial=reserve(prefix,[c],**kwargs,reserve_bytes=remaining,limit=1)
        if not trial['appended_ids']:continue
        sources=[s for s in c['_packet_item']['sources'] if s.get('content')]
        new=[s for s in sources if full_user(s) and source_key(s) not in known]
        shared=[s for s in sources if source_key(s) in known]
        same_observations=[s for s in sources if (observation(s),original_length(s)) in represented]
        links={mode:[] for mode in MODES}
        for s in new:
            for anchor in shared:
                if s['request_id']!=anchor['request_id']:continue
                link=dict(new_user_event=s['source_event_id'],new_user_index=s['message_index'],
                    shared_event=anchor['source_event_id'],shared_role=anchor['role'],shared_index=anchor['message_index'])
                links['coarse_shared_source'].append(link)
                if anchor['role']=='assistant' and abs(s['message_index']-anchor['message_index'])==1:
                    links['coarse_exact_adjacent'].append(link)
                    if mid in fine:links['fine_exact_adjacent'].append(link)
            for anchor in same_observations:
                if (anchor['request_id']==s['request_id'] and anchor['role']=='assistant'
                        and abs(s['message_index']-anchor['message_index'])==1):
                    link=dict(new_user_event=s['source_event_id'],new_user_index=s['message_index'],
                        shared_event=anchor['source_event_id'],shared_role=anchor['role'],shared_index=anchor['message_index'])
                    links['coarse_identity_adjacent'].append(link)
                    if mid in fine:links['fine_identity_adjacent'].append(link)
        links={mode:list({tuple(sorted(link.items())):link for link in values}.values()) for mode,values in links.items()}
        if not any(links.values()):continue
        candidates[mid]=c
        inventory.append(dict(candidate_id=mid,links=links,ce_score=c['_ce_score'],cost=trial['added_bytes']))
    packets={'baseline':prefix};proposals={}
    for mode in MODES:
        ordered=sorted([r for r in inventory if r['links'][mode]],key=lambda r:(-len(r['links'][mode]),-r['ce_score']))
        packets[mode],proposals[mode]=reserve(prefix,[candidates[r['candidate_id']] for r in ordered],
            **kwargs,reserve_bytes=remaining,limit=1)
        assert packets[mode][:len(prefix)]==prefix
    return dict(packets=packets,proposals=proposals,inventory=inventory,remaining_bytes=remaining)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    with patch('socket.socket.connect',side_effect=AssertionError('Offline diagnostic')):
        main(p.parse_args(),preparer=prepare,modes=MODES,limitation=__doc__,script_path=Path(__file__))
