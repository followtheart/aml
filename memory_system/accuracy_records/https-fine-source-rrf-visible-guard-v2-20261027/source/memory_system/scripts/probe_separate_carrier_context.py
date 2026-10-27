"""Budgeted complete adjacent messages supplied by distinct eligible carriers.

Use measured query ranks without reference labels. Only same-request adjacent
user/assistant pairs with no shared eligible AMU qualify. Both full messages
must already be visible in the original packet or actual scored coarse units.
Append at most one literal two-message context item in the original 5 KiB
remainder, preserving the packet and three lexical seats. No fact/support labels
are inherited from either carrier and no raw source text is rehydrated.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import main as score
from probe_semantic_source_context import ranked_sources
from app import answer_context, evidence_packet, fine_evidence, schemas, search_pipeline
from app.conversation_context import usable
from app.fine_context_lexical import propose as lexical

MODES = ('ranked_top8', 'retained_top40')


def full_key(s):
    if not isinstance(s.get('content'), str) or not s['content'].strip():
        return None
    span = s.get('content_span')
    if span and not (span.get('start') == 0 and span.get('end') == span.get('original_length') == len(s['content'])):
        return None
    return tuple(s.get(k) for k in ('request_id','message_index','source_event_id','role','content'))


def preparer(plan, rankings):
    measured = {(r['user_id'],tuple(q['query']['text'] for q in r['modes']['exchange'])):r for r in rankings['rankings']}
    if len(measured) != 56:
        raise ValueError('Require unique whole-cohort query signatures')
    pairs = {int(si):members for si,members in plan['exchange_sources'].items()
             if len(members)==2 and not set(members[0]['amu_ids']) & set(members[1]['amu_ids'])}
    for members in pairs.values():
        u,a = members
        if (u['role']!='user' or a['role']!='assistant' or u['user_id']!=a['user_id']
                or u['session_id']!=a['session_id'] or u['request_id']!=a['request_id']
                or u['message_index']+1!=a['message_index']):
            raise ValueError('Invalid adjacent pair')

    def prepare(t):
        measured_row = measured[t['user_id'],tuple(q['text'] for q in t['query_specs'])]
        ctx = t['cascade']['conversation_context']
        if ctx['policy']!='fine_user_lexical_context_v1':
            raise ValueError('Require original lexical context')
        base = [dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
        if evidence_packet.digest(base)!=ctx['base_packet_hash']:
            raise ValueError('Original base hash differs')
        pool = {c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
                if c.get('view_status')!='stale' and '_packet_item' in c}
        fine = [pool[mid] for mid in t['cascade']['fine']['selected_ids']
                if mid in pool and mid not in t['selection']['selected_ids']]
        prefix,old = lexical(base,fine,user_id=t['user_id'],top_k=t['top_k'],
            byte_budget=t['coverage_manifest']['token_budget'],reserve_bytes=5120,query=t['query'],options=t['options'])
        if [dict(p,packet_hash=evidence_packet.digest(prefix)) for p in prefix]!=t['returned']:
            raise ValueError('Original full packet differs')
        remaining = 5120-old['added_bytes']
        kept = {p['id'] for p in prefix}
        known = {key:s for p in prefix for s in p['sources'] if usable(s,t['user_id']) and (key:=full_key(s))}
        carriers = {}
        for mid in t['cascade']['coarse']['selected_ids']:
            if mid not in pool:
                continue
            c = pool[mid]
            ce = c.get('_ce_score')
            if type(ce) not in (int,float) or not math.isfinite(ce) or c['_packet_item'].get('is_constraint'):
                continue
            group,missing = fine_evidence.closure(c,pool)
            if missing or not group<=kept|{mid}:
                continue
            for s in c['_packet_item']['sources']:
                key = full_key(s)
                if key and usable(s,t['user_id']) and s['content'] in c['_packet_item']['content']:
                    carriers.setdefault(key,[]).append((ce,mid,s))
        source_order = ranked_sources(measured_row['modes']['exchange'])
        rank = {si:i+1 for i,si in enumerate(source_order)}
        details = []
        for si,members in pairs.items():
            if members[0]['user_id']!=t['user_id']:
                continue
            keys = [full_key(s) for s in members]
            sources, suppliers, ce_scores = [],[],[]
            for key in keys:
                if key in known:
                    sources.append(copy.deepcopy(known[key]))
                    suppliers.append('original_packet')
                elif key in carriers:
                    ce,mid,s = sorted(carriers[key],key=lambda x:(-x[0],x[1]))[0]
                    sources.append(copy.deepcopy(s))
                    suppliers.append(mid)
                    ce_scores.append(ce)
                else:
                    suppliers.append(None)
            visible = len(sources)==2
            item = None
            if visible:
                identity = hashlib.sha256(json.dumps(keys,ensure_ascii=False).encode()).hexdigest()[:24]
                item = schemas.SearchItem.model_validate(dict(id='context_pair_'+identity,
                    content=answer_context.with_evidence('Conversation excerpt:',sources),
                    memory_type='episode',score_kind='adjacent_source_context',personal_evidence=False,
                    coverage_ids=[],is_constraint=False,equivalent_ids=[],sources=sources,
                    source_count=2,packet_hash_version=4)).model_dump()
            details.append(dict(source_index=si,source_rank=rank.get(si),suppliers=suppliers,
                retained_members=sum(k in known for k in keys),full_visible_members=len(sources),
                eligible=visible and any(k not in known for k in keys),item=item,
                supplier_ce_scores=ce_scores,added_bytes=len(item['content'].encode())+bool(prefix) if item else None))
        packets, proposals = {'baseline':prefix},{}
        used = sum(len(p['content'].encode()) for p in prefix)+max(0,len(prefix)-1)
        for mode in MODES:
            ordered = sorted([d for d in details if d['source_rank'] is not None
                and d['source_rank']<=(8 if mode=='ranked_top8' else 40)
                and (mode=='ranked_top8' or d['retained_members']>0)],key=lambda d:d['source_rank'])
            packet = list(prefix)
            proposal = dict(proposed=[],appended_ids=[],added_bytes=0)
            for d in ordered:
                reason = ('incomplete_visible_pair' if not d['eligible'] else
                    'context_limit' if proposal['appended_ids'] else
                    'top_k' if len(packet)>=t['top_k'] else
                    'reserve_bytes' if d['added_bytes']>remaining else
                    'token_budget' if used+d['added_bytes']>t['coverage_manifest']['token_budget'] else 'admitted')
                proposal['proposed'].append(dict(source_index=d['source_index'],suppliers=d['suppliers'],
                    reason=reason,added_bytes=d['added_bytes']))
                if reason=='admitted':
                    packet.append(d['item'])
                    proposal['appended_ids'].append(d['item']['id'])
                    proposal['added_bytes']=d['added_bytes']
            if packet[:len(prefix)]!=prefix:
                raise ValueError('Original prefix changed')
            packets[mode],proposals[mode]=packet,proposal
        return dict(packets=packets,proposals=proposals,remaining_bytes=remaining,
            inventory=[{k:v for k,v in d.items() if k!='item'} for d in details])
    return prepare


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'))
    args=p.parse_args()
    plan=json.loads((args.probe/'plan.json').read_text())
    ranks=json.loads((args.probe/'rankings.json').read_text())
    if (plan.get('pairing_policy')!='eligible_adjacent' or
            hashlib.sha256((args.probe/'plan.json').read_bytes()).hexdigest()!=ranks['plan_sha256']):
        raise ValueError('Require explicit verified adjacent measurement')
    args.runs=[args.run/'control']
    with patch('socket.socket.connect',side_effect=AssertionError('Offline proposal')):
        score(args,preparer=preparer(plan,ranks),modes=MODES,limitation=__doc__,script_path=Path(__file__))
    result=json.loads(args.output.read_text())
    for s in result['summary']:
        mode=s['mode']
        s['complete_sets']={m:sum(bool(r['coverage'][m]['user']) and all(r['coverage'][m]['user'])
            for r in result['rows']) for m in ('baseline',mode)}
    result['measured_plan_sha256']=ranks['plan_sha256']
    result['measured_rankings_sha256']=hashlib.sha256((args.probe/'rankings.json').read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
