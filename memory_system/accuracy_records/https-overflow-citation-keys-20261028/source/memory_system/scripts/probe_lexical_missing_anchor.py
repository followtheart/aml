"""Preserve the original packet and append one missing literal-query source.

Use independently reconstructed eligible BM25 hits, deduplicate source identity
within each query, then top6/query RRF60. Rank full visible coarse carriers by
the actual matched user message rather than a memory's unrelated linked text.
Compare novelty/depth/capacity without changing queries, CE scores or sources.
No embeddings, generated content, database rehydration or provider calls here.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import main as score
from app import evidence_packet,fine_evidence,persona_source,search_pipeline
from app.conversation_context import usable
from app.fine_context_lexical import propose as lexical
from app.fine_context_reserve import propose as reserve

VARIANTS={'all8_remainder':(8,False,False),'missing8_remainder':(8,True,False),
          'missing40_remainder':(40,True,False),'missing40_extra4k':(40,True,True)}


def identity(s):
    return s['request_id'],s['message_index'],s['content']


def anchor_ranks(row):
    ranks,seen={},set()
    for lane in row['hits']:
        if lane['query'] in seen:continue
        seen.add(lane['query'])
        sources={}
        for hits in lane['sources'].values():
            for s in hits:
                if s['role']=='user' and not persona_source.is_persona_message(s['content']):
                    key=identity(s)
                    if key in sources and sources[key]['score']!=s['score']:
                        raise ValueError('Same query/source received conflicting BM25 score')
                    sources[key]=s
        ordered=sorted(sources,key=lambda key:(-sources[key]['score'],key))
        for rank,key in enumerate(ordered[:6],1):ranks[key]=ranks.get(key,0)+1/(60+rank)
    return sorted(ranks,key=lambda key:(-ranks[key],key))[:40]


def preparer(measured):
    by_id={r['search_id']:r for r in measured['rows']}
    def prepare(t):
        row=by_id[t['search_id']]
        if row['user_id']!=t['user_id']:raise ValueError('User mismatch')
        ctx=t['cascade']['conversation_context']
        if ctx['policy']!='fine_user_lexical_context_v1':raise ValueError('Require original lexical prefix')
        base=[dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
        if evidence_packet.digest(base)!=ctx['base_packet_hash']:raise ValueError('Base hash mismatch')
        pool={c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
              if c.get('view_status')!='stale' and '_packet_item' in c}
        fine=[pool[mid] for mid in t['cascade']['fine']['selected_ids'] if mid in pool and mid not in t['selection']['selected_ids']]
        kwargs=dict(user_id=t['user_id'],top_k=t['top_k'],byte_budget=t['coverage_manifest']['token_budget'])
        prefix,old=lexical(base,fine,**kwargs,reserve_bytes=5120,query=t['query'],options=t['options'])
        if [dict(p,packet_hash=evidence_packet.digest(prefix)) for p in prefix]!=t['returned']:
            raise ValueError('Historical packet not reproduced')
        remaining=5120-old['added_bytes'];kept={p['id'] for p in prefix}
        def full_key(s):
            if not usable(s,t['user_id']):return None
            span=s.get('content_span')
            if span and not (span['start']==0 and span['end']==span['original_length']==len(s['content'])):return None
            return identity(s)
        known={key for p in prefix for s in p['sources'] if (key:=full_key(s))}
        eligible,keys,costs=[],{},{}
        for mid in t['cascade']['coarse']['selected_ids']:
            if mid not in pool or mid in kept:continue
            c=pool[mid];group,missing=fine_evidence.closure(c,pool)
            if missing or not group<=kept|{mid}:continue
            _,trial=reserve(prefix,[c],**kwargs,reserve_bytes=remaining,limit=1)
            if not trial['proposed']:continue
            eligible.append(c);costs[mid]=trial['proposed'][0]['added_bytes']
            keys[mid]={key for s in c['_packet_item']['sources'] if (key:=full_key(s))}
        inventory=[]
        for rank,key in enumerate(anchor_ranks(row),1):
            carriers=sorted([c for c in eligible if key in keys[c['id']]],key=lambda c:(costs[c['id']],-c['_ce_score'],c['id']))
            inventory.append(dict(source_rank=rank,source_key=list(key[:2]),source_sha256=hashlib.sha256(key[2].encode()).hexdigest(),
                anchor_already_delivered=key in known,candidate_ids=[c['id'] for c in carriers],costs=[costs[c['id']] for c in carriers]))
        packets,proposals={'baseline':prefix},{}
        for mode,(depth,novel,extra) in VARIANTS.items():
            ordered,seen=[],set()
            for anchor in inventory:
                if anchor['source_rank']>depth or novel and anchor['anchor_already_delivered']:continue
                for mid in anchor['candidate_ids']:
                    if mid not in seen:ordered.append(pool[mid]);seen.add(mid)
            packets[mode],proposals[mode]=reserve(prefix,ordered,**kwargs,reserve_bytes=4096 if extra else remaining,limit=1)
            if packets[mode][:len(prefix)]!=prefix:raise ValueError('Original prefix changed')
        return dict(packets=packets,proposals=proposals,inventory=inventory,remaining_bytes=remaining)
    return prepare


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'))
    args=p.parse_args()
    source=args.run/'source-hit-identity-proposal.json'
    measured=json.loads(source.read_text())
    if (measured['questions']!=56 or measured['trace_sha256']!=hashlib.sha256((args.run/'search/server-search-debug.jsonl').read_bytes()).hexdigest()):
        raise ValueError('Incomplete or changed source-hit measurement')
    args.runs=[args.run/'control']
    with patch('socket.socket.connect',side_effect=AssertionError('Offline literal-anchor proposal')):
        score(args,preparer=preparer(measured),modes=tuple(VARIANTS),limitation=__doc__,script_path=Path(__file__))
    result=json.loads(args.output.read_text())
    for s in result['summary']:
        s['complete_sets']={m:sum(bool(r['coverage'][m]['user']) and all(r['coverage'][m]['user']) for r in result['rows'])
                            for m in ('baseline',s['mode'])}
    result['source_hit_measurement_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
