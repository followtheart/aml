"""Missing-source anchors versus carrier novelty under explicit context budgets.

The older semantic residual can nominate an already-delivered source whose
carrier contains a different new message. Compare requiring the actual ranked
user source to be missing, with the same complete prepared coarse carriers.
Separate source-rank depth and the explicit extra 4 KiB allowance. Original
packet/lexical seats, roles, dependencies, top_k and total byte cap stay intact.
Use saved vectors and original CE only; no new retrieval facts or provider calls.
All whole-cohort selections finish before gold scoring.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import main as score
from probe_semantic_source_context import preparer as old_preparer, ranked_sources
from app import evidence_packet, fine_evidence, search_pipeline
from app.conversation_context import usable
from app.fine_context_lexical import propose as lexical
from app.fine_context_reserve import propose as reserve

VARIANTS = {
    'historical8_remainder': (8, False, False),
    'novel8_remainder': (8, True, False),
    'novel40_remainder': (40, True, False),
    'any40_extra4k': (40, False, True),
    'novel40_extra4k': (40, True, True),
}
MODES = tuple(f'{view}_{variant}' for view in ('whole','window2','exchange') for variant in VARIANTS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preparer(measurements, previous):
    by_query = {view:{(r['user_id'],tuple(q['query']['text'] for q in r['modes'][view])):r
        for r in ranks['rankings']} for view,(plan,ranks) in measurements.items()}
    old = {view:old_preparer(plan,ranks,(view,)) for view,(plan,ranks) in measurements.items()}

    def prepare(t):
        ctx=t['cascade']['conversation_context']
        if ctx['policy']!='fine_user_lexical_context_v1':
            raise ValueError('Original lexical prefix required')
        base=[dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
        if evidence_packet.digest(base)!=ctx['base_packet_hash']:
            raise ValueError('Base hash mismatch')
        pool={c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
            if c.get('view_status')!='stale' and '_packet_item' in c}
        fine=[pool[mid] for mid in t['cascade']['fine']['selected_ids']
              if mid in pool and mid not in t['selection']['selected_ids']]
        kwargs=dict(user_id=t['user_id'],top_k=t['top_k'],byte_budget=t['coverage_manifest']['token_budget'])
        prefix,original=lexical(base,fine,**kwargs,reserve_bytes=5120,query=t['query'],options=t['options'])
        if [dict(p,packet_hash=evidence_packet.digest(prefix)) for p in prefix]!=t['returned']:
            raise ValueError('Full original packet mismatch')
        remaining=5120-original['added_bytes']
        kept={p['id'] for p in prefix}
        def identity(s):
            if not usable(s,t['user_id']):return None
            span=s.get('content_span')
            if span and not (span.get('start')==0 and span.get('end')==span.get('original_length')==len(s['content'])):
                return None
            return s['request_id'],s['message_index'],s['content']
        known={key for p in prefix for s in p['sources'] if (key:=identity(s))}
        eligible,costs,identities=[],{},{}
        for mid in t['cascade']['coarse']['selected_ids']:
            if mid not in pool or mid in kept:continue
            c=pool[mid]
            group,missing=fine_evidence.closure(c,pool)
            if missing or not group<=kept|{mid}:continue
            _,trial=reserve(prefix,[c],**kwargs,reserve_bytes=remaining,limit=1)
            if not trial['proposed']:continue
            costs[mid]=trial['proposed'][0]['added_bytes']
            identities[mid]={key for s in c['_packet_item']['sources'] if (key:=identity(s))}
            eligible.append(c)
        packets,proposals,inventory={'baseline':prefix},{},[]
        for view,(plan,rankings) in measurements.items():
            measured=by_query[view][t['user_id'],tuple(s['text'] for s in t['query_specs'])]
            order=ranked_sources(measured['modes'][view])[:40]
            anchors=[]
            for rank,si in enumerate(order,1):
                source=plan['sources'][si]
                if source['user_id']!=t['user_id']:raise ValueError('Cross-user measurement')
                key=source['request_id'],source['message_index'],source['content']
                exchange=plan.get('exchange_sources',{}).get(str(si),[source])
                required={(s['request_id'],s['message_index'],s['content']) for s in exchange}
                carriers=sorted([c for c in eligible if key in identities[c['id']]
                    and required<=identities[c['id']]|known],key=lambda c:(costs[c['id']],-c['_ce_score'],c['id']))
                detail=dict(view=view,source_index=si,source_rank=rank,anchor_already_delivered=key in known,
                    candidate_ids=[c['id'] for c in carriers],costs=[costs[c['id']] for c in carriers])
                inventory.append(detail)
                anchors.append((detail,carriers))
            for variant,(depth,novel,extra) in VARIANTS.items():
                mode=f'{view}_{variant}'
                ordered,seen=[],set()
                for detail,carriers in anchors:
                    if detail['source_rank']>depth or novel and detail['anchor_already_delivered']:continue
                    for c in carriers:
                        if c['id'] not in seen:
                            ordered.append(c);seen.add(c['id'])
                # EXTRA is beyond the entire original lexical prefix; no hidden
                # reuse of remaining 5 KiB and no change to global request cap.
                allowance=4096 if extra else remaining
                packets[mode],prop=reserve(prefix,ordered,**kwargs,reserve_bytes=allowance,limit=1)
                proposals[mode]=dict(prop,anchor_depth=depth,require_missing_anchor=novel,
                    allowance_kind='extra4k' if extra else 'original_remainder',
                    ranked_source_indices=order[:depth])
                if packets[mode][:len(prefix)]!=prefix:raise ValueError('Prefix changed')
            prior=old[view](t)
            historical=f'{view}_historical8_remainder'
            if packets[historical]!=prior['packets'][view]:
                raise ValueError('Historical semantic policy not reproduced')
            prop=proposals[historical]
            for key in ('appended_ids','added_bytes','total_bytes','proposed'):
                if prop[key]!=prior['proposals'][view][key]:raise ValueError('Historical proposal differs')
            # Verify against independently saved original output as well.
            saved=previous[view][t['search_id']]
            for key in ('appended_ids','added_bytes','total_bytes','proposed'):
                if prop[key]!=saved['proposals'][view][key]:raise ValueError('Archived proposal differs')
        return dict(packets=packets,proposals=proposals,inventory=inventory,remaining_bytes=remaining)
    return prepare


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'))
    args=p.parse_args()
    measurements,hashes={},{}
    for directory,views in (('source-vector-diagnostic',('whole','window2')),('exchange-vector-diagnostic',('exchange',))):
        root=args.run/directory
        plan=json.loads((root/'plan.json').read_text());ranks=json.loads((root/'rankings.json').read_text())
        if digest(root/'plan.json')!=ranks['plan_sha256'] or digest(root/'vectors.npz')!=ranks['vectors_sha256']:
            raise ValueError('Measurement changed')
        for view in views:measurements[view]=(plan,ranks)
        hashes[directory]=dict(plan_sha256=digest(root/'plan.json'),rankings_sha256=digest(root/'rankings.json'))
    ids={j['qa']['id']:j['payload']['search_id'] for j in map(json.loads,(args.run/'http-reader-inputs.jsonl').open())}
    previous={}
    for name,views in (('semantic-source-context-proposal.json',('whole','window2')),('exchange-source-context-proposal.json',('exchange',))):
        data=json.loads((args.run/name).read_text())
        for view in views:previous[view]={ids[r['qa_id']]:r for r in data['rows']}
        hashes[name]=digest(args.run/name)
    args.runs=[args.run/'control']
    with patch('socket.socket.connect',side_effect=AssertionError('Offline semantic budget study')):
        score(args,preparer=preparer(measurements,previous),modes=MODES,limitation=__doc__,script_path=Path(__file__))
    result=json.loads(args.output.read_text())
    for summary in result['summary']:
        mode=summary['mode']
        summary['complete_sets']={m:sum(bool(r['coverage'][m]['user']) and all(r['coverage'][m]['user'])
            for r in result['rows']) for m in ('baseline',mode)}
    result['measured_inputs']=hashes
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
