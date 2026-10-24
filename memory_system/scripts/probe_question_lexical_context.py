"""Factorial residual-source probe: question-only versus all-target lexical rank.

Compare literal words and existing WordNet first-noun normalization, with the
same scored coarse corpus and complete-source eligibility. Keep every original
packet item and all three context seats, add at most one whole candidate inside
the remaining original 5 KiB and total request caps. No topic-novelty gate:
an already represented word does not imply the same source was delivered.
No providers or personal-fact labels; gold is consulted after all selections.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import main as score
from wordnet_context_diagnostic import NounConcepts, scores as noun_scores
from app import config, evidence_packet, fine_evidence, search_pipeline
from app.fine_context_lexical import propose as lexical, source_scores
from app.fine_context_reserve import propose as reserve

MODES=('all_targets_literal','question_literal','all_targets_noun','question_noun')


def preparer(concepts):
    def prepare(t):
        ctx=t['cascade']['conversation_context']
        if ctx['policy']!='fine_user_lexical_context_v1':
            raise ValueError('Require original lexical control')
        base=[dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
        if evidence_packet.digest(base)!=ctx['base_packet_hash']:
            raise ValueError('Original base packet changed')
        pool={c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
            if c.get('view_status')!='stale' and '_packet_item' in c}
        fine=[pool[mid] for mid in t['cascade']['fine']['selected_ids'] if mid in pool and mid not in t['selection']['selected_ids']]
        kw=dict(user_id=t['user_id'],top_k=t['top_k'],byte_budget=t['coverage_manifest']['token_budget'])
        prefix,old=lexical(base,fine,**kw,reserve_bytes=5120,query=t['query'],options=t['options'])
        digest=evidence_packet.digest(prefix)
        if [dict(p,packet_hash=digest) for p in prefix]!=t['returned']:
            raise ValueError('Original packet did not reproduce')
        remaining=5120-old['added_bytes']
        corpus=[pool[mid] for mid in t['cascade']['coarse']['selected_ids'] if mid in pool]
        kept={p['id'] for p in prefix}
        eligible=[];costs={}
        for c in corpus:
            group,missing=fine_evidence.closure(c,pool)
            if missing or not group<=kept|{c['id']}:
                continue
            _,trial=reserve(prefix,[c],**kw,reserve_bytes=remaining,limit=1)
            if not trial['appended_ids']:
                continue
            eligible.append(c);costs[c['id']]=trial['added_bytes']
        packets=dict(baseline=prefix);proposals={};inventory=[]
        for mode in MODES:
            options=[] if mode.startswith('question_') else t['options']
            args=dict(user_id=t['user_id'],query=t['query'],options=options)
            values,matches=(noun_scores(corpus,**args,concepts=concepts)
                            if mode.endswith('_noun') else source_scores(corpus,**args))
            ordered=sorted([c for c in eligible if values.get(c['id'],0)>0],
                key=lambda c:(-values[c['id']],-c['_ce_score'],c['id']))
            packets[mode],proposals[mode]=reserve(prefix,ordered,**kw,reserve_bytes=remaining,limit=1)
            proposals[mode]['ordered_ids']=[c['id'] for c in ordered]
            inventory.extend(dict(mode=mode,candidate_id=c['id'],score=values[c['id']],
                original_ce=c['_ce_score'],cost=costs[c['id']],matches=matches[c['id']]) for c in ordered)
            if packets[mode][:len(prefix)]!=prefix:
                raise ValueError('Original packet changed')
        return dict(packets=packets,proposals=proposals,inventory=inventory,remaining_bytes=remaining)
    return prepare


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wordnet',type=Path,default=Path('data/lexicons/wordnet.zip'))
    p.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'))
    args=p.parse_args();concepts=NounConcepts(args.wordnet)
    with patch('socket.socket.connect',side_effect=AssertionError('Offline diagnostic')):
        score(args,preparer=preparer(concepts),modes=MODES,limitation=__doc__,script_path=Path(__file__))
    result=json.loads(args.output.read_text())
    result['lexical_resources']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (args.wordnet,Path(__file__).with_name('wordnet_context_diagnostic.py'))}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
