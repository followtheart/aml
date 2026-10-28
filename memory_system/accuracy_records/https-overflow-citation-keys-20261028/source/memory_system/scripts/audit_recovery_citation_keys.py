"""Reconstruct citation keys independently and check their source/response binding."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from batched_records import json_records,file_digest

# Deliberately independent from the runtime builder and its lookup table.
INSTRUCTIONS='''For each candidate, return target_id and citation_keys only.
If useful, choose the target ID and one or two keys from that candidate's enum.
A key is candidate:evidence_kind@source.part. It binds its candidate, kind and part. Two keys MUST share a kind.
An allowed key does not establish relevance. Select meaningful statements with subjects and qualifiers.
Use user_context for actual first-person circumstances including negative/past circumstances; topic_interest
for user questions; counterevidence for user denials/qualifications; third_party_context for others/fictional
context; assistant_context for relevant assistant material. This proves no personal fact.
Never change kinds just to force retention. If not useful, return target_id="" and citation_keys=[].
Do not generate quotes or new keys. Return every candidate key exactly once.
Call emit_json_result with the candidate-keyed object only.
'''


def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def text_digest(v):return hashlib.sha256(v.encode()).hexdigest()


def reconstruct(original):
    from app import listwise_recovery as recovery,recovery_spans
    cards={c['candidate_id']:c for c in original['cards']};maps={};properties={}
    for key,mid in original['candidate_map'].items():
        choices={}
        for kind in recovery_spans.KINDS:
            if kind=='none':continue
            for part,source in original['owners'][key].items():
                quote=original['spans'][part]['quote'].strip()
                if quote and recovery._check_citation(recovery.Citation(source_id=source,quote=quote),kind,cards[mid]['sources']) is None:
                    choices[f'{key}:{kind}@{part}']=dict(kind=kind,part=part)
        maps[key]=choices
        properties[key]={'type':'object','additionalProperties':False,'required':['target_id','citation_keys'],
            'properties':{'target_id':{'type':'string','enum':['']+list(original['targets'])},
                'citation_keys':{'type':'array','maxItems':2 if choices else 0,
                    'items':{'type':'string','enum':list(choices)} if choices else {'type':'string'}}}}
    old=original['prompt'];start=old.index('For each candidate, return useful,')
    end=old.index('Call emit_json_result with the candidate-keyed object only.\n',start)
    end+=len('Call emit_json_result with the candidate-keyed object only.\n')
    return old[:start]+INSTRUCTIONS+old[end:],dict(type='object',additionalProperties=False,
        required=list(properties),properties=properties),maps


def check_response(payload,original,maps,batch):
    from app import recovery_spans
    if not isinstance(payload,dict) or set(payload)-set(original['candidate_map']):
        assert batch['status']=='error' and not batch.get('judgments')
        return
    observed={j['candidate_id']:j for j in batch.get('judgments',[])}
    assert len(observed)==len(batch['judgments'])==len(original['candidate_map'])
    for key,mid in original['candidate_map'].items():
        raw=payload.get(key);valid_structure=False
        if isinstance(raw,dict) and set(raw)=={'target_id','citation_keys'}:
            target=raw['target_id'];tokens=raw['citation_keys']
            valid_structure=(isinstance(target,str) and target in ['',*original['targets']]
                and isinstance(tokens,list) and len(tokens)<=2 and bool(target)==bool(tokens)
                and all(isinstance(t,str) and t in maps[key] for t in tokens))
            if valid_structure:valid_structure=len({maps[key][t]['kind'] for t in tokens})<=1
        if not valid_structure:
            assert not observed[mid]['valid'] and observed[mid]['validation_errors']==['citation_key_error']
            assert observed[mid]['proposed_decision']==raw
            continue
        values=[maps[key][t] for t in tokens]
        translated=dict(useful=bool(tokens),target_id=target,evidence_kind=values[0]['kind'] if values else 'none',
            citation_parts=[v['part'] for v in values])
        sub=dict(original,candidate_map={key:mid},cards=[c for c in original['cards'] if c['candidate_id']==mid])
        _,rows=recovery_spans.decode({key:translated},sub)
        assert rows[0]==observed[mid],'decoded judgment differs from model choice or original validator'


def audit_batch(req,batch,enabled,stage):
    from app import config,listwise_recovery as recovery,recovery_spans
    dispatched='prompt_sha256' in batch;marker=batch.get('citation_keys')
    active=enabled and stage=='overflow'
    if not dispatched and marker is None:return Counter()
    cards=batch['sources'];old=recovery.request_for(req,cards);expected=old;stats=Counter(requests=int(dispatched))
    if not active:
        assert marker is None,'keys must be disabled outside enabled overflow'
    else:
        assert marker is not None
        original=recovery_spans.build(req,cards,role_bound=config.LISTWISE_ROLE_BOUND_REFS,
            abbreviations=config.LISTWISE_ABBREVIATION_PARTS,transform_context=config.LISTWISE_TRANSFORM_CONTEXT)
        prompt,schema,maps=reconstruct(original)
        bound=old['reservation']+len(prompt.encode())-len(old['prompt'].encode())
        bound+=len(json.dumps(schema).encode())-len(json.dumps(old['schema']).encode())
        assert marker['policy']=='owned_citation_keys_v1'
        assert marker['candidate_ids']==batch['candidate_ids']==[c['candidate_id'] for c in cards]
        assert marker['source_sha256']==digest(cards) and marker['citation_key_map_sha256']==digest(maps)
        assert marker['original_schema_sha256']==digest(old['schema']) and marker['proposed_schema_sha256']==digest(schema)
        assert marker['original_prompt_sha256']==text_digest(old['prompt']) and marker['proposed_prompt_sha256']==text_digest(prompt)
        assert marker['original_reservation']==old['reservation'] and marker['proposed_reservation']==bound
        assert marker['prompt_limit_bytes']==batch['prompt_limit_bytes']
        available=marker['token_headroom'];calls=marker['call_headroom'];seconds=marker['deadline_seconds']
        reason=('prompt_budget' if len(prompt.encode())>batch['prompt_limit_bytes'] else
                'token_budget' if available is not None and bound>available else
                'call_budget' if calls is not None and calls<1 else
                'deadline' if seconds is not None and seconds<=0 else None)
        assert marker['reason']==reason and marker['status']==('fallback' if reason else 'applied')
        if reason is None:expected=dict(old,prompt=prompt,schema=schema,reservation=bound)
        assert marker['dispatched_schema_sha256']==digest(expected['schema'])
        assert marker['dispatched_prompt_sha256']==text_digest(expected['prompt'])
        assert marker['dispatched_reservation']==expected['reservation']
        stats[marker['status']]+=1;stats['extra_reservation_bytes']+=expected['reservation']-old['reservation']
        if reason is None and 'response' in marker:check_response(marker['response'],original,maps,batch)
        elif reason is None:assert not batch.get('judgments') and batch['status'] in ['error','budget_skipped']
        else:assert 'response' not in marker
    if dispatched:
        assert batch['prompt_sha256']==text_digest(expected['prompt'])
        assert batch['input_bytes']==len(expected['prompt'].encode()) and batch['request_bound']==expected['reservation']
    else:assert batch['status']=='budget_skipped' and batch['reason']=='deadline'
    for j in batch.get('judgments',[]):
        if not j['valid']:continue
        card=next(c for c in cards if c['candidate_id']==j['candidate_id'])
        value={k:j[k] for k in ['candidate_id','useful','target_id','target_quote','evidence_kind','citations']}
        _,checked=recovery.validate(dict(decisions=[value]),[card],recovery.targets_for(req))
        assert checked[0]['valid'];stats['valid_decisions']+=1
    return stats


def audit_trace(trace):
    from app import config,listwise_recovery as recovery
    settings=trace['versions']['settings'];enabled=settings.get('OVERFLOW_CITATION_KEYS',False)
    assert not settings['LISTWISE_VALID_CITATION_CONTRACT'] and not settings['FINE_SOURCE_RRF']
    req=SimpleNamespace(query=trace['query'],options=trace['options']);stats=Counter()
    pool={c['id']:c for c in trace['ranked']};cascade=trace['cascade']
    with patch.multiple(config,**settings):
        for stage,review,eligible in [('recovery',cascade['listwise'].get('recovery',{}),cascade['listwise'].get('candidate_ids',[])),
                ('overflow',cascade.get('overflow',{}).get('review',{}),cascade.get('overflow',{}).get('proposed_ids',[]))]:
            for batch in review.get('batches') or [review]:
                for card in batch.get('sources',[]):
                    assert card['candidate_id'] in eligible and card['candidate_id'] in pool
                    assert card['sources']==recovery.sources_for(pool[card['candidate_id']])
                stats.update(audit_batch(req,batch,enabled,stage))
    stats['questions']+=1
    return stats


def main(run):
    results={r['search_id']:r['packet_hash'] for r in json_records(run/'completed/data/results/personamem-v2-32k.jsonl')}
    stats=Counter();seen=set()
    for trace in json_records(run/'server-search-debug.jsonl'):
        sid=trace['search_id'];assert sid in results and sid not in seen;seen.add(sid)
        assert trace['packet_hash']==results[sid];stats.update(audit_trace(trace))
    assert seen==set(results)
    paths=[Path(__file__),run/'server-search-debug.jsonl',run/'completed/data/results/personamem-v2-32k.jsonl']
    result=dict(status='passed',summary=dict(stats),input_sha256={str(p):file_digest(p) for p in paths})
    with (run/'recovery-citation-keys-audit.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(stats)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run.resolve())
