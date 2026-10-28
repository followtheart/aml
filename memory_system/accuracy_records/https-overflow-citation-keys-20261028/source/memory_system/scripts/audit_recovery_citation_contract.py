"""Independent contract reconstruction and terminal source review audit."""
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


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def schema_for(built):
    from app import listwise_recovery as check,recovery_spans
    cards={c['candidate_id']:c for c in built['cards']}
    properties={}
    for key,mid in built['candidate_map'].items():
        branches=[]
        for kind in ['none']+[k for k in recovery_spans.KINDS if k!='none']:
            positive=kind!='none';parts=[]
            if positive:
                for pid,source in built['owners'][key].items():
                    quote=built['spans'][pid]['quote'].strip()
                    if quote and check._check_citation(check.Citation(source_id=source,quote=quote),kind,cards[mid]['sources']) is None:
                        parts.append(pid)
                if not parts:continue
            branches.append(dict(type='object',additionalProperties=False,
                required=['useful','target_id','evidence_kind','citation_parts'],properties={
                    'useful':dict(type='boolean',enum=[positive]),
                    'target_id':dict(type='string',enum=list(built['targets']) if positive else ['']),
                    'evidence_kind':dict(type='string',enum=[kind]),
                    'citation_parts':dict(type='array',minItems=int(positive),maxItems=2 if positive else 0,
                        items=dict(type='string',enum=parts) if positive else dict(type='string'))}))
        properties[key]=dict(anyOf=branches)
    return dict(built['schema'],properties=properties)


def audit_batch(req,batch,enabled):
    from app import config,listwise_recovery,recovery_spans
    marker=batch.get('citation_contract')
    dispatched='prompt_sha256' in batch
    if not dispatched and marker is None:return Counter()
    if not enabled:
        assert marker is None,'contract present while disabled'
    cards=batch.get('sources')
    if cards is None:
        assert marker and batch.get('status')=='budget_skipped' and batch.get('reason')=='deadline'
        return Counter(deadline_before_dispatch=1)
    old=listwise_recovery.request_for(req,cards)
    expected=old
    counts=Counter(requests=1)
    if enabled:
        assert marker is not None,'missing contract marker'
        built=recovery_spans.build(req,cards,role_bound=config.LISTWISE_ROLE_BOUND_REFS,
            transform_context=config.LISTWISE_TRANSFORM_CONTEXT,abbreviations=config.LISTWISE_ABBREVIATION_PARTS)
        proposed=schema_for(built)
        bound=old['reservation']+len(json.dumps(proposed).encode())-len(json.dumps(old['schema']).encode())
        assert marker['policy']=='existing_citation_validator_schema_v1'
        assert marker['candidate_ids']==[c['candidate_id'] for c in cards]==batch['candidate_ids']
        assert marker['original_schema_sha256']==digest(old['schema'])
        assert marker['proposed_schema_sha256']==digest(proposed)
        assert marker['original_reservation']==old['reservation'] and marker['proposed_reservation']==bound
        assert marker['source_sha256']==digest(cards)
        assert marker['prompt_sha256']==hashlib.sha256(old['prompt'].encode()).hexdigest()
        available=marker['token_headroom'];calls=marker['call_headroom'];seconds=marker['deadline_seconds']
        reason=('token_budget' if available is not None and bound>available else
                'call_budget' if calls is not None and calls<1 else
                'deadline' if seconds is not None and seconds<=0 else None)
        assert marker['reason']==reason
        assert marker['status']==('fallback' if reason else 'applied')
        if reason is None:expected=dict(old,schema=proposed,reservation=bound)
        assert marker['dispatched_schema_sha256']==digest(expected['schema'])
        assert marker['dispatched_reservation']==expected['reservation']
        counts[marker['status']]+=1
        counts['extra_reservation_bytes']+=expected['reservation']-old['reservation']
    assert batch['prompt_sha256']==hashlib.sha256(expected['prompt'].encode()).hexdigest()
    assert batch['input_bytes']==len(expected['prompt'].encode())
    assert batch['request_bound']==expected['reservation']
    for judgment in batch.get('judgments',[]):
        if not judgment.get('valid'):continue
        card=next(c for c in cards if c['candidate_id']==judgment['candidate_id'])
        value={k:judgment[k] for k in ['candidate_id','useful','target_id','target_quote','evidence_kind','citations']}
        _,checked=listwise_recovery.validate(dict(decisions=[value]),[card],listwise_recovery.targets_for(req))
        assert checked[0]['valid'],'invalid citation marked valid'
        counts['valid_decisions']+=1
    return counts


def audit_trace(trace):
    from app import config,listwise_recovery
    stats=Counter();settings=trace['versions']['settings']
    enabled=settings['LISTWISE_VALID_CITATION_CONTRACT']
    assert not settings['FINE_SOURCE_RRF']
    req=SimpleNamespace(query=trace['query'],options=trace['options'])
    with patch.multiple(config,**settings):
        pool={r['id']:r for r in trace['ranked']}
        stages=[(trace['cascade']['listwise'].get('recovery',{}),trace['cascade']['listwise'].get('candidate_ids',[])),
                (trace['cascade'].get('overflow',{}).get('review',{}),trace['cascade'].get('overflow',{}).get('proposed_ids',[]))]
        for review,eligible in stages:
            for batch in review.get('batches') or [review]:
                for card in batch.get('sources',[]):
                    mid=card['candidate_id']
                    assert mid in eligible and mid in pool,'review outside admitted candidates'
                    assert card['sources']==listwise_recovery.sources_for(pool[mid]),'source differs from immutable ranked unit'
                stats.update(audit_batch(req,batch,enabled))
    stats['questions']+=1
    return stats


def main(run):
    results={r['search_id']:r for r in json_records(run/'completed/data/results/personamem-v2-32k.jsonl')}
    seen=set();stats=Counter()
    for trace in json_records(run/'server-search-debug.jsonl'):
        sid=trace['search_id'];assert sid in results and sid not in seen;seen.add(sid)
        assert trace['packet_hash']==results[sid]['packet_hash']
        stats.update(audit_trace(trace))
    assert seen==set(results)
    paths=[run/'server-search-debug.jsonl',run/'completed/data/results/personamem-v2-32k.jsonl',Path(__file__)]
    report=dict(status='passed',summary=dict(stats),input_sha256={str(p):file_digest(p) for p in paths})
    with (run/'recovery-citation-audit.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report['summary']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run.resolve())
