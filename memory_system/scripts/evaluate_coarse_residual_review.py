"""One real bounded review of fixed coarse outsiders on a complete HTTPS cohort.

Baseline Search outputs stay frozen. No reader calls, selected-question reruns,
or retries. This diagnostic measures validated additions and source transport,
not end-to-end HTTPS accuracy. All original questions, including errors, remain.
"""
import argparse
import asyncio
from collections import Counter
import copy
import csv
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import sys
import time
from unittest.mock import patch


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def records(path):
    with path.open() as stream:return [json.loads(line) for line in stream if line.strip()]
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


async def main(args):
    run=args.run.resolve();root=run.parent;preflight=read(args.preflight)
    capacity=read(args.capacity);source_plan=read(root/'plan.json')
    assert preflight['summary']['mode']=='separate-batch'
    assert capacity['summary']['questions']==source_plan['questions']==preflight['summary']['questions']
    for report in (preflight,capacity):
        for name,expected in report.get('dependencies_sha256',{}).items():assert sha(Path(name))==expected,name
    for name,expected in capacity['inputs'].items():assert sha(Path(name))==expected,name
    for item in preflight['inputs']:
        assert Path(item['path']).stat().st_size==item['prefix_bytes']
        assert sha(Path(item['path']))==item['prefix_sha256']
    for name,expected in source_plan['source_files'].items():assert sha(root/'source'/name)==expected,name
    snapshot=root/'source/memory_system'
    sys.path.insert(0,str(snapshot));sys.path.insert(0,str(snapshot/'scripts'))
    from app import budget,config,evidence_packet,fine_context_reserve,fine_evidence,llm,listwise_recovery,recovery_spans,schemas
    from evaluation_model_guard import check_runtime,check_versions
    from evaluate_recall_context import reference_turns
    from audit_https_funnel import turn_coverage
    check_runtime();assert not config.FAKE
    manifest=read(run/'isolated-manifest.json');prepared=Path(manifest['data'])
    assert sha(prepared)==manifest['data_sha256']
    traces={t['search_id']:t for t in records(run/'server-search-debug.jsonl')}
    assert len(traces)==source_plan['questions']
    jobs=[]
    for proposal in preflight['rows']:
        t=traces[proposal['search_id']];check_versions(t['versions'],'frozen Search')
        lookup={c['id']:c for c in t['ranked']}
        pending=[lookup[mid] for mid in proposal['proposed_ids']]
        kept=[lookup[mid] for mid in t['cascade']['overflow']['selected_ids']]
        req=schemas.SearchRequest(user_id=t['user_id'],query=t['query'],options=t['options'])
        request=None
        with patch.multiple(config,**t['versions']['settings']):
            assert config.LISTWISE_SPAN_REFS
            if pending:
                cards=[dict(candidate_id=c['id'],sources=listwise_recovery.sources_for(c)) for c in pending]
                request=listwise_recovery.request_for(req,cards)
                assert len(pending)<=4 and len(request['prompt'].encode())<=12000
                assert sha_text(request['prompt'])==proposal['request_sha256']
                assert request['reservation']==proposal['request_reservation']<=proposal['remainder']['tokens']
                assert proposal['remainder']['calls']>=1 and proposal['remainder']['conservative_seconds_proxy']>0
        jobs.append((proposal,t,lookup,pending,kept,req,request))
    assert len(jobs)==source_plan['questions']
    args.output.mkdir(parents=True,exist_ok=False)
    plan=dict(protocol=__doc__,run=str(run),questions=len(jobs),model=config.LLM_MODEL,
        planned_calls=sum(bool(j[3]) for j in jobs),source_snapshot=str(snapshot),
        inputs={str(p.resolve()):sha(p) for p in (args.preflight,args.capacity,prepared,args.gold_csv,
            root/'plan.json',root/'final-evaluation.json',run/'memory.db',Path(__file__))},
        jobs=[dict(qa_id=p['qa_id'],search_id=t['search_id'],candidate_ids=[c['id'] for c in pending],
            request_sha256=digest(request) if request else None,remainder=p['remainder'])
            for p,t,_,pending,_,_,request in jobs])
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    handler=logging.FileHandler(args.output/'provider.log',mode='x')
    logger=logging.getLogger('aml.metrics');logger.addHandler(handler);logger.setLevel(logging.INFO)
    original=llm.complete_json;events=[];packets=[]
    for proposal,t,lookup,pending,kept,req,expected_request in jobs:
        captured=[]
        async def capture(prompt,**kwargs):
            actual=dict(prompt=prompt,**{k:v for k,v in kwargs.items() if k not in ('stage','attempts','timeout')})
            expected={k:v for k,v in expected_request.items() if k!='reservation'}
            assert actual==expected and kwargs['attempts']==1
            entry=dict(request=copy.deepcopy(actual),stage=kwargs['stage'],attempts=kwargs['attempts'])
            captured.append(entry)
            try:
                payload=await original(prompt,**kwargs)
                entry['payload']=copy.deepcopy(payload);return payload
            except BaseException as exc:
                entry['error_type']=type(exc).__name__;raise
        review=dict(status='not_needed',restored_ids=[],judgments=[],errors=[])
        calls=tokens=0
        if pending:
            with patch.multiple(config,**t['versions']['settings']):
                check_runtime()
                with budget.scope(seconds=proposal['remainder']['conservative_seconds_proxy'],calls=1,
                                  tokens=proposal['remainder']['tokens']) as usage,patch.object(llm,'complete_json',capture):
                    _,review=await listwise_recovery._recover_batch(req,pending,kept,set(range(len(pending))),
                        stage='diagnostic.coarse_residual_review',prompt_limit_bytes=12000)
                calls,tokens=usage.calls,usage.tokens
            assert calls<=1 and len(captured)<=1
            if captured:
                assert review['candidate_ids']==proposal['proposed_ids']
                assert review['prompt_sha256']==proposal['request_sha256']
        good=set(review['restored_ids'])
        packed=t['returned'];assert evidence_packet.digest(packed)==t['packet_hash']
        used=sum(len(p['content'].encode()) for p in packed)+max(0,len(packed)-1)
        byte_limit=t['coverage_manifest']['token_budget']
        extra=[c for c in pending if c['id'] in good]
        packet,packing=fine_context_reserve.propose(packed,extra,user_id=t['user_id'],top_k=t['top_k'],
            byte_budget=byte_limit,reserve_bytes=byte_limit-used,limit=4)
        blocked=set()
        while True:
            ids={p['id'] for p in packet};bad=set()
            for item in packet[len(packed):]:
                group,missing=fine_evidence.closure(lookup[item['id']],lookup)
                if missing or not group<=ids:bad.add(item['id'])
            if not bad:break
            blocked.update(bad);packet=[p for p in packet if p['id'] not in bad]
        assert packet[:len(packed)]==packed and len(packet)<=t['top_k']
        size=sum(len(p['content'].encode()) for p in packet)+max(0,len(packet)-1)
        assert size<=byte_limit
        event=dict(qa_id=proposal['qa_id'],conversation_id=proposal['conversation_id'],search_id=t['search_id'],
            calls=calls,tokens=tokens,review=review,model_calls=captured,packing=packing,
            dependency_blocked_ids=sorted(blocked),appended_ids=[p['id'] for p in packet[len(packed):]],
            added_bytes=size-used,packet_sha256=evidence_packet.digest(packet))
        with (args.output/'responses.jsonl').open('a') as stream:stream.write(json.dumps(event,ensure_ascii=False)+'\n')
        events.append(event);packets.append((t,packet))
        print(json.dumps(dict(qa_id=event['qa_id'],status=review['status'],calls=calls,tokens=tokens,
            appended=len(event['appended_ids']))),flush=True)
    handler.flush()
    # Only now score reference transport; prompts and selection never read gold.
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream:gold=list(csv.DictReader(stream))
    conversations={c['conversation_id']:c for c in records(prepared)};raw={}
    with sqlite3.connect((run/'memory.db').as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        for source in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(source['user_id'],{})[source['request_id'],source['message_index']]=dict(source)
    rows=[]
    for event,(t,packet) in zip(events,packets):
        conv=conversations[event['conversation_id']];q=next(q for q in conv['qa'] if q['id']==event['qa_id'])
        assert q['question']==t['query'] and q['options']==t['options']
        turns=reference_turns(q,conv['source_history'],gold[int(event['qa_id'])])
        def flags(items):
            cards=[dict(s,id=f'{i}:{j}',text=s.get('content','')) for i,p in enumerate(items) for j,s in enumerate(p.get('sources',[]))]
            return [v['full'] for v in turn_coverage(turns,cards,raw[t['user_id']])]
        before,after=flags(t['returned']),flags(packet)
        assert len(before)==len(after) and all(not a or b for a,b in zip(before,after))
        rows.append(dict(qa_id=event['qa_id'],before=before,after=after,appended_ids=event['appended_ids']))
    summary=dict(questions=len(rows),calls=sum(e['calls'] for e in events),tokens=sum(e['tokens'] for e in events),
        statuses=dict(Counter(e['review']['status'] for e in events)),appended_items=sum(len(e['appended_ids']) for e in events),
        added_bytes=sum(e['added_bytes'] for e in events),reference_questions={a:sum(any(r[a]) for r in rows) for a in ('before','after')},
        reference_turns={a:sum(sum(r[a]) for r in rows) for a in ('before','after')},
        gained=[r['qa_id'] for r in rows if any(r['after']) and not any(r['before'])],
        gained_turn_questions=[r['qa_id'] for r in rows if any(b and not a for a,b in zip(r['before'],r['after']))])
    (args.output/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows,limitation=__doc__),indent=2)+'\n')
    (args.output/'exit-code').write_text('0\n')
    print(json.dumps(summary),flush=True)


def sha_text(value):return hashlib.sha256(value.encode()).hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('run','preflight','capacity','output'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    asyncio.run(main(p.parse_args()))
