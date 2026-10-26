"""Reconstruct complete coarse-review requests, validation, packets and ledger.

Offline only. Formal citation validity is not semantic relevance; this audit
does not establish reader accuracy or unseen online timing behavior.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
from unittest.mock import patch


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def records(path):
    with path.open() as stream:return [json.loads(line) for line in stream if line.strip()]
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def main(args):
    output=args.probe.resolve();plan=read(output/'plan.json');report=read(output/'result.json')
    assert (output/'exit-code').read_text().strip()=='0'
    for name,expected in plan['inputs'].items():assert sha(Path(name))==expected,name
    run=Path(plan['run']);snapshot=Path(plan['source_snapshot']);source_plan=read(run.parent/'plan.json')
    for name,expected in source_plan['source_files'].items():assert sha(run.parent/'source'/name)==expected,name
    sys.path.insert(0,str(snapshot));sys.path.insert(0,str(snapshot/'scripts'))
    from app import config,evidence_packet,fine_context_reserve,fine_evidence,listwise_recovery,recovery_spans,schemas
    from evaluate_recall_context import reference_turns
    from audit_https_funnel import turn_coverage
    events=records(output/'responses.jsonl')
    assert len(events)==len(report['rows'])==len(plan['jobs'])==plan['questions']
    traces={t['search_id']:t for t in records(run/'server-search-debug.jsonl')}
    fixed=[];dispatches=0
    with patch('socket.socket.connect',side_effect=AssertionError('Offline audit')):
        for job,event,row in zip(plan['jobs'],events,report['rows']):
            assert job['qa_id']==event['qa_id']==row['qa_id'] and job['search_id']==event['search_id']
            t=traces[job['search_id']];lookup={c['id']:c for c in t['ranked']}
            pending=[lookup[mid] for mid in job['candidate_ids']];review=event['review'];calls=event['model_calls']
            assert event['calls'] in (0,1) and len(calls)<=1
            req=schemas.SearchRequest(user_id=t['user_id'],query=t['query'],options=t['options'])
            with patch.multiple(config,**t['versions']['settings']):
                if calls:
                    entry=calls[0];dispatches+=1
                    assert entry['attempts']==1 and entry['stage']=='diagnostic.coarse_residual_review'
                    cards=[dict(candidate_id=c['id'],sources=listwise_recovery.sources_for(c)) for c in pending]
                    request=listwise_recovery.request_for(req,cards)
                    assert digest(request)==job['request_sha256']
                    assert entry['request']=={k:v for k,v in request.items() if k!='reservation'}
                    assert cards==review['sources'] and [c['id'] for c in pending]==review['candidate_ids']
                    assert len(request['prompt'].encode())==review['input_bytes']<=12000
                    assert request['reservation']==review['request_bound']<=job['remainder']['tokens']
                    assert hashlib.sha256(request['prompt'].encode()).hexdigest()==review['prompt_sha256']
                    if 'payload' in entry:
                        try:
                            restored,judgments=recovery_spans.decode_partial(entry['payload'],
                                recovery_spans.build(req,cards,abbreviations=config.LISTWISE_ABBREVIATION_PARTS))
                        except Exception as exc:
                            assert review['status']=='error' and review['error_type']==type(exc).__name__
                            assert not review['restored_ids']
                        else:
                            assert judgments==review['judgments']
                            assert [c['id'] for c in pending if c['id'] in restored]==review['restored_ids']
                    else:
                        assert review['status']=='error' and not review['restored_ids']
                        # asyncio.timeout cancels the awaited provider coroutine,
                        # then exposes TimeoutError to the enclosing review.
                        assert (entry['error_type']==review['error_type'] or
                                (entry['error_type']=='CancelledError' and review['error_type']=='TimeoutError'
                                 and review['time_limit_seconds']>0))
                else:assert not review['restored_ids']
            packed=t['returned'];used=sum(len(p['content'].encode()) for p in packed)+max(0,len(packed)-1)
            accepted=set(review['restored_ids']);extra=[c for c in pending if c['id'] in accepted]
            packet,packing=fine_context_reserve.propose(packed,extra,user_id=t['user_id'],top_k=t['top_k'],
                byte_budget=t['coverage_manifest']['token_budget'],reserve_bytes=t['coverage_manifest']['token_budget']-used,limit=4)
            assert packing==event['packing']
            blocked=set()
            while True:
                ids={p['id'] for p in packet};bad=set()
                for item in packet[len(packed):]:
                    group,missing=fine_evidence.closure(lookup[item['id']],lookup)
                    if missing or not group<=ids:bad.add(item['id'])
                if not bad:break
                blocked.update(bad);packet=[p for p in packet if p['id'] not in bad]
            assert sorted(blocked)==event['dependency_blocked_ids']
            assert packet[:len(packed)]==packed and len(packet)<=t['top_k']
            assert [p['id'] for p in packet[len(packed):]]==event['appended_ids']==row['appended_ids']
            size=sum(len(p['content'].encode()) for p in packet)+max(0,len(packet)-1)
            assert size-used==event['added_bytes'] and size<=t['coverage_manifest']['token_budget']
            assert evidence_packet.digest(packet)==event['packet_sha256']
            fixed.append((event,t,packet,row))
    prepared=Path(read(run/'isolated-manifest.json')['data']);conversations={c['conversation_id']:c for c in records(prepared)}
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream:gold=list(csv.DictReader(stream))
    assert str(args.gold_csv.resolve()) in plan['inputs']
    raw={}
    with sqlite3.connect((run/'memory.db').as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        for source in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(source['user_id'],{})[source['request_id'],source['message_index']]=dict(source)
    for event,t,packet,row in fixed:
        conv=conversations[event['conversation_id']];q=next(q for q in conv['qa'] if q['id']==event['qa_id'])
        assert q['question']==t['query'] and q['options']==t['options']
        turns=reference_turns(q,conv['source_history'],gold[int(event['qa_id'])])
        for name,items in (('before',t['returned']),('after',packet)):
            cards=[dict(s,id=f'{i}:{j}',text=s.get('content','')) for i,p in enumerate(items) for j,s in enumerate(p.get('sources',[]))]
            assert [v['full'] for v in turn_coverage(turns,cards,raw[t['user_id']])]==row[name]
        assert all(not a or b for a,b in zip(row['before'],row['after']))
    metrics=[]
    for line in (output/'provider.log').read_text().splitlines():
        if 'provider_call ' not in line:continue
        fields=dict(re.findall(r'(\w+)=([^\s]+)',line))
        assert fields['model']==plan['model'] and fields['kind']=='llm'
        assert fields['stage']=='diagnostic.coarse_residual_review'
        assert fields['attempt']=='1' and fields['retries']=='0'
        metrics.append(fields)
    assert len(metrics)==sum(e['calls'] for e in events)==report['summary']['calls']
    assert sum(int(m['total_tokens']) for m in metrics)==sum(e['tokens'] for e in events)==report['summary']['tokens']
    assert report['summary']['questions']==len(events)
    for name in ('before','after'):
        assert report['summary']['reference_questions'][name]==sum(any(r[name]) for r in report['rows'])
        assert report['summary']['reference_turns'][name]==sum(sum(r[name]) for r in report['rows'])
    result=dict(status='passed',questions=len(events),requests_reconstructed=dispatches,
        provider_calls=len(metrics),provider_error_calls=sum(m['status']!='ok' for m in metrics),
        preserved_prefixes=len(events),provider_calls_during_audit=0,
        limitation=__doc__+' Failed/cancelled provider calls may have unreported billed usage.',
        inputs={str(p.resolve()):sha(p) for p in (output/'plan.json',output/'result.json',output/'responses.jsonl',
            output/'provider.log',output/'exit-code',Path(__file__))})
    with (output/'audit.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
