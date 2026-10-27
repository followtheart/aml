"""Offline verification of complete paired CE views, batches, failures and stages."""
import argparse
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context
import probe_ce_bridge_summary as probe
from probe_ce_relevance_guidance import sha, records
import audit_https_funnel as funnel


def metrics_from(path):
    pending=b''; result=[]; logged=[]
    with path.open('rb') as stream:
        while block:=stream.read(4*1024*1024):
            batch=(pending+block).split(b'\n'); pending=batch.pop()
            for raw in batch:
                line=raw.decode('utf-8')
                if line.startswith('provider_call '):
                    result.append(dict(re.findall(r'(\w+)=([^\s]+)',line)))
                elif line.startswith('{'):
                    row=json.loads(line)
                    if 'qa_id' in row and 'arm' in row:logged.append(row)
    if pending.strip():
        raise ValueError('Unterminated provider log record')
    return result,logged


def audit(directory, log, gold_csv):
    plan=json.loads((directory/'plan.json').read_text())
    assert plan['policy']==probe.POLICY
    assert plan['diagnostic_dispatch']==dict(rate_limit_retries=0,ordinary_attempts=1,
        arm_seconds=240,arm_max_calls=100,arm_max_tokens=1000000)
    run=Path(plan['run'])
    for path,key in ((run/'server-search-debug.jsonl','trace_sha256'),
                     (run/'completed/data/results/personamem-v2-32k.jsonl','result_sha256'),
                     (run/'funnel/stage-transport.json','stage_sha256'),
                     (run/'memory.db','database_sha256'),(Path(plan['prepared']),'data_sha256')):
        assert sha(path)==plan[key],key
    for path,digest in plan['script_hashes'].items():assert sha(Path(path))==digest,path
    repo=Path(__file__).resolve().parents[2]
    for path,digest in plan['versions']['file_hashes'].items():assert sha(repo/path)==digest,path
    assert (directory/'exit-code').read_text().strip()=='0'
    jobs,_,conversations=probe.prepare(run)
    assert [probe.planned_job(j) for j in jobs]==plan['jobs']
    assert probe.preflight(jobs)==plan['preflight']
    events=records(directory/'responses.jsonl') if (directory/'responses.jsonl').exists() else []
    batches=records(directory/'batches.jsonl') if (directory/'batches.jsonl').exists() else []
    expected=[(j['qa_id'],a) for j in jobs if j['changed'] for a in ('control','candidate')]
    assert [(e['qa_id'],e['arm']) for e in events]==expected
    lookup={(j['qa_id'],a):j for j in jobs for a in ('control','candidate')}
    expected_batches=[(j['qa_id'],a,i) for j in jobs if j['changed'] for a in ('control','candidate')
                      for i in range(len(j['batches'][a]))]
    assert [(b['qa_id'],b['arm'],b['batch']) for b in batches]==expected_batches
    metrics,logged=metrics_from(log)
    assert logged==[{k:e[k] for k in ('qa_id','arm','status','calls','tokens')} for e in events]
    cursor=0
    for batch in batches:
        job=lookup[batch['qa_id'],batch['arm']]
        assert batch['conversation_id']==job['conversation_id']
        assert batch['indexes']==job['batches'][batch['arm']][batch['batch']]
        assert batch['calls'] in (0,1)
        chunk=metrics[cursor:cursor+batch['calls']]; cursor+=batch['calls']
        assert len(chunk)==batch['calls']
        assert sum(int(m['total_tokens']) for m in chunk)==batch['tokens']
        for metric in chunk:
            assert metric['kind']=='rerank' and metric['model']==plan['versions']['cross_encoder_model']
            assert metric['stage']=='probe.ce_bridge_summary'
            assert int(metric['input_count'])==len(batch['indexes'])
            assert metric['attempt']=='1' and metric['retries']=='0'
            assert metric.get('will_retry','false')=='false'
        if batch['status']=='complete':
            assert batch['calls']==1 and chunk[0]['status']=='ok'
            assert len(batch['scores'])==len(batch['indexes'])
            assert all(type(v) in (float,int) and math.isfinite(v) for v in batch['scores'])
        else:
            assert batch['status']=='failed' and batch['scores'] is None and batch['error_type']
    assert cursor==len(metrics)
    by_event={}
    for event in events:
        key=(event['qa_id'],event['arm']);job=lookup[key]
        group=[b for b in batches if (b['qa_id'],b['arm'])==key]
        reconstructed=[None]*len(job['coarse'])
        for batch in group:
            if batch['status']=='complete':
                for i,value in zip(batch['indexes'],batch['scores']):reconstructed[i]=value
        failed=[b['batch'] for b in group if b['status']=='failed']
        assert event['conversation_id']==job['conversation_id'] and event['query']==job['queries'][event['arm']]
        assert event['scores']==reconstructed and event['failed_batches']==failed
        assert event['status']==('failed' if failed else 'complete')
        assert event['batches']==len(group)
        assert event['calls']==sum(b['calls'] for b in group)
        assert event['tokens']==sum(b['tokens'] for b in group)
        by_event[key]=event
    # Check literal source suffixes independently of the transformer, including
    # markers appearing inside quotes. No split on source text is used.
    raw={};verified=0
    from inventory_bridge_provenance import closure
    from inventory_support_carriers import batches
    provenance, originals, _ = probe.load_provenance(run)
    dependencies={}; support_metadata={}
    with sqlite3.connect((run/'memory.db').resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        for dep in batches(db.execute("SELECT * FROM view_dependencies WHERE view_kind='amu'")):
            dependencies.setdefault(dep['view_id'],[]).append(dict(dep))
        for support in batches(db.execute('SELECT id,user_id,content,type,resolution_status,valid_to,state FROM amu')):
            support_metadata[support['id']]=dict(support)
        for source in probe.source_batches(db):
            source=dict(source);raw[source['user_id'],funnel.identity(source)]=source
    for job in jobs:
        assert job['queries']['control']==job['queries']['candidate']
        for candidate,old,new,info in zip(job['coarse'],job['docs'],job['documents']['candidate'],job['views']):
            for source in candidate['_packet_item'].get('sources',[]):
                original=raw[job['t']['user_id'],funnel.identity(source)]
                assert source['role']==original['role']
                assert funnel.norm(source.get('content','')) in funnel.norm(original['content'])
                verified+=1
            assert len(new.encode())<=len(old.encode())
            if new!=old:
                assert info['changed'] and candidate.get('type')!='rule' and not candidate.get('_user_rule')
                assert not candidate.get('_relation_path_ids') and not candidate.get('_coverage_proofs')
                unit=candidate['_packet_item'];body=unit['_evidence_body']
                assert not unit.get('is_constraint')
                suffix=answer_context.with_evidence('',unit['sources'])
                assert old==body+suffix and suffix and new.endswith(suffix)
                spec=provenance[job['conversation_id'],job['qa_id']][candidate['id']]
                declared=closure([candidate['id'],*candidate.get('_equivalent_ids',[])],dependencies)
                assert sorted(declared)==spec['declared_closure']
                body_bytes=body.encode();offset=0;pieces=[];seen_ids=set()
                for span in info['spans']:
                    mid=span['id'];metadata=support_metadata[mid]
                    assert mid in candidate['_bridge_ids'] and mid not in declared and mid not in seen_ids
                    seen_ids.add(mid)
                    assert metadata['user_id']==job['t']['user_id']
                    assert metadata['type']!='rule' and metadata['resolution_status']=='accepted'
                    assert not metadata['valid_to'] and not (json.loads(metadata['state'] or 'null') or {}).get('experience')
                    assert info['primary_bytes']<=span['start']<span['end']<=len(body_bytes)
                    assert offset<=span['start']
                    assert body_bytes[span['start']:span['end']]==metadata['content'].encode()
                    assert originals[mid] and span['source_count']==len(originals[mid])
                    for source in originals[mid]:
                        assert any(funnel.identity(v)==funnel.identity(source)
                                   and v.get('role')==source.get('role')
                                   and v.get('timestamp')==source.get('timestamp')
                                   and source['content'] and source['content'] in v.get('content','')
                                   for v in unit['sources'])
                    pieces.extend([body_bytes[offset:span['start']],b'Conversation excerpt:'])
                    offset=span['end']
                pieces.append(body_bytes[offset:])
                assert new==b''.join(pieces).decode()+suffix
                assert new.encode()[:info['primary_bytes']]==old.encode()[:info['primary_bytes']]
            else:assert not info['changed']
    fixed=[probe.selections(j,{a:by_event[j['qa_id'],a] for a in ('control','candidate')} if j['changed'] else {}) for j in jobs]
    rows=probe.coverage_rows(run,jobs,fixed,conversations,gold_csv)
    report=json.loads((directory/'result.json').read_text())
    assert report==dict(summary=probe.summarize(jobs,rows,events),rows=rows)
    stages={(r['conversation_id'],r['qa_id']):r for r in json.loads((run/'funnel/stage-transport.json').read_text())['rows']}
    for row in rows:
        old=stages[row['conversation_id'],row['qa_id']]['visible']
        for stage,key in (('fine','fine'),('main','listwise_input')):
            assert row['variants']['archived'][stage]['user']==[v['full'] for v in old[key]['turn_coverage']]
    provider_failures=[m for m in metrics if m['status']!='ok']
    arms={a:dict(calls=sum(e['calls'] for e in events if e['arm']==a),
                 reported_tokens=sum(e['tokens'] for e in events if e['arm']==a)) for a in ('control','candidate')}
    return dict(status='passed',questions=len(jobs),all_planned_batches_recorded=True,
        successful_scores_complete=not report['summary']['failed_arms'],
        failed_arms=report['summary']['failed_arms'],provider_calls=len(metrics),
        provider_failures=len(provider_failures),unknown_usage_requests=sum(m['status']!='ok' for m in metrics),
        reported_tokens=sum(int(m['total_tokens']) for m in metrics),arms=arms,
        verified_source_instances=verified,archived_stage_coverage_exact=len(rows),
        historical_control_matches={stage:sum(r['variants']['control'] is not None and
            r['variants']['control'][stage]['ids']==r['variants']['archived'][stage]['ids'] for r in rows)
            for stage in ('fine','main')},
        inputs={str(p.resolve()):sha(p) for p in (Path(__file__),directory/'plan.json',directory/'result.json',
            directory/'responses.jsonl',directory/'batches.jsonl',log,gold_csv)},
        limitation='Audit passing proves accounting and replay, not successful completion of failed arms. '
        'Counts with unavailable vectors are partial and cannot pass adoption. No hidden proxy-routing or HTTPS attestation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    args=parser.parse_args()
    with patch('socket.socket.connect',side_effect=AssertionError('Offline audit')):
        result=audit(args.probe,args.log,args.gold_csv)
    with (args.probe/'audit.json').open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'}))
