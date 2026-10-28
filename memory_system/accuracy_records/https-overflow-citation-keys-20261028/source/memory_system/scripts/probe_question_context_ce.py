"""Real question-only CE ordering for one residual complete-source context seat.

Preserve the actual three-seat lexical prefix. Score only suppliers which fit
the remaining original byte/item caps, with one bounded request per question.
Documents are the exact complete visible source units that would be delivered.
No generated summary, option premise, gold, source expansion or fact promotion.
"""
import argparse
import ast
import asyncio
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice,budget,config,cross_encoder,evidence_packet,search_pipeline
from app.fine_context_lexical import propose as lexical
from app.fine_context_reserve import propose as reserve
from evaluation_model_guard import check_runtime,check_versions
import audit_https_funnel as audit


def records(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def prepare(t, result):
    if result['search_id'] != t['search_id'] or result['packet_hash'] != t['packet_hash']:
        raise ValueError('Result/trace identity mismatch')
    ctx=t['cascade']['conversation_context']
    base=[dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
    assert evidence_packet.digest(base)==ctx['base_packet_hash']
    pool={c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
          if c.get('view_status')!='stale' and '_packet_item' in c}
    fine=[pool[mid] for mid in t['cascade']['fine']['selected_ids'] if mid in pool and mid not in t['selection']['selected_ids']]
    kw=dict(user_id=t['user_id'],top_k=t['top_k'],byte_budget=t['coverage_manifest']['token_budget'])
    prefix,trace=lexical(base,fine,**kw,reserve_bytes=5120,query=t['query'],options=t['options'])
    assert [dict(p,packet_hash=evidence_packet.digest(prefix)) for p in prefix]==t['returned']
    remaining=5120-trace['added_bytes'];eligible=[];documents=[];omitted=[]
    for c in fine:
        packet,trial=reserve(prefix,[c],**kw,reserve_bytes=remaining,limit=1)
        if not trial['appended_ids']:continue
        text=packet[-1]['content']
        if len(text.encode())>config.CE_MAX_DOCUMENT_BYTES:
            omitted.append(dict(candidate_id=c['id'],reason='document_bytes'));continue
        eligible.append(c);documents.append(text)
    bound=cross_encoder.request_size(t['query'],documents)
    # Final accounting is added to the HTTP response after the debug trace.
    accounting=result['coverage_manifest']
    headroom=config.SEARCH_MAX_TOKENS-accounting['provider_tokens']
    time_headroom=config.SEARCH_DEADLINE_SECONDS-result['search_seconds']
    allowed=(len(eligible)<=config.CE_BATCH_SIZE and bound<=config.CE_MAX_REQUEST_BYTES and
             len(t['query'].encode())<=config.CE_MAX_DOCUMENT_BYTES and
             bound<=headroom and accounting['provider_calls']<config.SEARCH_MAX_CALLS and
             time_headroom>=config.CE_TIMEOUT_SECONDS)
    if not allowed:
        omitted.extend(dict(candidate_id=c['id'],reason='single_request_or_recorded_budget') for c in eligible)
        eligible=[];documents=[];bound=0
    return dict(prefix=prefix,fine=fine,eligible=eligible,documents=documents,bound=bound,
                headroom=headroom,time_headroom=time_headroom,remaining=remaining,omitted=omitted,kwargs=kw)


async def main(a):
    if a.execute and a.replay:
        raise ValueError('Replay must not dispatch provider calls')
    check_runtime()
    if config.FAKE:raise ValueError('Require real CE')
    if (a.run/'exit-code').read_text().strip()!='0':raise ValueError('Incomplete input run')
    plan_path=a.output.with_suffix('.plan.json');live_path=a.output.with_suffix('.live.jsonl')
    if any(p.exists() for p in (a.output,plan_path,live_path)):raise FileExistsError(a.output)
    rs={r['search_id']:r for r in records(a.run/'completed/data/results/personamem-v2-32k.jsonl')}
    jobs=[]
    for t in records(a.run/'server-search-debug.jsonl'):
        check_versions(t['versions'],'source trace')
        with patch.multiple(config,**t['versions']['settings']),patch('socket.socket.connect',side_effect=AssertionError('Prepare offline')):
            if config.PACKET_SOURCE_QUERY_CONTEXT or config.PACKET_FINE_CONTEXT_FILL:raise ValueError('Require original lexical control')
            jobs.append(dict(t=t,qa_id=rs[t['search_id']]['qa_id'],**prepare(t,rs[t['search_id']])))
    plan=dict(source_run=str(a.run),source_trace_sha256=hashlib.sha256((a.run/'server-search-debug.jsonl').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),llm_model=config.LLM_MODEL,ce_model=config.CE_MODEL,
        source_results_sha256=hashlib.sha256((a.run/'completed/data/results/personamem-v2-32k.jsonl').read_bytes()).hexdigest(),
        expected_calls=sum(bool(j['eligible']) for j in jobs),limitation=__doc__,
        jobs=[dict(qa_id=j['qa_id'],query=j['t']['query'],candidate_ids=[c['id'] for c in j['eligible']],
                   documents=j['documents'],bound=j['bound'],recorded_token_headroom=j['headroom'],
                   recorded_time_headroom=j['time_headroom'],
                   remaining_bytes=j['remaining'],omitted=j['omitted']) for j in jobs])
    plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(questions=len(jobs),expected_calls=plan['expected_calls'],request_bound=sum(j['bound'] for j in jobs))),flush=True)
    replay={}
    if a.replay:
        prior=json.loads(a.replay.with_suffix('.plan.json').read_text())
        for key in ('source_run','source_trace_sha256','source_results_sha256','llm_model','ce_model','expected_calls','jobs'):
            if prior[key]!=plan[key]:raise ValueError('Replay input changed: '+key)
        prior_rows=records(a.replay.with_suffix('.live.jsonl'))
        replay={r['qa_id']:r for r in prior_rows}
        expected={j['qa_id'] for j in jobs if j['eligible']}
        if len(replay)!=len(prior_rows) or set(replay)!=expected:
            raise ValueError('Replay has missing, duplicate or unexpected questions')
        for j in jobs:
            if not j['eligible']:continue
            row=replay[j['qa_id']]
            if (row.get('error_type') or row['candidate_ids']!=[c['id'] for c in j['eligible']] or
                    row['request_bound']!=j['bound'] or row['calls']!=1 or
                    len(row['scores'])!=len(j['eligible']) or
                    any(type(v) not in (int,float) or not math.isfinite(v) for v in row['scores'])):
                raise ValueError('Invalid replay scores or identity')
    if not a.execute and not a.replay:return
    scored=[];live=[]
    for j in jobs:
        values=[]
        if j['eligible']:
            row=dict(qa_id=j['qa_id'],candidate_ids=[c['id'] for c in j['eligible']],request_bound=j['bound'])
            if a.replay:
                row=copy.deepcopy(replay[j['qa_id']]);values=row['scores']
            else:
                with patch.multiple(config,**j['t']['versions']['settings']),budget.scope(seconds=30,calls=1,tokens=j['bound']) as limits:
                    check_runtime()
                    try:values=await cross_encoder.rerank(j['t']['query'],j['documents'],stage='diagnostic.question_context_ce');row['scores']=values
                    except Exception as exc:row.update(error_type=type(exc).__name__,error_detail=str(exc)[:500])
                row.update(calls=limits.calls,tokens=limits.tokens)
            live.append(row)
            with live_path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ('candidate_ids','scores','error_detail')}),flush=True)
            if row.get('error_type'):continue
        ordered=[c for _,c in sorted(zip(values,j['eligible']),key=lambda x:-x[0])]
        packet,trace=reserve(j['prefix'],ordered,**j['kwargs'],reserve_bytes=j['remaining'],limit=1)
        assert packet[:len(j['prefix'])]==j['prefix']
        scored.append(dict(job=j,packet=packet,trace=trace,scores=dict(zip([c['id'] for c in j['eligible']],values))))
    # Gold is loaded only after every request, ranking and proposed packet is fixed.
    csv.field_size_limit(32*1024*1024)
    with a.gold_csv.open() as f:gold=list(csv.DictReader(f))
    raw={}
    for event in records(a.run/'server-memory-debug.jsonl'):
        raw.setdefault(event['user_id'],{}).update({audit.identity(s):s for s in event['source_messages']})
    rows=[]
    for result in scored:
        j=result['job'];snippet=gold[int(j['qa_id'])]['related_conversation_snippet']
        try:snippet=json.loads(snippet)
        except ValueError:snippet=ast.literal_eval(snippet)
        coverage={};catalog_bytes={}
        for mode,packet in (('baseline',j['prefix']),('question_ce',result['packet'])):
            cards=[dict(s,text=s.get('content',''),id=f'{i}:{k}') for i,p in enumerate(packet) for k,s in enumerate(p.get('sources',[]))]
            coverage[mode]={role:[v['full'] for v in audit.turn_coverage([audit.norm(s['content']) for s in snippet if s['role']==role],cards,raw[j['t']['user_id']],role=role)] for role in ('user','assistant')}
            catalogs=answer_choice.build_catalog([dict(p,packet_hash=evidence_packet.digest(packet)) for p in packet])
            catalog_bytes[mode]=sum(len(c['text'].encode()) for cat in catalogs for c in cat.values())
        rows.append(dict(qa_id=j['qa_id'],coverage=coverage,catalog_bytes=catalog_bytes,proposal=result['trace'],scores=result['scores']))
    summary=dict(questions=len(rows),expected_questions=len(jobs),calls=sum(r['calls'] for r in live),tokens=sum(r['tokens'] for r in live),
        replay_from=str(a.replay) if a.replay else None,new_provider_calls=0 if a.replay else sum(r['calls'] for r in live),
        errors=sum(bool(r.get('error_type')) for r in live),
        reference_questions={m:sum(any(r['coverage'][m]['user']) for r in rows) for m in ('baseline','question_ce')},
        catalog_bytes={m:sum(r['catalog_bytes'][m] for r in rows) for m in ('baseline','question_ce')},
        appended_questions=sum(bool(r['proposal']['appended_ids']) for r in rows),
        gained=[r['qa_id'] for r in rows if any(r['coverage']['question_ce']['user']) and not any(r['coverage']['baseline']['user'])],
        lost=[r['qa_id'] for r in rows if any(r['coverage']['baseline']['user']) and not any(r['coverage']['question_ce']['user'])],
        turn_regressions=[r['qa_id'] for r in rows if any(x and not y for role in ('user','assistant') for x,y in zip(r['coverage']['baseline'][role],r['coverage']['question_ce'][role]))])
    a.output.write_text(json.dumps(dict(summary=summary,rows=rows,plan=str(plan_path),live=str(live_path),limitation=__doc__),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    if summary['errors'] or len(rows)!=len(jobs):raise RuntimeError('Incomplete diagnostic')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true',help='Execute the bounded real CE diagnostic after preparing every request')
    p.add_argument('--replay',type=Path,help='Existing output stem: verify exact inputs and reuse every saved score offline')
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    args=p.parse_args()
    if args.replay:
        with patch('socket.socket.connect',side_effect=AssertionError('Replay offline')):
            asyncio.run(main(args))
    else:
        asyncio.run(main(args))
