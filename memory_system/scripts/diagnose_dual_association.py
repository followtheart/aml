"""Four-arm candidate ablation with frozen write views, queries and CE scores.

The current original-query pipeline remains in all arms. New query generation
and embeddings are shared between C01/C11; each distinct original document is
CE-scored once per question. Ranking simulates identical conservative byte/call
budgets and debits query preparation in C01/C11. No reviewer/answer model runs.
"""
import argparse
import ast
import asyncio
import copy
import csv
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import (budget,cascade_rerank,config,cross_encoder,embeddings,llm,
                 query_associations as qa,run_metadata,schemas,search_pipeline as search,store)
from diagnose_association_views import cohort,identity,records,write,AdmissionReached

ARMS={'C00':(False,False),'C10':(True,False),'C01':(False,True),'C11':(True,True)}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def view_identity(st):
    return digest('\n'.join(sorted(repr(tuple(r)) for r in st.conn.execute('SELECT * FROM association_views'))))


def build(st,trace,vectors,query_vectors,arm):
    write_enabled,query_enabled=ARMS[arm]
    plan=copy.deepcopy(trace['plan'])
    plan.update(_anchor=trace['anchor_time'],_include_history=trace['include_history'],
        _include_sensitive=trace['include_sensitive'],_query_specs=trace['query_specs'],
        _used_queries=[s['text'] for s in trace['query_specs']],_fusion_query_vectors=vectors.tolist(),
        _routes=copy.deepcopy(trace['routes']),
        _coverage_requirements=[{k:r[k] for k in ('id','text','origin')}
            for r in trace['coverage_manifest']['coverage']['requirements']])
    req=schemas.SearchRequest(user_id=trace['user_id'],query=trace['query'],options=trace['options'],
        top_k=trace['top_k'],reference_time=trace['reference_time'],include_sensitive=trace['include_sensitive'],
        evidence_token_budget=trace['evidence_token_budget'])
    routes=[copy.deepcopy(r['candidates']) for r in trace['routes']]
    ids=list({c['id'] for route in routes for c in route})
    live={r['id']:r for r in st.get_amus_by_ids(ids,True,True)}
    for route in routes:
        for row in route: row['embedding']=live[row['id']].get('embedding')
    proposed,diagnostics=[],{}
    if write_enabled or query_enabled:
        proposed,diagnostics=qa.recall(st,req.user_id,vectors,query_vectors if query_enabled else [],
            write_enabled=write_enabled,include_history=trace['include_history'],include_sensitive=trace['include_sensitive'])
        search._route(plan,routes,'association',proposed,'bounded association hypotheses')
        # Match the actual insertion point after vector, FTS and original-source
        # lanes; do not change the existing graph/scene expansion seed lanes.
        routes.insert(3,routes.pop())
        plan['_routes'].insert(3,plan['_routes'].pop())
    fused=search._fuse_candidates(st,req,plan,routes)
    candidates=search._foresight_filter(search._prepare_candidates(st,req,plan,fused),plan)
    return req,plan,candidates,dict(associated_ids=[r['id'] for r in proposed],diagnostics=diagnostics)


async def shared_scores(prepared):
    query=None;documents=set()
    for req,plan,candidates,_ in prepared.values():
        current=cascade_rerank.query_text(req,plan)
        if query is not None and current!=query: raise ValueError('CE query changed across arms')
        query=current
        coarse,_=cascade_rerank.coarse_shortlist([c for c in candidates if not cascade_rerank.protected(c)],
            plan['_coverage_requirements'])
        for c in coarse:
            text=c.get('_rank_text',c['content'])
            if (len(text.encode())<=config.CE_MAX_DOCUMENT_BYTES and len(query.encode())<=config.CE_MAX_DOCUMENT_BYTES
                    and cross_encoder.request_size(query,[text])<=config.CE_MAX_REQUEST_BYTES):
                documents.add(text)
    batches=[];batch=[];size=0
    for document in sorted(documents,key=digest):
        cost=cross_encoder.request_size(query,[document])
        if batch and (len(batch)>=config.CE_BATCH_SIZE or size+cost>config.CE_MAX_REQUEST_BYTES):
            batches.append(batch);batch=[];size=0
        batch.append(document);size+=cost
    if batch: batches.append(batch)
    scores={};errors=[]
    with budget.scope(seconds=180,calls=max(20,len(batches)+1),tokens=2000000) as limits:
        for batch in batches:
            try:
                values=await cross_encoder.rerank(query,batch,stage='diagnostic.shared_ce')
                scores.update(zip(batch,values))
            except Exception as exc:
                errors.append(dict(error=type(exc).__name__,document_hashes=[digest(d) for d in batch]))
        usage=dict(calls=limits.calls,tokens=limits.tokens)
    return query,scores,errors,usage


async def rank(st,prepared,ce_query,scores,preparation):
    req,plan,candidates,details=prepared
    requests={};admission={}
    real_request=cascade_rerank.listwise_request

    def capture(req,plan,rows,**kwargs):
        result=real_request(req,plan,rows,**kwargs)
        requests[digest(result['prompt'])]=[r['id'] for r in rows]
        return result

    async def stop(prompt,**kwargs):
        if kwargs['stage']!='search.listwise': raise AssertionError('Unexpected main stage')
        admission.update(ids=requests[digest(prompt)],prompt=prompt)
        raise AdmissionReached()

    async def frozen_ce(query,documents,**kwargs):
        if query!=ce_query or any(d not in scores for d in documents):
            raise ValueError('Unscored CE document')
        size=cross_encoder.request_size(query,documents)
        limits=budget.current.get()
        limits.reserve_tokens(size)
        try:
            limits.before_call()
            limits.tokens+=size  # same conservative byte proxy for every arm
            return [scores[d] for d in documents]
        finally: limits.release_tokens(size)

    with budget.scope(seconds=max(.01,config.SEARCH_DEADLINE_SECONDS-preparation.get('seconds',0)),
                      calls=config.SEARCH_MAX_CALLS,tokens=config.SEARCH_MAX_TOKENS) as limits:
        limits.calls=preparation.get('calls',0);limits.tokens=preparation.get('tokens',0)
        with patch.object(cascade_rerank,'listwise_request',capture),patch.object(llm,'complete_json',stop), \
             patch.object(cross_encoder,'rerank',frozen_ce):
            try: await cascade_rerank.rank(req,plan,candidates)
            except AdmissionReached: pass
        cascade=plan['_cascade']
        stages=dict(coarse=cascade['coarse'].get('selected_ids',[]),fine=cascade['fine'].get('selected_ids',[]),
                    main_input=admission.get('ids',[]))
        needed=set().union(*map(set,stages.values()))
        by_id={r['id']:r for r in candidates if r['id'] in needed}
        sources=st.sources_for_amus(list(by_id))
        return dict(**details,stages=stages,cascade=cascade,simulated_calls=limits.calls,simulated_tokens=limits.tokens,
            main_prompt=admission.get('prompt'),
            candidates={mid:dict(sources=sources[mid],rank_text=c.get('_rank_text',c['content'])) for mid,c in by_id.items()})


async def run(args):
    if config.FAKE: raise ValueError('Real providers required')
    jobs=cohort(args)
    for key,value in jobs[0][1]['versions']['settings'].items(): setattr(config,key,value)
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    source=args.index_run/'memory.db'
    index_manifest=json.loads((args.index_run/'manifest.json').read_text())
    if not (args.index_run/'index-complete.json').exists(): raise ValueError('Complete write index required')
    with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(args.output/'memory.db') as dst:
        src.backup(dst)
    st=store.Store(str(args.output/'memory.db'))
    original=identity(st);views=view_identity(st)
    if original!=index_manifest['identity']: raise ValueError('Original index identity mismatch')
    run_metadata.versions.cache_clear();versions=run_metadata.versions()
    scripts=[Path(__file__),ROOT/'scripts/diagnose_association_views.py']
    files=dict(versions['file_hashes'])
    files.update({str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts})
    for name,sha in files.items():
        path=ROOT.parent/name;destination=args.output/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
    write(args.output/'manifest.json',dict(baseline=str(args.baseline.resolve()),index_run=str(args.index_run.resolve()),
        questions=len(jobs),arms=ARMS,versions=versions,files=files,original_identity=original,view_identity=views,
        gate='C11 main-input visible reference coverage exceeds every other arm, no C00 reference losses, no generation/CE failures.',
        scope='Frozen historical planning/recall; shared query generation/vectors and fixed original-document CE scores. Ranking budget uses input-byte bounds; extra query preparation charged to C01/C11. Not HTTPS or answer accuracy.'))
    try:
        with (args.output/'results.jsonl').open('x') as output,(args.output/'queries.jsonl').open('x') as queries, \
             (args.output/'ce-scores.jsonl').open('x') as ce_file:
            for index,(result,trace) in enumerate(jobs):
                base_vectors=await embeddings.embed([s['text'] for s in trace['query_specs']],stage='diagnostic.shared_base_queries')
                req=schemas.SearchRequest(user_id=trace['user_id'],query=trace['query'],options=trace['options'])
                started=time.monotonic()
                with budget.scope(seconds=config.SEARCH_DEADLINE_SECONDS,calls=config.SEARCH_MAX_CALLS,tokens=config.SEARCH_MAX_TOKENS) as limits:
                    query_views,query_vectors,diagnostics=await qa.prepare(req)
                    preparation=dict(calls=limits.calls,tokens=limits.tokens,seconds=time.monotonic()-started)
                queries.write(json.dumps(dict(qa_id=str(result['qa_id']),query=trace['query'],options=trace['options'],
                    base_specs=trace['query_specs'],base_vectors=base_vectors.tolist(),query_views=query_views,
                    query_vectors=query_vectors.tolist(),diagnostics=diagnostics,preparation=preparation),ensure_ascii=False)+'\n');queries.flush()
                with st.snapshot(trace['user_id']) as snap:
                    prepared={arm:build(snap,trace,base_vectors,query_vectors,arm) for arm in ARMS}
                    ce_query,scores,errors,usage=await shared_scores(prepared)
                    ce_file.write(json.dumps(dict(qa_id=str(result['qa_id']),query=ce_query,
                        documents=[dict(text=d,score=s,sha256=digest(d)) for d,s in scores.items()],errors=errors,usage=usage),ensure_ascii=False)+'\n');ce_file.flush()
                    order=list(ARMS);order=order[index%4:]+order[:index%4]
                    arms={}
                    for arm in order:
                        arms[arm]=await rank(snap,prepared[arm],ce_query,scores,preparation if ARMS[arm][1] else {})
                row=dict(qa_id=str(result['qa_id']),conversation_id=result['conversation_id'],user_id=result['user_id'],
                    query_diagnostics=diagnostics,preparation=preparation,ce_errors=errors,arms=arms,
                    historical_coarse_ids=trace['cascade']['coarse']['selected_ids'])
                output.write(json.dumps(row,ensure_ascii=False)+'\n');output.flush()
                print(json.dumps(dict(completed=index+1,questions=len(jobs),qa_id=row['qa_id'],query_views=len(query_views),
                    generation_status=diagnostics['status'],main_counts={arm:len(data['stages']['main_input']) for arm,data in arms.items()})),flush=True)
        assert identity(st)==original and view_identity(st)==views
        for name,sha in files.items():
            if hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()!=sha: raise ValueError('Frozen code changed: '+name)
        write(args.output/'complete.json',dict(questions=len(jobs),original_state_unchanged=True,write_views_unchanged=True))
    finally: st.conn.close()


def analyze(args):
    import audit_https_funnel as audit
    complete=json.loads((args.output/'complete.json').read_text())
    rows=list(records(args.output/'results.jsonl'))
    if len(rows)!=complete['questions'] or len({(r['conversation_id'],r['qa_id']) for r in rows})!=len(rows):
        raise ValueError('Incomplete or duplicate result inventory')
    csv.field_size_limit(32*1024*1024)
    with (ROOT/'data/raw/personamem-v2.csv').open() as stream: gold=list(csv.DictReader(stream))
    report=[]
    for row in rows:
        raw=gold[int(row['qa_id'])]['related_conversation_snippet']
        try: snippet=json.loads(raw)
        except ValueError: snippet=ast.literal_eval(raw)
        turns=[audit.norm(s['content']) for s in snippet if s['role']=='user']
        scores={}
        for arm,data in row['arms'].items():
            scores[arm]={}
            for stage,ids in data['stages'].items():
                carriers={mid:[audit.norm(s['content']) for s in data['candidates'][mid]['sources'] if s['role']=='user'] for mid in ids}
                visible=[t for t in turns if t and any(t in audit.norm(data['candidates'][mid]['rank_text'])
                    and any(t in s for s in carriers[mid]) for mid in ids)]
                scores[arm][stage]=dict(carrier=any(t and any(t in s for texts in carriers.values() for s in texts) for t in turns),
                    visible=bool(visible),visible_reference_turns=len(visible))
        report.append(dict(qa_id=row['qa_id'],scores=scores))
    summary={arm:{stage:{kind:sum(r['scores'][arm][stage][kind] for r in report)
        for kind in ('carrier','visible','visible_reference_turns')} for stage in ('coarse','fine','main_input')} for arm in ARMS}
    paired={}
    for arm in ('C10','C01','C11'):
        paired[arm]=dict(gains=[r['qa_id'] for r in report if r['scores'][arm]['main_input']['visible'] and not r['scores']['C00']['main_input']['visible']],
            losses=[r['qa_id'] for r in report if r['scores']['C00']['main_input']['visible'] and not r['scores'][arm]['main_input']['visible']])
    generation_failures=sum(r['query_diagnostics']['status']!='ok' for r in rows)
    ce_failures=sum(len(r['ce_errors']) for r in rows)
    gate=(summary['C11']['main_input']['visible']>max(summary[a]['main_input']['visible'] for a in ('C00','C10','C01'))
          and not paired['C11']['losses'] and not generation_failures and not ce_failures)
    result=dict(questions=len(rows),summary=summary,paired=paired,generation_failures=generation_failures,ce_failures=ce_failures,
        gate_passed=gate,rows=report,caveat='Reference transport only. No main review, answering, HTTPS accuracy or latency measured.')
    write(args.output/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--index-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('run','analyze'),required=True)
    args=parser.parse_args();logging.basicConfig(level=logging.INFO)
    if args.phase=='analyze': analyze(args)
    else: asyncio.run(run(args))


if __name__=='__main__': main()
