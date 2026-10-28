"""Paired witness retrieval with immutable catalogs and CE-ordered source cards.

Only the witness prompt's source order changes. All source IDs/text/roles and
option/quote validators remain identical. The diagnostic isolates witness
retrieval; it does not measure answer accuracy or production deadline effects.
Gold labels and historical answers never reach either provider.
"""
import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget,choice_witness,config,cross_encoder,llm,retrieval_queries
from evaluation_model_guard import check_runtime,check_provider_log
from probe_keyed_witness import prepare as prepare_original,records,fingerprint,reservation
from probe_context_label import exception_chain


def query_for(question,options):
    return ('Question: '+question+'\nAlternative answer premises to assess (not established facts):\n'
            +'\n'.join(retrieval_queries.compact_option(o) for o in options))


def cards_for(sources):
    return [dict(id=s['id'],role=s.get('declared') or s['role'],text=s['text']) for s in sources.values()]


def documents_for(cards):
    return [c['role']+': '+c['text'] for c in cards]


def ordered_request(original,cards,scores):
    if (len(cards)!=len(scores) or any(type(v) not in (int,float) or not math.isfinite(v) for v in scores)):
        raise ValueError('Every source needs exactly one score')
    order=sorted(range(len(cards)),key=lambda i:-scores[i])
    before=json.dumps(cards,ensure_ascii=False)
    after=json.dumps([cards[i] for i in order],ensure_ascii=False)
    marker='<sources>\n'+before+'\n</sources>'
    if original['prompt'].count(marker)!=1:
        raise ValueError('Witness source block did not reproduce')
    result=copy.deepcopy(original)
    result['prompt']=result['prompt'].replace(marker,'<sources>\n'+after+'\n</sources>',1)
    if len(result['prompt'].encode())!=len(original['prompt'].encode()):
        raise ValueError('Source ordering changed prompt byte count')
    return result,[cards[i]['id'] for i in order]


def batches_for(query,cards):
    batches=[]
    current=[]
    documents=documents_for(cards)
    for i,doc in enumerate(documents):
        if len(doc.encode())>config.CE_MAX_DOCUMENT_BYTES:
            raise ValueError('A complete source exceeds the CE document bound')
        trial=current+[i]
        if (len(trial)>config.CE_BATCH_SIZE or cross_encoder.request_size(query,
                [documents[j] for j in trial])>config.CE_MAX_REQUEST_BYTES):
            if not current:
                raise ValueError('Source cannot fit CE request')
            batches.append(current)
            current=[i]
        else:
            current=trial
    if current:
        batches.append(current)
    if len(query.encode())>config.CE_MAX_DOCUMENT_BYTES:
        raise ValueError('Query exceeds CE bound')
    return batches


def decode(payload,job):
    detail={}
    failure=None
    try:
        semantic=choice_witness.validate(payload,job['options'],job['sources'],detail)
        final=choice_witness.supplement_first_party(job['options'],job['sources'],semantic)
    except Exception as exc:
        failure=exception_chain(exc)
        semantic={}
        final=choice_witness.build(job['options'],job['sources'])
    if job['coverage_witnesses']:
        final=choice_witness.supplement_from_coverage(job['options'],job['sources'],final,job['packet'])
    return dict(semantic=semantic,final_hints=final,validation=detail,
                validation_failure=failure,status='fallback' if failure else 'partial' if detail.get('rejected_witnesses') else 'ok')


async def prepare(run):
    jobs,settings=await prepare_original(run)
    for key in ('CE_BATCH_SIZE','CE_MAX_DOCUMENT_BYTES','CE_MAX_REQUEST_BYTES','CE_TIMEOUT_SECONDS'):
        if getattr(config,key)!=settings[key]:
            raise ValueError('Runtime CE limits differ from archived baseline')
    traces={t['search_id']:t for t in records(run/'server-search-debug.jsonl')}
    results={r['qa_id']:r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
    for job in jobs:
        r=results[job['qa_id']];t=traces[r['search_id']]
        request=job.pop('requests')['control'];cards=cards_for(job['sources'])
        query=query_for(t['query'],job['options']);batches=batches_for(query,cards)
        job.update(request=request,cards=cards,ce_query=query,ce_batches=batches,packet=t['returned'],
                   coverage_witnesses=settings['CHOICE_COVERAGE_WITNESSES'])
        old=next(c for c in r['answer_calls'] if c['stage']=='eval.choice_witness')
        reproduced=decode(old.get('response',{}),job)
        if reproduced['final_hints']!=r['answer_witness_prefill']:
            raise ValueError('Historical final witness hints did not reproduce')
        # Identity permutation proves exact source-block reconstruction.
        identity,order=ordered_request(request,cards,[0.0]*len(cards))
        if identity!=request or order!=list(job['sources']):
            raise ValueError('Identity order differs')
    return jobs,settings


async def main(args):
    check_runtime()
    if config.CE_MODEL!='qwen3-rerank':
        raise ValueError('Expected fixed qwen3-rerank')
    suffixes=['.json','.plan.json','.live.jsonl','.ce.jsonl','.provider.log']
    if any(args.output.with_suffix(s).exists() for s in suffixes):
        raise FileExistsError('Output already exists')
    jobs,settings=await prepare(args.run)
    schedule=[dict(qa_id=j['qa_id'],order=['control','treatment'] if i%2==0 else ['treatment','control'])
              for i,j in enumerate(jobs)]
    plan=dict(run=str(args.run.resolve()),model=config.LLM_MODEL,ce_model=config.CE_MODEL,questions=len(jobs),
        live=args.live,expected_ce_calls=sum(len(j['ce_batches']) for j in jobs) if args.live else 0,
        expected_llm_calls=2*len(jobs) if args.live else 0,concurrency=2,schedule=schedule,jobs=jobs,
        settings=settings,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes={n:hashlib.sha256((args.run/n).read_bytes()).hexdigest() for n in
            ['server-search-debug.jsonl','completed/data/results/personamem-v2-32k.jsonl']},limitation=__doc__)
    args.output.with_suffix('.plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    if not args.live:
        args.output.write_text(json.dumps(dict(questions=len(jobs),reconstructed_hints=len(jobs),
            ce_batches=sum(len(j['ce_batches']) for j in jobs),source_counts=[len(j['sources']) for j in jobs],
            provider_calls=0),indent=2)+'\n')
        print(json.dumps(dict(prepared=len(jobs),ce_batches=sum(len(j['ce_batches']) for j in jobs))))
        return
    if config.FAKE:
        raise ValueError('Real providers required')
    logger=logging.getLogger('aml.metrics');logger.setLevel(logging.INFO)
    handler=logging.FileHandler(args.output.with_suffix('.provider.log'));handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    rows=[];ce_rows=[];semaphore=asyncio.Semaphore(2)
    def save(row,suffix):
        with args.output.with_suffix(suffix).open('a') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    async def paired(job,scheduled):
        async with semaphore:
            scores=[None]*len(job['cards']);documents=documents_for(job['cards']);ce_failed=False
            for bi,indices in enumerate(job['ce_batches']):
                docs=[documents[i] for i in indices]
                bound=cross_encoder.request_size(job['ce_query'],docs)
                row=dict(qa_id=job['qa_id'],batch=bi,indices=indices,query_sha256=hashlib.sha256(job['ce_query'].encode()).hexdigest(),
                         document_sha256=[hashlib.sha256(d.encode()).hexdigest() for d in docs],reservation=bound)
                start=time.monotonic()
                with budget.scope(seconds=config.CE_TIMEOUT_SECONDS+1,calls=1,tokens=bound) as limits:
                    try:
                        values=await cross_encoder.rerank(job['ce_query'],docs,
                            stage=f'diagnostic.witness_order.ce.{job["qa_id"]}.{bi}')
                        row.update(scores=values,status='ok')
                        for i,v in zip(indices,values):scores[i]=v
                    except Exception as exc:
                        ce_failed=True
                        row.update(status='error',exception_chain=exception_chain(exc))
                row.update(calls=limits.calls,tokens=limits.tokens,seconds=time.monotonic()-start)
                ce_rows.append(row);save(row,'.ce.jsonl')
            if ce_failed:
                treatment=job['request'];order=list(job['sources'])
            else:
                treatment,order=ordered_request(job['request'],job['cards'],scores)
            for arm in scheduled['order']:
                request=copy.deepcopy(treatment if arm=='treatment' else job['request'])
                bound=reservation(request);start=time.monotonic()
                row=dict(qa_id=job['qa_id'],arm=arm,request_sha256=fingerprint(request),
                    prompt_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest(),reservation=bound,
                    source_order=order if arm=='treatment' else list(job['sources']),ce_fallback=ce_failed)
                with budget.scope(seconds=choice_witness.TIMEOUT_SECONDS+1,calls=1,tokens=bound) as limits:
                    try:
                        async with asyncio.timeout(choice_witness.TIMEOUT_SECONDS):
                            payload=await llm.complete_json(**request,stage=f'diagnostic.witness_order.{arm}.{job["qa_id"]}')
                        row.update(response=payload,**decode(payload,job))
                    except Exception as exc:
                        row.update(status='provider_error',exception_chain=exception_chain(exc),**{
                            k:v for k,v in decode({},job).items() if k!='status'})
                row.update(calls=limits.calls,tokens=limits.tokens,seconds=time.monotonic()-start)
                rows.append(row);save(row,'.live.jsonl')
                print(json.dumps(dict(completed=len(rows),planned=plan['expected_llm_calls'],qa_id=job['qa_id'],
                    arm=arm,status=row['status'],ce_fallback=ce_failed,
                    semantic_quotes=sum(map(len,row['semantic'].values())))),flush=True)
    try:
        await asyncio.gather(*(paired(j,s) for j,s in zip(jobs,schedule)))
    finally:
        logger.removeHandler(handler);handler.close()
    summary=dict(questions=len(jobs),ce_calls=sum(r['calls'] for r in ce_rows),ce_tokens=sum(r['tokens'] for r in ce_rows),
        ce_statuses=dict(Counter(r['status'] for r in ce_rows)),
        llm_dispatches=check_provider_log(args.output.with_suffix('.provider.log').read_text(),'source order'),
        arms={arm:dict(calls=sum(r['calls'] for r in rows if r['arm']==arm),
            tokens=sum(r['tokens'] for r in rows if r['arm']==arm),
            statuses=dict(Counter(r['status'] for r in rows if r['arm']==arm)),
            semantic_quotes=sum(len(ws) for r in rows if r['arm']==arm for ws in r['semantic'].values()))
            for arm in ['control','treatment']})
    args.output.write_text(json.dumps(dict(summary=summary,live=rows,ce=ce_rows,limitation=__doc__),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if any(r['status']=='error' for r in ce_rows) or any(r['status']=='provider_error' for r in rows):
        raise RuntimeError('Original failed requests retained')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--live',action='store_true')
    args=p.parse_args()
    try:
        asyncio.run(main(args))
    except Exception as exc:
        print(json.dumps(dict(error=type(exc).__name__,exception_chain=exception_chain(exc))),flush=True)
        sys.exit(1)
