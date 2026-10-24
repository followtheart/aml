"""Frozen full-history LongMemEval four-arm experiment with crossed readers."""
import argparse
import asyncio
import json
import logging
import random
import time
from pathlib import Path
from pydantic import Field
from . import guarded_benchmark as base
from .longmemeval_data import prepare, SEED, STRATA
from .real_benchmark import save, append, CONTEXT, LegacyUsage
from .benchmark_metrics import file_hash, common_packet, quote_metrics
from memory_connection.app.providers import HTTPProvider
from memory_connection.app.store import Store
from memory_connection.app.service import MemoryService
from memory_connection.app.schemas import AddRequest, SearchRequest, Strict
from memory_connection.app.answering import answer
from memory_connection.app.expansion import clean_expansion

ROOT='memory_connection/runs/optimization-v5-longmemeval-20261024'


class Judgement(Strict):
    correct: bool
    reason: str=Field(max_length=1200)


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def settings():
    return base.settings_from_file('memory_connection/.env').model_copy(update={
        'search_seconds':180,'recall_seconds':45,'rerank_seconds':30,'verify_seconds':100,
        'add_seconds':300,'batch_bytes':96000,'rerank_document_bytes':18000,
        'rerank_request_bytes':180000,'rerank_batch_size':12,
        'exclusive_bridge_slots':0,'complementary_slots':2})


async def prepare_query(root,conv,provider):
    folder=root/'query-preparations';folder.mkdir(exist_ok=True);path=folder/f"{conv['conversation_id']}-0.json"
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    qa=conv['qa'][0];query={'query':qa['question'],'options':[],'reference_time':qa['question_date'],'as_of':None}
    context={'arm':'query_shared','phase':'query_prepare','conversation_id':conv['conversation_id'],'qa_id':'0'}
    CONTEXT.set({**context,'component':'original_embed'});started=time.perf_counter()
    original=(await base.preparation_call(provider,lambda:provider.embed([qa['question']])))[0]
    original_seconds=time.perf_counter()-started
    CONTEXT.set({**context,'component':'query_expansion'});started=time.perf_counter()
    expansion=await base.preparation_call(provider,lambda:provider.expand_query(query))
    views=clean_expansion(expansion,provider.settings.query_views,original=qa['question'])
    CONTEXT.set({**context,'component':'derived_embed'})
    vectors=await base.preparation_call(provider,lambda:provider.embed([v.text for v in views])) if views else []
    result={'query':query,'expansion':expansion,'vectors':{qa['question']:original},
            'original_seconds':original_seconds,'expanded_seconds':original_seconds+time.perf_counter()-started}
    result['vectors'].update({v.text:vec for v,vec in zip(views,vectors)})
    save(path,result);return result


async def read_and_score(provider,qa,packet,reader):
    from memory_system.app import eval_scoring,llm
    started=time.perf_counter();errors=[];detail={};prediction=''
    try:
        if reader=='v5':
            detail=await asyncio.wait_for(answer(provider,qa['question'],packet,qa['question_date']),60)
            prediction=detail['answer']
        else:
            prompt=eval_scoring.answer_prompt(qa,[{'id':str(i),'content':json.dumps(p,ensure_ascii=False)} for i,p in enumerate(packet)])
            prediction=(await asyncio.wait_for(llm.complete(prompt,stage='eval.answer'),60)).strip()
    except Exception as exc:errors.append({'stage':'answer','type':type(exc).__name__})
    answer_seconds=time.perf_counter()-started;started=time.perf_counter();judgement={}
    if not errors:
        try:
            judgement=Judgement.model_validate(await asyncio.wait_for(provider.complete('judge_complete_v1',{
                'question':qa['question'],'reference':qa['answer'],'proposed':prediction},Judgement),60)).model_dump()
        except Exception as exc:errors.append({'stage':'judge','type':type(exc).__name__})
    return {'prediction':prediction,'score':float(judgement.get('correct',False)) if not errors else 0.,
            'reader_details':detail,'judge':judgement,'reader_errors':errors,
            'answer_seconds':answer_seconds,'judge_seconds':time.perf_counter()-started}


async def run(root,cfg,plan,phase):
    provider=HTTPProvider(cfg,lambda row:append(root/'calls.jsonl',{**CONTEXT.get(),**row}))
    handler=LegacyUsage(lambda row:append(root/'calls.jsonl',{**CONTEXT.get(),**row}))
    logger=logging.getLogger('aml.metrics');logger.setLevel(logging.INFO);logger.addHandler(handler)
    old=base.before_modules(root)
    oldcfg=old['config'].Settings(**{k:v for k,v in cfg.model_dump().items() if k in old['config'].Settings.model_fields})
    oldcfg=oldcfg.model_copy(update={'exclusive_bridge_slots':1})
    oldprovider=old['providers'].HTTPProvider(oldcfg,provider.usage_sink);stores={}
    try:
        for c in plan:
            for family in ('C0','C1'):stores[(c['conversation_id'],family)]=Store(str(root/f"{c['conversation_id']}-{family}.db"),provider.space_id)
        if phase in ('all','ingest'):
            payloads=[(c,f,si,start,s[start:start+6]) for c in plan for f in ('C0','C1') for si,s in enumerate(c['sessions']) for start in range(0,len(s),6)]
            gate=asyncio.Semaphore(24);done=0;started=time.perf_counter()
            async def ingest(p):
                nonlocal done
                c,f,si,start,messages=p;cid=c['conversation_id']
                async with gate:
                    CONTEXT.set({'arm':f,'phase':'add','conversation_id':cid})
                    service=MemoryService(stores[(cid,f)],provider,cfg.model_copy(update={'memory_expansion':f=='C1'}))
                    response=await service.add(AddRequest(request_id=f'{cid}-{si}-{start}',user_id=cid,
                        session_id=c['session_ids'][si],messages=messages))
                    if response['degradation']:append(root/'ingestion-degradation.jsonl',{'conversation_id':cid,'family':f,'degradation':response['degradation']})
                    done+=1;save(root/'status.json',{'status':'ingesting','completed_batches':done,'total_batches':len(payloads)})
                    if done%40==0:print(f'ingested {done}/{len(payloads)}',flush=True)
            outcomes=await asyncio.gather(*(ingest(p) for p in payloads),return_exceptions=True)
            error=next((x for x in outcomes if isinstance(x,BaseException)),None)
            if error:raise error
            append(root/'ingestion-runs.jsonl',{'seconds':time.perf_counter()-started,'batches':done})
        if phase=='ingest':return
        for c in plan:
            for family in ('C0','C1'):
                actual=len(stores[(c['conversation_id'],family)].snapshot(SearchRequest(user_id=c['conversation_id'],query='integrity'))[2])
                if actual!=sum(map(len,c['sessions'])):raise ValueError('incomplete full history')
        search_rows=rows(root/'evidence.jsonl');completed={(r['arm'],r['conversation_id']) for r in search_rows}
        scored=rows(root/'results.jsonl');score_keys={(r['arm'],r['reader'],r['conversation_id']) for r in scored}
        for conv in plan:
            cid=conv['conversation_id'];qa=conv['qa'][0];prepared=await prepare_query(root,conv,provider)
            tasks=[(v,a) for v in ('before','after') for a in base.ARMS];random.Random(f'{SEED}:{cid}').shuffle(tasks)
            async def evaluate(version,arm):
                label=version+'/'+arm;context={'arm':label,'conversation_id':cid,'qa_id':'0'}
                if (label,cid) in completed:
                    entry=next(x for x in search_rows if x['arm']==label and x['conversation_id']==cid)
                else:
                    CONTEXT.set({**context,'phase':'search'});started=time.perf_counter();errors=[]
                    cls=MemoryService if version=='after' else old['service'].MemoryService
                    scfg=cfg if version=='after' else oldcfg;delegate=provider if version=='after' else oldprovider
                    service=cls(stores[(cid,arm[:2])],base.PreparedQueryProvider(delegate,prepared),scfg.model_copy(update={'memory_expansion':arm[1]=='1','query_expansion':arm[2]=='1'}))
                    try:response=await service.search(SearchRequest(user_id=cid,query=qa['question'],reference_time=qa['question_date'],top_k=10,debug=True))
                    except Exception as exc:response={};errors=[{'stage':'search','type':type(exc).__name__}]
                    observed=time.perf_counter()-started;packet=common_packet(response.get('data',[]),24000)
                    entry={**context,'packet':packet,'response':response,'search_errors':errors,'search_seconds_observed':observed,
                           'search_seconds':observed+prepared['expanded_seconds' if arm[2]=='1' else 'original_seconds']}
                    append(root/'evidence.jsonl',entry)
                for reader in ('v4','v5'):
                    if (label,reader,cid) in score_keys:continue
                    CONTEXT.set({**context,'phase':'answer_judge','reader':reader})
                    result=await read_and_score(provider,qa,entry['packet'],reader)
                    if entry['search_errors']:result['score']=0.
                    row={**context,'reader':reader,'question':qa['question'],'gold':qa['answer'],'category':qa['category'],
                         'retrieval':quote_metrics(entry['packet'],qa,conv['source_texts']),**result}
                    append(root/'results.jsonl',row);scored.append(row)
                    save(root/'status.json',{'status':'evaluating','completed_results':len(scored),'expected_results':16*len(plan)})
                    print(f"evaluated {cid} {label} reader={reader} score={result['score']}",flush=True)
            await asyncio.gather(*(evaluate(v,a) for v,a in tasks))
        save(root/'status.json',{'status':'complete','completed_results':len(scored),'expected_results':16*len(plan)})
    except BaseException as exc:
        save(root/'status.json',{'status':'interrupted','error_type':type(exc).__name__});raise
    finally:
        for store in stores.values():store.close()
        await provider.aclose();await oldprovider.aclose();logger.removeHandler(handler)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['plan','all','ingest','evaluate'],default='plan')
    args=parser.parse_args();root=Path(ROOT);cfg=settings()
    metadata=json.loads(Path('memory_connection/data/longmemeval-source.json').read_text())
    if file_hash('memory_connection/data/longmemeval_s_cleaned.json')!=metadata['sha256']:
        raise ValueError('dataset hash mismatch')
    plan=prepare('memory_connection/data/longmemeval_s_cleaned.json')
    hashes=base.hashes()
    for p in (Path(__file__),Path(__file__).with_name('longmemeval_data.py')):hashes[str(p.relative_to(Path.cwd()) if p.is_absolute() else p)]=file_hash(p)
    manifest={'protocol':'full-longmemeval-crossed-readers-v5','seed':SEED,'strata':STRATA,'source_hashes':hashes,
              'dataset':json.loads(Path('memory_connection/data/longmemeval-source.json').read_text()),'settings':base.safe_settings(cfg),
              'selected':[{'conversation_id':c['conversation_id'],'category':c['qa'][0]['category'],
                           'messages':len(c['originals']),'chunks':sum(map(len,c['sessions'])),'sessions':len(c['sessions'])} for c in plan],
              'readers':['v4','v5'],'expected_searches':8*len(plan),'expected_results':16*len(plan),
              'add_concurrency':24,'query_concurrency':8,'evidence_bytes':24000,'timestamp_assumption':'Dataset wall-clock timestamps treated as UTC consistently',
              'splitting':'lossless newline-preferred chunks of 2400 characters; all sessions retained; no gold-based filtering'}
    base.before_modules(root)
    if (root/'manifest.json').exists():
        if json.loads((root/'manifest.json').read_text(encoding='utf-8'))!=manifest:raise ValueError('frozen experiment drift')
    else:save(root/'manifest.json',manifest);save(root/'plan.json',plan);save(root/'prices.snapshot.json',json.loads(Path('memory_connection/benchmark-prices-usd.json').read_text()))
    registry_path=Path('memory_connection/data/usage_registry.json')
    registry=json.loads(registry_path.read_text()) if registry_path.exists() else []
    current={'run':ROOT,'dataset_sha256':metadata['sha256'],'question_ids':[c['conversation_id'] for c in plan],
             'session_ids':[sid for c in plan for sid in c['session_ids']]}
    for previous in registry:
        if previous['run']!=ROOT and set(previous['session_ids'])&set(current['session_ids']):
            raise ValueError('previously used history overlap')
    if current not in registry:registry.append(current);save(registry_path,registry)
    print(json.dumps({'selected':manifest['selected'],'expected_results':manifest['expected_results']}),flush=True)
    if args.phase!='plan':asyncio.run(run(root,cfg,plan,args.phase))


if __name__=='__main__':main()
