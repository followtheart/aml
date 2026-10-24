"""Real CE diagnostic: complete visible user messages as alternative rank views.

No source is translated, clipped, rehydrated or promoted to a personal fact.
By default at most two complete original user messages per candidate are scored;
--all-user-sources includes every eligible complete visible user message.
identical source views are shared across candidates. Whole prepared units stay
intact for fine/main admission and eventual evidence delivery. Baseline CE
scores are historical; source-view scores are fresh real provider responses.
"""
import argparse
import ast
import asyncio
import copy
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_context,budget,config,cross_encoder,fine_evidence,schemas
from app.conversation_context import source_key,usable
from app.fine_context_reserve import full_user
from evaluation_model_guard import check_runtime,check_versions
from probe_coarse_fine_core import admit
import audit_https_funnel as audit


def records(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def build_views(coarse,user_id,query,*,max_sources=2):
    if max_sources is not None and (type(max_sources) is not int or max_sources < 1):
        raise ValueError('Source cap must be positive or None for all eligible sources')
    views={};members={};omitted=[]
    for candidate in coarse:
        item=candidate.get('_packet_item',{});sources=[s for s in item.get('sources',[]) if s.get('content')]
        if (candidate.get('view_status')=='stale' or item.get('is_constraint') or
                any(not usable(s,user_id) or s['content'] not in item.get('content','') for s in sources)):
            continue
        chosen=[]
        for s in sources:
            if not full_user(s) or s['content'] not in candidate.get('_rank_text',''):continue
            key=hashlib.sha256(source_key(s).encode()).hexdigest()
            if key in chosen:continue
            text=answer_context.with_evidence('Conversation excerpt:',[s])
            if (len(text.encode())>config.CE_MAX_DOCUMENT_BYTES or
                    cross_encoder.request_size(query,[text])>config.CE_MAX_REQUEST_BYTES):
                omitted.append(dict(candidate_id=candidate['id'],source_key=key,reason='input_budget'));continue
            views.setdefault(key,dict(id=key,text=text,source=copy.deepcopy(s)))
            chosen.append(key)
            if max_sources is not None and len(chosen)>=max_sources:break
        members[candidate['id']]=chosen
    batches=[];batch=[]
    for view in views.values():
        if batch and (len(batch)>=config.CE_BATCH_SIZE or cross_encoder.request_size(query,[v['text'] for v in batch+[view]])>config.CE_MAX_REQUEST_BYTES):
            batches.append(batch);batch=[]
        batch.append(view)
    if batch:batches.append(batch)
    return views,members,batches,omitted


def select(t,coarse):
    c=t['cascade'];baseline=set(c['coarse'].get('baseline_ids',c['coarse']['selected_ids']))
    allowed=None;pool=coarse
    if config.COARSE_EVIDENCE_EXTEND and config.FINE_COARSE_CORE_QUOTA:
        extras=sorted([v for v in coarse if v['id'] not in baseline],key=fine_evidence.key,reverse=True)
        allowed=baseline|{v['id'] for v in extras[:1]};pool=[v for v in coarse if v['id'] in baseline]
    fused={v['id'] for v in sorted(pool,key=lambda v:-(v.get('_fused') or 0))[:config.CASCADE_FUSED_RESERVE]}
    fine,selection=fine_evidence.propose(coarse,min(config.CASCADE_FINE_LIMIT,len(coarse)),
        t['coverage_manifest']['coverage']['requirements'],fused_ids=fused,extra_limit=config.FINE_EXTRA_LIMIT,
        core_pool_ids=allowed,soft_promotion_limit=1 if config.FINE_WEAK_PROMOTION_ONE else None)
    main,size=admit(t,fine,selection)
    return dict(fine=fine,main=main,main_bytes=size)


async def main(args):
    check_runtime()
    if config.FAKE:raise ValueError('Real provider required')
    plan_path=args.output.with_suffix('.plan.json');live_path=args.output.with_suffix('.live.jsonl')
    if any(p.exists() for p in (args.output,plan_path,live_path)):raise FileExistsError(args.output)
    if (args.run/'exit-code').read_text().strip()!='0':raise ValueError('Require completed successful arm')
    manifest=json.loads((args.run/'isolated-manifest.json').read_text())
    if manifest.get('audit_exit_code')!=0 or 'server_exit_code' not in manifest:
        raise ValueError('Source launcher/audit have not finished')
    if hashlib.sha256(args.prepared.read_bytes()).hexdigest()!=manifest['data_sha256']:
        raise ValueError('Prepared dataset identity differs')
    conversations={c['conversation_id']:c for c in records(args.prepared)}
    result_rows=records(args.run/'completed/data/results/personamem-v2-32k.jsonl')
    results={r['search_id']:r for r in result_rows}
    if len(results)!=len(result_rows) or len({r['qa_id'] for r in result_rows})!=len(result_rows):
        raise ValueError('Duplicate source question/Search identity')
    if args.all_questions:
        expected=manifest.get('questions')
        if expected is None:
            evaluation=manifest['evaluation']
            if evaluation['AML_EVAL_LIMIT']!=0:raise ValueError('Require complete histories')
            selected=list(conversations.values())[:evaluation['AML_EVAL_CONVS'] or None]
            expected=sum(len(c['qa']) for c in selected)
        if len(result_rows)!=expected:raise ValueError('Incomplete full cohort')
    wanted=({r['qa_id'] for r in result_rows} if args.all_questions else
            set((args.qas or '5,7,25,30,31,47').split(',')));jobs=[]
    for t in records(args.run/'server-search-debug.jsonl'):
        qa=results[t['search_id']]['qa_id']
        if qa not in wanted:continue
        result=results[t['search_id']]
        conv=conversations[result['conversation_id']]
        question=next(q for q in conv['qa'] if q['id']==qa)
        if (result.get('error_stage') or result.get('fake') is not False or result['transport']!='https'
                or result['packet_hash']!=t['packet_hash'] or result['user_id']!=t['user_id']
                or t['query']!=question['question'] or t['options']!=question['options']):
            raise ValueError('Source result/question identity differs')
        check_versions(t['versions'],'source trace')
        if t['cascade']['cross_encoder']['status']!='ok':raise ValueError('Historical CE incomplete')
        byid={c['id']:c for c in t['ranked']};coarse=[byid[mid] for mid in t['cascade']['coarse']['selected_ids']]
        with patch.multiple(config,**t['versions']['settings']):
            if config.CE_MODEL!=t['cascade']['cross_encoder']['model']:raise ValueError('CE model changed')
            if not config.FINE_EVIDENCE_ADAPTIVE or config.FINE_TOPIC_CONTEXT:raise ValueError('Unsupported fine mode')
            original=select(t,coarse)
            if ([c['id'] for c in original['fine']]!=t['cascade']['fine']['selected_ids'] or
                    [c['id'] for c in original['main']]!=t['cascade']['listwise']['candidate_ids'] or
                    original['main_bytes']!=t['cascade']['listwise']['input_bytes']):
                raise ValueError('Historical fine/main did not reproduce')
            query=t['cascade']['cross_encoder']['query']
            views,members,batches,omitted=build_views(coarse,t['user_id'],query,
                max_sources=None if args.all_user_sources else 2)
        jobs.append(dict(qa_id=qa,t=t,coarse=coarse,query=query,views=views,members=members,batches=batches,omitted=omitted))
    if {j['qa_id'] for j in jobs}!=wanted:raise ValueError('Missing question')
    plan=dict(run=str(args.run),llm_model=config.LLM_MODEL,ce_model=config.CE_MODEL,
        source_policy='all_eligible_complete_user_sources' if args.all_user_sources else 'first_two_complete_user_sources',
        questions=len(jobs),prepared_sha256=manifest['data_sha256'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        expected_calls=sum(len(j['batches']) for j in jobs),source_trace_sha256=hashlib.sha256((args.run/'server-search-debug.jsonl').read_bytes()).hexdigest(),
        jobs=[dict(qa_id=j['qa_id'],query=j['query'],views=list(j['views'].values()),members=j['members'],omitted=j['omitted'],
                   batches=[[v['id'] for v in batch] for batch in j['batches']]) for j in jobs],limitation=__doc__)
    if args.preflight:
        print(json.dumps(dict(questions=len(jobs),source_policy=plan['source_policy'],
            expected_calls=plan['expected_calls'],source_views=sum(len(j['views']) for j in jobs),
            omitted_sources=sum(len(j['omitted']) for j in jobs),reproduced_fine_and_main=len(jobs),
            provider_calls=0,plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest())))
        return
    plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    live=[];scored=[]
    for job in jobs:
        scores={}
        with patch.multiple(config,**job['t']['versions']['settings']):
            for index,batch in enumerate(job['batches']):
                check_runtime();texts=[v['text'] for v in batch];bound=cross_encoder.request_size(job['query'],texts)
                row=dict(qa_id=job['qa_id'],batch=index,view_ids=[v['id'] for v in batch],request_bound=bound)
                with budget.scope(seconds=30,calls=1,tokens=bound) as limits:
                    try:
                        values=await cross_encoder.rerank(job['query'],texts,stage='diagnostic.source_ce')
                        scores.update(zip(row['view_ids'],values));row['scores']=values
                    except Exception as exc:row.update(error=type(exc).__name__,detail=str(exc))
                row.update(calls=limits.calls,tokens=limits.tokens);live.append(row)
                with live_path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('view_ids','scores','detail')}),flush=True)
            if len(scores)!=len(job['views']):continue
            variants={'baseline':select(job['t'],job['coarse'])};rankings={}
            for mode in ('source_only','maximum'):
                revised=copy.deepcopy(job['coarse'])
                for c in revised:
                    values=[scores[k] for k in job['members'].get(c['id'],[])]
                    if values:c['_ce_score']=max(values+([c['_ce_score']] if mode=='maximum' else []))
                variants[mode]=select(job['t'],revised)
                rankings[mode]={c['id']:c['_ce_score'] for c in revised}
            scored.append(dict(job=job,variants=variants,rankings=rankings,source_scores=scores))
    # Gold is consulted only after provider requests, scores and selections are fixed.
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:gold=list(csv.DictReader(f))
    raw={}
    for event in records(args.run/'server-memory-debug.jsonl'):
        raw.setdefault(event['user_id'],{}).update({audit.identity(s):s for s in event['source_messages']})
    rows=[]
    for record in scored:
        j=record['job'];snippet=gold[int(j['qa_id'])]['related_conversation_snippet']
        try:snippet=json.loads(snippet)
        except ValueError:snippet=ast.literal_eval(snippet)
        variants={}
        for mode,variant in record['variants'].items():
            result=dict(main_bytes=variant['main_bytes'])
            for stage in ('fine','main'):
                cards=[dict(s,text=s.get('content',''),id=f'{i}:{k}') for i,c in enumerate(variant[stage])
                       for k,s in enumerate(c['_packet_item'].get('sources',[]))]
                result[stage]=dict(ids=[c['id'] for c in variant[stage]],**{role:[v['full'] for v in audit.turn_coverage(
                    [audit.norm(s['content']) for s in snippet if s['role']==role],cards,raw[j['t']['user_id']],role=role)]
                    for role in ('user','assistant')})
            variants[mode]=result
        rows.append(dict(qa_id=j['qa_id'],variants=variants,rankings=record['rankings'],source_scores=record['source_scores']))
    summary=dict(questions=len(rows),calls=sum(r['calls'] for r in live),tokens=sum(r['tokens'] for r in live),errors=sum('error' in r for r in live),variants={})
    for mode in ('baseline','source_only','maximum'):
        summary['variants'][mode]={stage:dict(reference_questions=sum(any(r['variants'][mode][stage]['user']) for r in rows),
            user_turns=sum(sum(r['variants'][mode][stage]['user']) for r in rows),
            gained=[r['qa_id'] for r in rows if any(r['variants'][mode][stage]['user']) and not any(r['variants']['baseline'][stage]['user'])],
            lost=[r['qa_id'] for r in rows if any(r['variants']['baseline'][stage]['user']) and not any(r['variants'][mode][stage]['user'])]) for stage in ('fine','main')}
    args.output.write_text(json.dumps(dict(summary=summary,rows=rows,live=live,plan=str(plan_path),limitation=__doc__),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    if summary['errors'] or len(rows)!=len(jobs):raise RuntimeError('Incomplete diagnostic')


if __name__=='__main__':
    provider_logger=logging.getLogger('aml.metrics')
    provider_logger.setLevel(logging.INFO)
    provider_logger.addHandler(logging.StreamHandler())
    provider_logger.propagate=False
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    choice=p.add_mutually_exclusive_group()
    choice.add_argument('--qas')
    choice.add_argument('--all-questions',action='store_true')
    p.add_argument('--all-user-sources',action='store_true')
    p.add_argument('--preflight',action='store_true',help='Validate all inputs and print cost bounds without writing or calling providers')
    p.add_argument('--prepared',type=Path,default=Path(__file__).resolve().parents[1]/'data/prepared/personamem-v2-32k.jsonl')
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    asyncio.run(main(p.parse_args()))
