"""Paired full-option CE diagnostic on all questions of a completed cohort.

The input question and existing query prefix remain intact; all original
options must fit the existing auxiliary bound together. Candidate documents,
coarse order, source roles, fine policy and main admission policy stay fixed.
Unchanged queries reuse archived scores explicitly. No answer accuracy claim.
"""
import argparse
import asyncio
import copy
import csv
import hashlib
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget, config, cross_encoder, rerank_query, run_metadata
import probe_current_ce_topics as base
from probe_ce_source_units import score_documents
from evaluate_recall_context import reference_turns
from evaluation_model_guard import check_runtime
import audit_https_funnel as funnel


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def queries_for(t):
    req=SimpleNamespace(query=t['query'],options=t['options'])
    plan=copy.deepcopy(t['plan']);plan['_query_specs']=copy.deepcopy(t['query_specs'])
    with patch.object(config,'CE_FULL_OPTION_TEXT',False):
        original=rerank_query.build(req,plan)
    if original!=t['cascade']['cross_encoder']['query']:
        raise ValueError('Original CE query did not reproduce')
    with patch.object(config,'CE_FULL_OPTION_TEXT',True):
        candidate=rerank_query.build(req,plan)
    return dict(control=original,candidate=candidate)


def prepare(run, *, query_builder=None):
    manifest=json.loads((run/'isolated-manifest.json').read_text())
    if manifest.get('audit_exit_code')!=0 or 'server_exit_code' not in manifest:
        raise ValueError('Source run is not audited and terminal')
    prepared=Path(manifest['data'])
    if sha(prepared)!=manifest['data_sha256']:
        raise ValueError('Prepared input changed')
    conversations=base.records(prepared)
    questions={(c['conversation_id'],q['id']):q for c in conversations for q in c['qa']}
    results=base.records(run/'completed/data/results/personamem-v2-32k.jsonl')
    expected=manifest.get('questions')
    if expected is None:
        if manifest['evaluation']['AML_EVAL_LIMIT']!=0:
            raise ValueError('Require complete selected histories')
        expected=sum(len(c['qa']) for c in conversations[:manifest['evaluation']['AML_EVAL_CONVS'] or None])
    if len(results)!=expected:
        raise ValueError('Incomplete question cohort')
    by_search={r['search_id']:r for r in results}
    with patch.object(base,'queries_for',query_builder or queries_for):
        jobs=base.prepare(run)
    for j in jobs:
        t=j['t'];r=by_search[t['search_id']];q=questions[r['conversation_id'],r['qa_id']]
        if (r.get('error_stage') or r['packet_hash']!=t['packet_hash'] or r['user_id']!=t['user_id']
                or t['query']!=q['question'] or t['options']!=q['options']):
            raise ValueError('Source result/question identity mismatch')
        j['conversation_id']=r['conversation_id']
    return jobs,manifest,conversations


def batches_for(query,docs):
    result=[];pending=[]
    for doc in docs:
        if len(query.encode())>config.CE_MAX_DOCUMENT_BYTES:
            raise ValueError('Query exceeds CE byte bound')
        if pending and (len(pending)>=config.CE_BATCH_SIZE or
                cross_encoder.request_size(query,pending+[doc])>config.CE_MAX_REQUEST_BYTES):
            result.append(pending);pending=[]
        pending.append(doc)
    if pending:result.append(pending)
    return len(result)


def coverage(job,selection,turns,raw):
    entry={'main_bytes':selection['main_bytes']}
    for stage in ('fine','main'):
        cards=[dict(s,text=s.get('content',''),id=f'{i}:{k}')
               for i,c in enumerate(selection[stage]) for k,s in enumerate(c['_packet_item'].get('sources',[]))]
        values=funnel.turn_coverage(turns,cards,raw)
        entry[stage]=dict(ids=[c['id'] for c in selection[stage]],user=[v['full'] for v in values])
    return entry


async def main(args):
    jobs,manifest,conversations=prepare(args.run,query_builder=getattr(args,'query_builder',None))
    expected_calls=0
    for j in jobs:
        if j['changed']:
            with patch.multiple(config,**j['t']['versions']['settings']):
                expected_calls+=sum(batches_for(q,j['docs']) for q in j['queries'].values())
    preflight=dict(questions=len(jobs),changed=sum(j['changed'] for j in jobs),
        unchanged_reuse=sum(not j['changed'] for j in jobs),expected_calls=expected_calls,
        reconstructed_fine_and_main=len(jobs),max_auxiliary_bytes=max(
            len(j['queries']['candidate'][len(j['t']['query']):].encode()) for j in jobs))
    if args.preflight:
        print(json.dumps(preflight));return
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    sources={str(p):sha(p) for p in [Path(__file__),Path(base.__file__),
        Path(__file__).with_name('probe_source_ce.py'),Path(__file__).with_name('probe_coarse_fine_core.py'),
        Path(__file__).with_name('probe_ce_source_units.py'),*getattr(args,'additional_scripts',[])]}
    plan=dict(run=str(args.run.resolve()),prepared=str(Path(manifest['data']).resolve()),
        data_sha256=manifest['data_sha256'],trace_sha256=sha(args.run/'server-search-debug.jsonl'),
        result_sha256=sha(args.run/'completed/data/results/personamem-v2-32k.jsonl'),
        versions=run_metadata.versions(),script_hashes=sources,preflight=preflight,
        query_policy=getattr(args,'query_policy','complete_options'),limitation=getattr(args,'limitation',__doc__),
        jobs=[dict(qa_id=j['qa_id'],conversation_id=j['conversation_id'],search_id=j['t']['search_id'],
                   queries=j['queries'],changed=j['changed'],candidate_ids=[c['id'] for c in j['coarse']],
                   document_hashes=[hashlib.sha256(d.encode()).hexdigest() for d in j['docs']]) for j in jobs])
    (output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(preflight),flush=True)
    live=[];selections=[]
    for job in jobs:
        scores={'archived':[c['_ce_score'] for c in job['coarse']]};usage_by_arm={}
        for arm,query in job['queries'].items():
            if not job['changed']:
                scores[arm]=scores['archived'];continue
            with patch.multiple(config,**job['t']['versions']['settings']):
                check_runtime()
                with budget.scope(seconds=240,calls=100,tokens=1000000) as usage:
                    values,batches=await score_documents(query,job['docs'])
            base.replay(job,values)
            event=dict(qa_id=job['qa_id'],conversation_id=job['conversation_id'],arm=arm,
                       query=query,scores=values,calls=usage.calls,tokens=usage.tokens,batches=batches)
            with (output/'responses.jsonl').open('a') as stream:
                stream.write(json.dumps(event,ensure_ascii=False)+'\n')
            live.append(event);scores[arm]=values;usage_by_arm[arm]=usage.tokens
            print(json.dumps({k:event[k] for k in ('qa_id','arm','calls','tokens')}),flush=True)
        variants={name:base.replay(job,values) for name,values in scores.items()}
        delta=max(0,usage_by_arm.get('candidate',0)-usage_by_arm.get('control',0))
        adjusted=copy.deepcopy(job)
        headroom=adjusted['t']['cascade']['admission_budget']['token_headroom']
        if headroom is not None:
            adjusted['t']['cascade']['admission_budget']['token_headroom']=max(0,headroom-delta)
        variants['candidate_cost_adjusted']=base.replay(adjusted,scores['candidate'])
        selections.append((job,variants,delta))
    # No reference label is consulted until every request and selection is fixed.
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream:gold=list(csv.DictReader(stream))
    raw={}
    for event in base.records(args.run/'server-memory-debug.jsonl'):
        raw.setdefault(event['user_id'],{}).update({funnel.identity(s):s for s in event['source_messages']})
    convs={c['conversation_id']:c for c in conversations};rows=[]
    for job,variants,delta in selections:
        conv=convs[job['conversation_id']];qa=next(q for q in conv['qa'] if q['id']==job['qa_id'])
        turns=reference_turns(qa,conv['source_history'],gold[int(job['qa_id'])])
        rows.append(dict(qa_id=job['qa_id'],conversation_id=job['conversation_id'],changed=job['changed'],
            additional_token_cost=delta,variants={name:coverage(job,v,turns,raw[job['t']['user_id']])
                                                for name,v in variants.items()}))
    summary=dict(**preflight,calls=sum(e['calls'] for e in live),tokens=sum(e['tokens'] for e in live),variants={})
    for name in ('archived','control','candidate','candidate_cost_adjusted'):
        summary['variants'][name]={stage:dict(
            reference_questions=sum(any(r['variants'][name][stage]['user']) for r in rows),
            user_turns=sum(sum(r['variants'][name][stage]['user']) for r in rows),
            gained=[r['qa_id'] for r in rows if any(r['variants'][name][stage]['user']) and not any(r['variants']['control'][stage]['user'])],
            lost=[r['qa_id'] for r in rows if any(r['variants']['control'][stage]['user']) and not any(r['variants'][name][stage]['user'])],
            fewer_turns=[r['qa_id'] for r in rows if sum(r['variants'][name][stage]['user'])<sum(r['variants']['control'][stage]['user'])])
            for stage in ('fine','main')}
    summary['cost_adjustment_limitation']='Subtracts positive fresh candidate minus fresh control CE token difference from recorded main admission headroom; not a replay of live CE scheduling or deadlines.'
    (output/'result.json').write_text(json.dumps(dict(summary=summary,rows=rows),ensure_ascii=False,indent=2)+'\n')
    (output/'exit-code').write_text('0\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    args=p.parse_args()
    asyncio.run(main(args))
