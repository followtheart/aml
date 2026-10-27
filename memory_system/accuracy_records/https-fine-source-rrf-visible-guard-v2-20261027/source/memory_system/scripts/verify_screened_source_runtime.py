"""Verify current default-off baseline and source admission against frozen cases.

No providers or references are accessed; actual audited source scores are replayed
only after exact current inventory/request identity has been established.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget, cascade_rerank, config, fine_evidence, schemas, screened_source_nomination
from batched_records import json_records,file_digest
from audit_retained_porter_https import same,require,selection_code
from probe_assistant_admission_order import replay
from probe_screened_source_nomination import read
import local_candidate_ids


async def main(args):
    experiment=args.experiment.resolve()
    plan=read(experiment/'plan.json');audit=read(experiment/'audit.json')
    same(audit['decision'],'proceed_to_context_review','Audited actual gain')
    for p,digest in audit['input_sha256'].items():same(file_digest(Path(p)),digest,'Audited paid artifacts')
    spec=plan['cohorts'][args.cohort]
    run=Path(spec['runtime_source']).parent.parent/'control'
    selections={r['search_id']:r for r in json_records(experiment/(args.cohort+'-selections.jsonl'))}
    calls={(r['search_id'],r['batch']):r for r in json_records(experiment/(args.cohort+'-requests.jsonl')) if r['status']=='ok'}
    code,code_hash=selection_code(cascade_rerank)
    rows=[]
    with patch('socket.socket.connect',side_effect=AssertionError('Offline runtime parity')):
        for trace,expected in zip(json_records(run/'server-search-debug.jsonl'),json_records(Path(spec['inventory'])),strict=True):
            sid=trace['search_id'];same(sid,expected['search_id'],'Trace inventory identity')
            fixed=selections[sid]
            settings=dict(trace['versions']['settings'],SOURCE_CE_LATE_ADMISSION=False)
            with patch.multiple(config,**settings):
                baseline,_,size=replay(trace,(cascade_rerank,config,fine_evidence,schemas),code)
                same(baseline['trace']['fine'],trace['cascade']['fine'],'Exact fine baseline')
                same(baseline['trace'].get('adaptive_admission'),trace['cascade'].get('adaptive_admission'),'Exact admission')
                original=baseline['submitted'];before=copy.deepcopy(original)
                same([c['id'] for c in original],expected['original_main_ids'],'Original main order')
                same(size,expected['original_bytes'],'Original bytes')
                req,selection_plan=baseline['req'],baseline['plan']
                with patch.object(screened_source_nomination,'prepare',side_effect=AssertionError('Disabled must not prepare')):
                    disabled,details=await screened_source_nomination.append(req,selection_plan,baseline['coarse'],baseline['fine'],original,cascade_rerank)
                    require(disabled is original and details['status']=='disabled','Exact default-off no-op')
                job=screened_source_nomination.prepare(req,selection_plan,baseline['coarse'],baseline['fine'],original,
                    trace['cascade']['admission_budget'],cascade_rerank)
                for name in ('ce_query','decisions','views','batches','batch_bounds','scheduling','omitted_source_views',
                             'represented_view_ids','original_display_audit','encoded_original_bytes'):
                    same(job[name],expected[name],'Runtime inventory '+name)
                same(job['nominees'],[{k:v for k,v in c.items() if k!='cards'} for c in expected['nominees']],
                    'Every nominee object, order, complete sources and display audit')
                index=0
                async def ce(query,documents,stage,**kwargs):
                    nonlocal index
                    same(query,job['ce_query'],'Exact runtime CE query')
                    keys=job['batches'][index]
                    same(documents,[job['views'][key]['text'] for key in keys],'Exact runtime ordered CE documents')
                    same(kwargs,dict(rate_limit_retries=0),'No source retry policy')
                    event=calls[sid,index]
                    limits=budget.current.get()
                    require(limits.reserved_calls==1 and limits.reserved_tokens==job['scheduling']['main_reservation'],
                        'Protected live main reservations')
                    limits.before_call();limits.tokens+=event['tokens']
                    index+=1
                    return event['scores']
                with patch.object(config,'SOURCE_CE_LATE_ADMISSION',True), \
                        patch.object(screened_source_nomination,'prepare',return_value=job), \
                        patch.object(screened_source_nomination.cross_encoder,'rerank',side_effect=ce), \
                        budget.scope(seconds=expected['original_headroom']['deadline_seconds'],
                            calls=expected['original_headroom']['call_headroom'],tokens=expected['original_headroom']['token_headroom']) as limits:
                    proposed,details=await screened_source_nomination.append(req,selection_plan,baseline['coarse'],baseline['fine'],original,cascade_rerank)
                    same((limits.reserved_calls,limits.reserved_tokens),(0,0),'All reservations released')
                    same(limits.calls,fixed['budget_calls'],'Actual provider call count parity')
                    same(limits.tokens,fixed['reported_tokens'],'Actual provider usage parity')
                same([c['id'] for c in proposed],fixed['proposed_main_ids'],'Exact runtime nomination')
                same(original,before,'Original candidate immutability')
                same(proposed[:len(original)],original,'Complete main prefix retained')
                request=cascade_rerank.listwise_request(req,selection_plan,proposed,display_ids=details['display_ids'])
                if details['addition']:
                    reference,reference_audit=local_candidate_ids.request(req,selection_plan,proposed,cascade_rerank)
                    same(request,reference,'Entire production request versus audited display adapter')
                    same(details['display_audit'],reference_audit,'Displayed graph and identities')
                else:
                    same(request,cascade_rerank.listwise_request(req,selection_plan,original),'No-op unencoded request')
                rows.append(dict(qa_id=expected['qa_id'],baseline_exact=True,inventory_exact=True,
                    runtime_nomination_exact=True,addition=details['addition'],display_bytes=len(request['prompt'].encode())))
            print(json.dumps(rows[-1]),flush=True)
    same(len(rows),len(spec['search_ids']),'Complete cohort')
    report=dict(status='passed',cohort=args.cohort,questions=len(rows),rows=rows,provider_calls=0,selection_ast_sha256=code_hash,
        input_sha256={str(p):file_digest(p) for p in [experiment/'plan.json',experiment/'audit.json',Path(spec['inventory']),
            run/'server-search-debug.jsonl',Path(__file__),*(Path(__file__).resolve().parents[1]/'app').glob('*.py')]})
    with args.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--cohort',choices=('regression','validation'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    asyncio.run(main(parser.parse_args()))
