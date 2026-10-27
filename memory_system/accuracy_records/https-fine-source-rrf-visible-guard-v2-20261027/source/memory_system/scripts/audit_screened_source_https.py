"""Audit actual HTTPS source nomination against each request's own baseline.

Reconstruct complete input objects, views and display requests offline. Actual
provider scores are checked against their view identities; no scores are reused
to execute a new request. Downstream packet/answer changes are not attributed
causally by this audit.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from unittest.mock import patch

from batched_records import file_digest,json_records,physical_records
from audit_retained_porter_https import selection_code,same,require
from probe_assistant_admission_order import replay
from probe_screened_source_nomination import read


def main(args):
    run=args.run.resolve();plan=read(run.parent/'plan.json')
    require(plan['toggle']=='SOURCE_CE_LATE_ADMISSION','Late-source HTTPS plan')
    for name,digest in plan['source_files'].items():same(file_digest(run.parent/'source'/name),digest,'Frozen HTTPS source')
    sys.path.insert(0,str(run.parent/'source/memory_system'))
    from app import cascade_rerank,config,fine_evidence,schemas,screened_source_nomination as source
    code,_=selection_code(cascade_rerank)
    results={r['search_id']:r for r in json_records(run/'completed/data/results/personamem-v2-32k.jsonl')}
    rows=[];seen=set();batches=0;successful_batches=0
    with patch('socket.socket.connect',side_effect=AssertionError('Offline HTTPS source audit')):
        for trace in json_records(run/'server-search-debug.jsonl'):
            sid=trace['search_id'];require(sid in results and sid not in seen,'Unique terminal search')
            seen.add(sid);recorded=trace['cascade'];details=recorded.get('source_nomination')
            with patch.multiple(config,**trace['versions']['settings']):
                baseline,_,original_bytes=replay(trace,(cascade_rerank,config,fine_evidence,schemas),code)
                same(baseline['trace']['fine'],recorded['fine'],'Original fine unchanged')
                same(baseline['trace'].get('adaptive_admission'),recorded.get('adaptive_admission'),'Original adaptive admission')
                original=baseline['submitted'];ids=[c['id'] for c in original]
                req,selection_plan=baseline['req'],baseline['plan']
                actual=list(original);added=None
                if not config.SOURCE_CE_LATE_ADMISSION:
                    require(details is None,'Default-off source stage not run')
                    display=False
                else:
                    require(details is not None,'Source stage trace required')
                    same(ids,details['original_main_ids'],'Own original main prefix')
                    display=details['display_ids']
                    if 'scheduling' in details:
                        job=source.prepare(req,selection_plan,baseline['coarse'],baseline['fine'],original,
                                           details['original_headroom'],cascade_rerank)
                        same(job['decisions'],details['decisions'],'Actual eligibility and budget decisions')
                        same([c['id'] for c in job['nominees']],details['eligible_ids'],'Actual nominees')
                        same(job['ce_query'],details['query'],'Actual source query')
                        same(list(job['views']),details['source_view_ids'],'Actual view namespace')
                        scores={}
                        for index,batch in enumerate(details['batches']):
                            same(batch['batch'],index,'Sequential unique batch')
                            same(batch['view_ids'],job['batches'][index],'Exact batch membership/order')
                            same(batch['bound'],job['batch_bounds'][index],'Source input bound')
                            batches+=1
                            if batch['status']=='ok':
                                successful_batches+=1
                                require(len(batch['scores'])==len(batch['view_ids']) and
                                    all(type(v) in (int,float) and math.isfinite(v) for v in batch['scores']),'Complete finite scores')
                                scores.update(zip(batch['view_ids'],batch['scores'],strict=True))
                            else:require(batch['status']=='error','Terminal source batch')
                        if details['status']=='admitted':
                            require(len(details['batches'])==len(job['batches']) and
                                    all(b['status']=='ok' for b in details['batches']),'Complete source pass before nomination')
                            decision=source.nominate(job,scores)
                            same(details['addition'],decision['addition'],'Actual score-based nomination')
                            same(details['ranking'],decision['ranking'],'Complete source ranking')
                            added=next(c for c in job['nominees'] if c['id']==decision['addition'])
                            actual.append(added['candidate'])
                            same(details['display_audit'],added['display_audit'],'Original graph/display audit')
                    if details['status']!='admitted':
                        require(not display and details['addition'] is None,'No-op and failures preserve original display')
                same([c['id'] for c in actual],recorded['listwise']['candidate_ids'],'Actual original-plus-one main')
                request=cascade_rerank.listwise_request(req,selection_plan,actual,display_ids=display)
                same(len(request['prompt'].encode()),recorded['listwise']['input_bytes'],'Actual prompt bytes')
                require(len(actual)<=config.CASCADE_LLM_LIMIT+config.FINE_EXTRA_LIMIT,'Original count budget')
                require(len(request['prompt'].encode())<=config.RERANK_MAX_PROMPT_BYTES,'Original byte budget')
                rows.append(dict(search_id=sid,qa_id=results[sid]['qa_id'],baseline_exact=True,
                    status=details['status'] if details else 'disabled',addition=added['id'] if added else None,
                    original_bytes=original_bytes,actual_bytes=recorded['listwise']['input_bytes']))
    same(seen,set(results),'Complete terminal search coverage')
    ledger=[]
    for record in physical_records(run/'server.log'):
        text=record.decode(errors='replace')
        if 'provider_call ' in text and 'stage=search.source_nomination ' in text:
            fields=dict(re.findall(r'(\w+)=([^\s]+)',text))
            require(fields['model']=='qwen3-rerank' and fields['kind']=='rerank' and fields['attempt']=='1'
                    and fields['retries']=='0','Fixed source provider policy')
            ledger.append(fields)
    require(len(ledger)<=batches,'No extra source dispatch or retry')
    require(sum(r['status']=='ok' for r in ledger)>=successful_batches,'Every successful source batch recorded')
    summary=dict(questions=len(rows),statuses=dict(Counter(r['status'] for r in rows)),
        appended=sum(r['addition'] is not None for r in rows),source_batches=batches,source_calls=len(ledger),
        source_reported_tokens=sum(int(r['total_tokens']) for r in ledger),
        source_provider_errors=sum(r['status']!='ok' for r in ledger))
    inputs=[run.parent/'plan.json',run/'server-search-debug.jsonl',run/'server.log',run/'completed/data/results/personamem-v2-32k.jsonl']
    with (run/'source-nomination-audit.json').open('x') as stream:
        json.dump(dict(status='passed',summary=summary,rows=rows,limitation=__doc__,provider_calls_by_audit=0,
                       input_sha256={str(p):file_digest(p) for p in inputs}),stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args())
