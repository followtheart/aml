"""Compare archived source-CE query/document batches with current full cohorts.

No selector, labels, provider calls, or claims of full provider-wire equivalence.
Exact query/model/document ordering is a necessary compatibility check; runtime
provider settings and request construction still need checking before reuse.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

from batched_records import file_digest, json_records
from audit_retained_porter_https import require, same


def read(path):
    data=bytearray()
    with path.open('rb') as stream:
        while block:=stream.read(4*1024*1024): data.extend(block)
    return json.loads(data)


def main(args):
    experiment=args.experiment.resolve()
    experiment_plan=read(experiment/'plan.json')
    run=Path(experiment_plan['cohorts'][args.cohort]['run'])
    archive=args.archive.resolve()
    audit=read(archive/'audit.json')['cohorts'][args.cohort]
    archive_inputs=[]
    for suffix,digest in audit['artifact_hashes'].items():
        path=archive/(args.cohort+'.'+suffix)
        same(file_digest(path),digest,'Audited source CE artifact')
        archive_inputs.append(path)
    old=read(archive/(args.cohort+'.plan.json'))
    require(old['source_policy']=='all_eligible_complete_user_sources','Full source view archive')
    archived_jobs={str(j['qa_id']):j for j in old['jobs']}
    results={r['search_id']:r for r in json_records(run/'completed/data/results/personamem-v2-32k.jsonl')}
    same({str(r['qa_id']) for r in results.values()},set(archived_jobs),'Complete question identities')
    snapshot=run.parent
    plan=read(snapshot/'plan.json')
    for name,digest in plan['source_files'].items():
        same(file_digest(snapshot/'source'/name),digest,'Frozen current runtime')
    sys.path.insert(0,str(snapshot/'source/memory_system'))
    from app import answer_context,config,cross_encoder
    from app.conversation_context import source_key,usable
    from app.fine_context_reserve import full_user
    # Execute only the pure source-view construction function, not legacy I/O.
    source=Path(__file__).with_name('probe_source_ce.py')
    tree=ast.parse(source.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_views']
    require(len(functions)==1,'Unique pure view builder')
    env=dict(hashlib=hashlib,copy=copy,config=config,cross_encoder=cross_encoder,
             answer_context=answer_context,source_key=source_key,usable=usable,full_user=full_user)
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(source),'exec'),env)
    build_views=env['build_views']
    rows,seen=[],set()
    with patch('socket.socket.connect',side_effect=AssertionError('Offline compatibility inventory')):
        for trace in json_records(run/'server-search-debug.jsonl'):
            result=results[trace['search_id']]
            qa=str(result['qa_id']);require(qa not in seen,'Duplicate trace');seen.add(qa)
            same(result['packet_hash'],trace['packet_hash'],'Current packet identity')
            old_job=archived_jobs[qa]
            pool={r['id']:r for r in trace['ranked']}
            coarse=[pool[mid] for mid in trace['cascade']['coarse']['selected_ids']]
            query=trace['cascade']['cross_encoder']['query']
            with patch.multiple(config,**trace['versions']['settings']):
                views,members,batches,omitted=build_views(coarse,trace['user_id'],query,max_sources=None)
                same(config.CE_MODEL,old['ce_model'],'Archived and current CE model')
                same(trace['cascade']['cross_encoder']['model'],old['ce_model'],'Recorded current CE model')
            old_views={v['id']:v for v in old_job['views']}
            old_texts={v['text'] for v in old_job['views']}
            old_batches=[tuple(old_views[mid]['text'] for mid in batch) for batch in old_job['batches']]
            query_equal=query==old_job['query']
            current_batches=[]
            for i,batch in enumerate(batches):
                texts=tuple(v['text'] for v in batch)
                matching=[j for j,old_batch in enumerate(old_batches) if query_equal and texts==old_batch]
                current_batches.append(dict(batch=i,documents=len(texts),matching_archive_batches=matching))
            missing=[dict(view_id=key,text_sha256=hashlib.sha256(v['text'].encode()).hexdigest())
                     for key,v in views.items() if v['text'] not in old_texts]
            rows.append(dict(qa_id=qa,query_equal=query_equal,
                current_query_sha256=hashlib.sha256(query.encode()).hexdigest(),
                archive_query_sha256=hashlib.sha256(old_job['query'].encode()).hexdigest(),
                current_views=len(views),archive_views=len(old_views),
                matching_view_texts=len(views)-len(missing),missing_views=missing,batches=current_batches,
                candidate_members=members,omitted=omitted,
                ce_api_format=trace['versions']['settings']['CE_API_FORMAT']))
    same(seen,set(archived_jobs),'All current and archive questions inspected')
    summary=dict(questions=len(rows),exact_queries=sum(r['query_equal'] for r in rows),
        current_views=sum(r['current_views'] for r in rows),
        exact_view_texts=sum(r['matching_view_texts'] for r in rows),
        questions_with_all_view_texts=sum(not r['missing_views'] for r in rows),
        current_batches=sum(len(r['batches']) for r in rows),
        exact_ordered_batches=sum(bool(b['matching_archive_batches']) for r in rows for b in r['batches']),
        fully_batch_matched_questions=sum(all(b['matching_archive_batches'] for b in r['batches']) for r in rows),
        prepared_data_hash_equal=old['prepared_sha256']==plan['data_sha256'],provider_calls=0)
    inputs=[Path(__file__),source,archive/'audit.json',experiment/'plan.json',snapshot/'plan.json',
            run/'server-search-debug.jsonl',run/'completed/data/results/personamem-v2-32k.jsonl',*archive_inputs]
    report=dict(status='complete_source_ce_input_compatibility_inventory',cohort=args.cohort,
        summary=summary,rows=rows,provider_calls=0,runtime_changed=False,provider_wire_parity_verified=False,
        limitation=__doc__,input_sha256={str(p):file_digest(p) for p in inputs})
    with args.output.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(cohort=args.cohort,**summary)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--cohort',choices=('regression','validation'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
