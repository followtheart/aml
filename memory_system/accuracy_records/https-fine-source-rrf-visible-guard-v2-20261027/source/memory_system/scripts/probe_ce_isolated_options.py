"""Full-cohort CE diagnostic: one option per query, original evidence fixed."""
import argparse,asyncio,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget,config,rerank_query
from probe_ce_source_units import score_documents,records

def queries_for(trace):
    result={'control':trace['cascade']['cross_encoder']['query']}
    for i,option in enumerate(trace['options']):
        suffix='\n\nThis single alternative answer is a hypothesis, not an established user fact. Rank original evidence relevant to checking its personal premises:\n'+option
        if len(suffix.encode())>rerank_query.MAX_AUXILIARY_BYTES:raise ValueError('Option exceeds auxiliary bound')
        result[f'option_{i}']=trace['query']+suffix
    return result

def round_robin(orders):
    merged=[];seen=set()
    for rank in range(max(map(len,orders),default=0)):
        for order in orders:
            if rank<len(order) and order[rank] not in seen:
                seen.add(order[rank]);merged.append(order[rank])
    return merged

async def main(a):
    if config.FAKE:raise ValueError('Real CE required')
    qa=[q for c in records(a.prepared) for q in c['qa']];expected=[str(q['id']) for q in qa]
    results=list(records(a.run/'completed/data/results/personamem-v2-32k.jsonl'))
    assert [r['qa_id'] for r in results]==expected,'Require complete ordered cohort'
    by_search={r['search_id']:r for r in results};by_qa={str(q['id']):q for q in qa}
    plan={'run':str(a.run),'prepared':str(a.prepared),'prepared_sha256':hashlib.sha256(a.prepared.read_bytes()).hexdigest(),'qa_ids':expected,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'method':'Single raw question plus one complete alternative only. Frozen ordered coarse pool and original rank text. Round-robin per-option ranks, option starting position rotates by question index.','caveat':'CE-only diagnostic with expanded scoring budget. No retrieval/listwise/reader score claim; no gold supplied.'}
    with a.output.with_suffix('.plan.json').open('x') as f:json.dump(plan,f,indent=2)
    completed=[]
    with a.output.open('x') as out:
        for trace in records(a.run/'server-search-debug.jsonl'):
            if trace.get('search_id') not in by_search:continue
            r=by_search[trace['search_id']];q=by_qa[r['qa_id']]
            assert trace['query']==q['question'] and trace['options']==q['options']
            ids=trace['cascade']['coarse']['selected_ids'];pool={c['id']:c for c in trace['ranked']};docs=[pool[i].get('_rank_text',pool[i]['content']) for i in ids]
            queries=queries_for(trace);arms=[]
            for name,query in queries.items():
                with budget.scope(seconds=240,calls=100,tokens=1000000) as usage:scores,batches=await score_documents(query,docs)
                arms.append({'name':name,'query':query,'scores':scores,'order':[ids[i] for i in sorted(range(len(ids)),key=lambda i:-scores[i])],'calls':usage.calls,'tokens':usage.tokens,'batches':batches})
            order_inputs=[arm['order'] for arm in arms[1:]];offset=expected.index(r['qa_id'])%len(order_inputs);order_inputs=order_inputs[offset:]+order_inputs[:offset]
            row={'qa_id':r['qa_id'],'search_id':r['search_id'],'candidate_ids':ids,'document_hashes':[hashlib.sha256(d.encode()).hexdigest() for d in docs],'arms':arms,'option_offset':offset,'aggregate_order':round_robin(order_inputs)}
            out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();completed.append(r['qa_id']);print(json.dumps({'qa_id':r['qa_id'],'calls':sum(x['calls'] for x in arms)}),flush=True)
    assert completed==expected
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['run','prepared','output']:p.add_argument('--'+k,type=Path,required=True)
    asyncio.run(main(p.parse_args()))
