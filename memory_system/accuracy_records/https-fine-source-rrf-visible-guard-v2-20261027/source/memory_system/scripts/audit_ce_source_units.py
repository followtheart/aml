"""Validate CE unit probes against archived text and report rank-only transport."""
import argparse, ast, csv, hashlib, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context
from audit_https_funnel import norm
from inventory_snippet_provenance import parse_snippet

def records(path):
    for line in path.open():
        if line.strip(): yield json.loads(line)

def audit(run, prepared, probe, limit):
    qa = {str(q['id']): q for conv in records(prepared) for q in conv['qa']}
    results = {str(r['qa_id']): r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
    probes = list(records(probe))
    assert len(probes) == len(qa) and {p['qa_id'] for p in probes} == set(qa) == set(results), 'Require complete unique cohort'
    by_search = {r['search_id']: q for q,r in results.items()}
    by_qa = {p['qa_id']: p for p in probes}
    gold = list(csv.DictReader(Path('data/raw/personamem-v2.csv').open()))
    rows = []
    for trace in records(run/'server-search-debug.jsonl'):
        qid = by_search.get(trace.get('search_id'))
        if qid is None: continue
        p = by_qa[qid]; pool = {c['id']: c for c in trace['ranked']}
        ids = trace['cascade']['coarse']['selected_ids']; candidates = [pool[i] for i in ids]
        assert p['query'] == trace['cascade']['cross_encoder']['query'] and p['query_plan'] is None
        assert [r['id'] for r in p['candidates']] == ids
        assert p['whole_documents'] == [c.get('_rank_text',c['content']) for c in candidates]
        docs, lookup, mappings = [], {}, []
        for c, text in zip(candidates,p['whole_documents']):
            refs = []
            for source in (c.get('_packet_item') or {}).get('sources',[]):
                content = source.get('content')
                if not content: continue
                assert content in text
                document = answer_context.with_evidence('',[source])
                if document not in lookup: lookup[document]=len(docs); docs.append(document)
                refs.append(dict(index=lookup[document],role=source.get('role'),request_id=source.get('request_id'),message_index=source.get('message_index'),content_span=source.get('content_span')))
            mappings.append(refs)
        assert docs == p['source_documents'] and len(docs) == len(p['source_scores'])
        assert all(isinstance(v,(float,int)) and math.isfinite(v) for v in p['source_scores'])
        for c,refs in zip(p['candidates'],mappings):
            assert c['refs'] == refs and math.isfinite(c['whole_score'])
            for mode,selected in [('source_max',refs),('user_source_max',[r for r in refs if r['role']=='user'])]:
                assert c[mode] == max((p['source_scores'][r['index']] for r in selected),default=c['whole_score'])
        for mode,order in p['orders'].items():
            assert order == [r['id'] for r in sorted(p['candidates'],key=lambda r:-r[mode])]
        meta = gold[int(qid)]
        assert norm(ast.literal_eval(meta['user_query'])['content']) == norm(qa[qid]['question'])
        turns = [norm(s['content']) for s in parse_snippet(meta['related_conversation_snippet']) if s['role']=='user']
        carriers = [c['id'] for c in candidates if any(t and t in norm(s.get('content','')) for t in turns for s in (c.get('_packet_item') or {}).get('sources',[]) if s.get('role')=='user')]
        ranks = {mode:min((order.index(i)+1 for i in carriers),default=None) for mode,order in p['orders'].items()}
        rows.append(dict(qa_id=qid,score=results[qid]['score'],carriers=carriers,best_user_turn_rank=ranks,top_k_has_turn={m:r is not None and r<=limit for m,r in ranks.items()},changed_top_k=p['orders']['whole_score'][:limit]!=p['orders']['user_source_max'][:limit]))
    assert len(rows)==len(qa), 'Missing/duplicate archived searches'
    return dict(status='passed',questions=len(rows),probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),calls=sum(p['calls'] for p in probes),rank_limit=limit,coverage={m:sum(r['top_k_has_turn'][m] for r in rows) for m in ['whole_score','source_max','user_source_max']},user_max_gains=[r['qa_id'] for r in rows if r['top_k_has_turn']['user_source_max'] and not r['top_k_has_turn']['whole_score']],user_max_losses=[r['qa_id'] for r in rows if r['top_k_has_turn']['whole_score'] and not r['top_k_has_turn']['user_source_max']],rows=rows,caveat='Provider scores are archived observations, not independently reproduced by this audit. Exact query, ordered candidate/document identity, roles, mappings, aggregation and full population checked. Rank-only top-k ignores fine reservations, prompt admission and reader gates; source turn containment is not entailment. User max falls back to full unit for parents without user references. No accuracy claim.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','prepared','probe','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--limit',type=int,default=16);a=p.parse_args()
    result=audit(a.run,a.prepared,a.probe,a.limit)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','caveat']}))
