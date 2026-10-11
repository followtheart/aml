"""Offline route-capacity coverage; no gold or target IDs enter retrieval ranking."""
import argparse,ast,csv,hashlib,json,sqlite3,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.search_pipeline import _merge_query_results
from audit_https_funnel import norm
from inventory_snippet_provenance import parse_snippet

def analyze(probe,run,capacity):
 data=json.loads(probe.read_text());assert all(r['archived_route_exact'] for r in data['rows'])
 original=[json.loads(l) for l in (run/'completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines()]
 assert [r['qa_id'] for r in original]==[r['qa_id'] for r in data['rows']]
 assert capacity>data['rows'][0]['limit']
 manifest=json.loads((run/'isolated-manifest.json').read_text())
 questions={q['id']:q for line in Path(manifest['data']).read_text().splitlines() for q in json.loads(line)['qa']}
 assert set(questions)=={r['qa_id'] for r in original}
 meta=list(csv.DictReader(open('data/raw/personamem-v2.csv')))
 db=sqlite3.connect(f'file:{(run/"memory.db").resolve()}?mode=ro',uri=True)
 memories=[(mid,norm(content)) for mid,content in db.execute('SELECT id,content FROM amu')]
 sources=[norm(content) for role,content in db.execute('SELECT role,content FROM source_messages') if role=='user'];db.close()
 traces={}
 for line in (run/'server-search-debug.jsonl').open():
  r=json.loads(line);traces[r['search_id']]={x['id'] for x in r['fused']}
 rows=[]
 for p,r in zip(data['rows'],original):
  metadata=meta[int(r['qa_id'])];question=questions[r['qa_id']]
  assert metadata['correct_answer']==question['answer']
  assert norm(ast.literal_eval(metadata['user_query'])['content'])==norm(question['question'])
  turns=[norm(m['content']) for m in parse_snippet(meta[int(r['qa_id'])]['related_conversation_snippet']) if m['role']=='user']
  valid=[t for t in turns if t and any(t in source for source in sources)]
  carriers={mid for mid,content in memories if any(t in content for t in valid)}
  routes=[[dict(id=mid,score=score) for mid,score in ranking[:capacity]] for ranking in p['rankings']]
  expanded=[x['id'] for x in _merge_query_results(routes,capacity)]
  baseline=set(p['merged_ids']);extra=set(expanded)
  rows.append(dict(qa_id=r['qa_id'],score=r['score'],raw_carrier_ids=sorted(carriers),baseline_vector=sorted(carriers&baseline),expanded_vector=sorted(carriers&extra),baseline_fused=sorted(carriers&traces[r['search_id']]),expanded_ids=expanded))
 return dict(questions=len(rows),capacity=capacity,baseline_capacity=data['rows'][0]['limit'],archived_vector_routes_exact=len(rows),coverage={k:sum(bool(r[k]) for r in rows) for k in ['baseline_vector','expanded_vector','baseline_fused']},new_beyond_fused=[r['qa_id'] for r in rows if r['expanded_vector'] and not r['baseline_fused']],rows=rows,probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),caveat='Literal original user turn in AMU raw content, not semantic support or rendered-rank visibility. Expanded dense route only; full fusion/coarse/fine/listwise/reader not evaluated. Other channels may already retrieve the same carrier. Offline labels never enter rankings.')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--capacity',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=analyze(a.probe,a.run,a.capacity)
 with a.output.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in r.items() if k not in ['rows','caveat']}))
