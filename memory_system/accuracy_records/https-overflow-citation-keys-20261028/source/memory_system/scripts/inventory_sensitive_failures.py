"""Offline benchmark error attribution; sensitive labels never enter runtime."""
import csv,json,sys,hashlib
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from replay_query_context import records
jobs=json.loads(Path('runs/https-cycle-20261011-constraint-term-forms-jobs.json').read_text())+[dict(name='cohort29_partial',run='runs/https-cycle-20261011-cohort29-control',prepared='runs/https-cycle-20261011-cohort29-input.jsonl')]
gold=list(csv.DictReader(open('data/raw/personamem-v2.csv')));rows=[];total=0
for job in jobs:
 qs={str(q['id']):q for c in records(Path(job['prepared'])) for q in c['qa']}
 for r in records(Path(job['run'])/'completed/data/results/personamem-v2-32k.jsonl'):
  total+=1;meta=gold[int(r['qa_id'])]
  if meta['sensitive_info'].lower()!='true':continue
  q=qs[r['qa_id']];label=q['gold_labels'][0];assert meta['correct_answer']==q['answer']
  alignment=next((x for x in r['choice_alignment'] if x['letter']==label),{})
  rows.append(dict(cohort=job['name'],qa_id=r['qa_id'],score=r['score'],prediction=r['prediction'],gold=label,gold_eligible=label in r['answer_eligible_options'],gold_status=alignment.get('status'),gold_claims=[dict(text=x['text'],status=x['status'],errors=x['validation_errors']) for x in alignment.get('claims',[])],sensitive_type=meta['preference'].split(':',1)[0],question_sha256=hashlib.sha256(q['question'].encode()).hexdigest()))
summary=dict(questions=total,sensitive_count=len(rows),sensitive_score=sum(r['score'] for r in rows),wrong_eligible=sum(not r['score'] and r['gold_eligible'] for r in rows),wrong_ineligible=sum(not r['score'] and not r['gold_eligible'] for r in rows),types=dict(Counter(r['sensitive_type'] for r in rows)))
out=dict(summary=summary,rows=rows,caveat='Offline benchmark labels only. Partial cohort29 snapshot; not a runtime rule or a claim that all reproduction of user-supplied identifiers is disallowed. A missing historical snippet is not by itself the causal error when the current request supplies the information.')
Path('runs/https-cycle-20261011-sensitive-failure-inventory.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary))
