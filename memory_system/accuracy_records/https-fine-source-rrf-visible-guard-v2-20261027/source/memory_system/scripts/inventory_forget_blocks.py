"""Offline inventory of gold options blocked by recorded forget constraints."""
import json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_https_funnel import gold_forget_matches
from replay_query_context import records
jobs=json.loads(Path('runs/https-cycle-20261011-constraint-term-forms-jobs.json').read_text())+[dict(name='cohort29',run='runs/https-cycle-20261011-cohort29-control',prepared='runs/https-cycle-20261011-cohort29-input.jsonl')]
rows=[];count=0;wrong=0
for j in jobs:
 qs={str(q['id']):q for c in records(Path(j['prepared'])) for q in c['qa']}
 for r in records(Path(j['run'])/'completed/data/results/personamem-v2-32k.jsonl'):
  count+=1;wrong+=int(r['score']<1);q=qs[r['qa_id']];label=q['gold_labels'][0];matches=gold_forget_matches(r,label)
  if label not in r.get('answer_blocked_options',[]) or not matches:continue
  alignment=next((x for x in r['choice_alignment'] if x['letter']==label),{})
  rows.append(dict(cohort=j['name'],qa_id=r['qa_id'],score=r['score'],gold=label,prediction=r['prediction'],question=q['question'],gold_option=q['options'][ord(label)-65],gold_kind=alignment.get('kind'),gold_status=alignment.get('status'),matches=matches,other_errors=alignment.get('validation_errors',[])))
out=dict(questions=count,wrong_answers=wrong,gold_forget_blocks=len(rows),by_cohort=dict(Counter(x['cohort'] for x in rows)),rows=rows,caveat='Offline label-based diagnosis only. A recorded block is not proof of error, and removing it need not produce the gold answer. Current-question context and actual action/object must be reviewed.')
Path('runs/https-cycle-20261011-forget-block-inventory.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['rows','caveat']}))
