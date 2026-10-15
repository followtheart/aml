"""Read-only original first/final generic-selection review counterfactual inventory."""
import json
from pathlib import Path
b=Path('runs');stem='https-cycle-20261011-';jobs=json.loads((b/(stem+'constraint-term-forms-jobs.json')).read_text())
for n in range(29,37):jobs.append({'name':f'cohort{n}','run':f'runs/{stem}cohort{n}-'+('control' if n<31 else 'term-forms-retry' if n==35 else 'term-forms'),'prepared':f'runs/{stem}cohort{n}-input.jsonl'})
rows=[];total=0;score=0
for job in jobs:
 qs={str(q['id']):q for l in Path(job['prepared']).open() for q in json.loads(l)['qa']}
 result_path=Path(job['run'])/'completed/data/results/personamem-v2-32k.jsonl'
 archived=[json.loads(line) for line in result_path.open()]
 assert len(archived)==len(qs) and {r['qa_id'] for r in archived}==set(qs), 'Require full unique cohort'
 assert (Path(job['run'])/'exit-code').read_text().strip()=='0'
 for r in archived:
  assert r['score']==float(r['prediction'] in qs[r['qa_id']]['gold_labels'])
  total+=1;score+=r['score'];eligible=r['answer_eligible_options'];entries=[e for e in r['choice_alignment'] if e['letter'] in eligible]
  if not entries or any(e['kind']!='generic' or e['status']!='generic' for e in entries):continue
  first=[c for c in r['answer_calls'] if c['stage'] in ['eval.choice_select','eval.choice_select.repair'] and c.get('status')=='ok' and set(c.get('response',{}))=={'answer'}]
  review=[c for c in r['answer_calls'] if c['stage']=='eval.choice_select_review' and c.get('status')=='ok' and set(c.get('response',{}))=={'answer'}]
  if not first or not review:continue
  initial=first[-1]['response']['answer'];final=review[-1]['response']['answer'];assert initial in eligible and final==r['prediction'] and final in eligible
  counter=float(initial in qs[r['qa_id']]['gold_labels']);rows.append({'cohort':job['name'],'qa_id':r['qa_id'],'first':initial,'final':final,'score_with_review':r['score'],'score_without_review':counter})
report={'status':'read_only_counterfactual_inventory','questions':total,'baseline_score':score,'all_generic_reviewed':len(rows),'without_generic_review_score':score+sum(r['score_without_review']-r['score_with_review'] for r in rows),'review_benefits':[r for r in rows if r['score_with_review']>r['score_without_review']],'review_regressions':[r for r in rows if r['score_with_review']<r['score_without_review']],'rows':rows,'caveat':'First/final selection outputs from original traces; all eligible options generic. Not an executed modified runtime or fresh HTTPS score. No change to retrieval, source, support or defaults.'}
(b/(stem+'generic-review-counterfactual-inventory.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['rows','caveat']},indent=2))
