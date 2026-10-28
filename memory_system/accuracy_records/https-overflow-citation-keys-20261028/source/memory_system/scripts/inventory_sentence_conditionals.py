"""Read-only inventory of later-sentence conditional interest claims."""
import copy,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac
p=Path('runs');stem='https-cycle-20261011-';jobs=json.loads((p/(stem+'constraint-term-forms-jobs.json')).read_text())
for n in range(29,35):jobs.append(dict(name=f'cohort{n}',run=f'runs/{stem}cohort{n}-'+('control' if n<31 else 'term-forms')))
rows=[];population=[]
for job in jobs:
 lines=(Path(job['run'])/'completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines()
 for i,line in enumerate(lines):
  if not line.strip():continue
  try:r=json.loads(line)
  except json.JSONDecodeError:
   if i==len(lines)-1:break
   raise
  population.append(dict(cohort=job['name'],qa_id=r['qa_id']))
  for entry in r.get('choice_alignment',[]):
   option=entry['option']
   for claim in entry.get('claims',[]):
    if ac._nonasserted_scope_reason(claim,option):continue
    for boundary in re.finditer(r'(?<=[.!?])\s+(?=If\s+you\b)',option):
     start=boundary.end();condition=ac._CONDITIONAL_INTEREST_OPEN.match(option[start:]);span=claim.get('option_span') or {}
     if (not condition or ac._CONDITIONAL_PRESUPPOSED.search(condition.group()) or type(span.get('start')) is not int or type(span.get('end')) is not int or span['start']<start or span['end']>start+condition.end()):continue
     shifted=copy.deepcopy(claim);shifted['option_span']={'start':span['start']-start,'end':span['end']-start}
     if ac._nonasserted_scope_reason(shifted,option[start:])!='conditional_interest_not_asserted':continue
     rows.append(dict(cohort=job['name'],qa_id=r['qa_id'],letter=entry['letter'],claim=claim,option=option,offset=start,score=r['score'],eligible=entry['letter'] in r.get('answer_eligible_options',[])))
report=dict(questions=len(population),population=population,rows=rows,caveat='Raw punctuation boundary inventory only, not an approved sentence parser. Quote/abbreviation/conditional-parent scope needs validation. Existing nested fact exclusions retained; no runtime or answer changes. Cohort34 may be partial.')
(p/(stem+'sentence-conditionals-inventory.json')).open('x').write(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(questions=len(population),claims=len(rows))))
for r in rows:print(json.dumps({k:r[k] for k in ['qa_id','letter','claim']},ensure_ascii=False))
