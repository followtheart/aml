"""Read-only final-claim causal relation inventory, never a semantic verdict."""
import json,re
from pathlib import Path
p=Path('runs');stem='https-cycle-20261011-';jobs=json.loads((p/(stem+'constraint-term-forms-jobs.json')).read_text())
for n in range(29,35):jobs.append(dict(name=f'cohort{n}',run=f'runs/{stem}cohort{n}-'+('control' if n<31 else 'term-forms')))
pattern=re.compile(r'\b(?:rooted in|stems? from|driven by|due to|because of|trigger(?:ed|s)? by)\b',re.I)
rows=[];population=[]
for job in jobs:
 path=Path(job['run'])/'completed/data/results/personamem-v2-32k.jsonl'
 lines=path.read_text().splitlines()
 for i,line in enumerate(lines):
  if not line.strip():continue
  try:r=json.loads(line)
  except json.JSONDecodeError:
   if i==len(lines)-1:break
   raise
  population.append(dict(cohort=job['name'],qa_id=r['qa_id']))
  for entry in r.get('choice_alignment',[]):
   for claim in entry.get('claims',[]):
    if pattern.search(claim['text']):rows.append(dict(cohort=job['name'],qa_id=r['qa_id'],letter=entry['letter'],claim=claim,option=entry['option'],prediction=r['prediction'],score=r['score'],eligible=entry['letter'] in r.get('answer_eligible_options',[])))
assert len(population)==len({(r['cohort'],r['qa_id']) for r in population})
report=dict(questions=len(population),population=population,rows=rows,caveat='Lexical final-claim inventory only; presence of causal wording does not imply a false claim. Earlier unsupported or omitted causal claims outside pattern are not exhaustively counted. Cohort34 is a live partial snapshot. No prompts or runtime modified.')
(p/(stem+'causal-relations-inventory.json')).open('x').write(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(questions=len(population),claims=len(rows))))
for r in rows:print(json.dumps(dict(qa_id=r['qa_id'],letter=r['letter'],text=r['claim']['text'],status=r['claim']['status']),ensure_ascii=False))
