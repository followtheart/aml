"""Offline inventory of answer-speaker first-person claims, without relabeling."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from replay_query_context import records
jobs=json.loads(Path('runs/https-cycle-20261011-constraint-term-forms-jobs.json').read_text())+[dict(name=f'cohort{n}',run=f'runs/https-cycle-20261011-cohort{n}-control',prepared=f'runs/https-cycle-20261011-cohort{n}-input.jsonl') for n in [29,30]]
rows=[];population=[]
for job in jobs:
 for r in records(Path(job['run'])/'completed/data/results/personamem-v2-32k.jsonl'):
  population.append(dict(cohort=job['name'],qa_id=r['qa_id']))
  for alignment in r.get('choice_alignment',[]):
   for removed,claims in [(False,alignment.get('claims',[])),(True,alignment.get('removed_claims',[]))]:
    for c in claims:
     text=c.get('text','')
     if not re.match(r"^I(?:\s|['’])",text):continue
     rows.append(dict(cohort=job['name'],qa_id=r['qa_id'],letter=alignment['letter'],claim=text,option=alignment['option'],removed=removed,status=c.get('status'),removal_reason=c.get('reason') if removed else None,eligible=alignment['letter'] in r.get('answer_eligible_options',[])))
out=dict(population=population,questions=len(population),claims=rows,caveat='Answer text first-person prefix only; speaker identity and embedded user facts require context. Cohort30 is a partial snapshot, no semantic relabeling or provider calls.')
Path('runs/https-cycle-20261011-answer-first-person-inventory.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(questions=len(population),claims=len(rows),retained=sum(not r['removed'] for r in rows))))
