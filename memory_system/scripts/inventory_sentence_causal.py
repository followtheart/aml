import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));from app.choice_premises import asserted_causal_prefix
base=Path('runs');stem='https-cycle-20261011-';jobs=json.loads((base/(stem+'constraint-term-forms-jobs.json')).read_text())
for n in range(29,33):jobs.append(dict(name=f'cohort{n}',run=f'runs/{stem}cohort{n}-'+('control' if n<31 else 'term-forms')))
rows=[];population=[]
for job in jobs:
 path=Path(job['run'])/'completed/data/results/personamem-v2-32k.jsonl'
 for line in path.open():
  r=json.loads(line);population.append((job['name'],r['qa_id']))
  for entry in r.get('choice_alignment',[]):
   option=entry['option']
   for match in re.finditer(r'(?<=[.!?])\s+(?=(?:Since|Because|Given)\b)',option):
    start=match.end();suffix=option[start:]
    for removed in entry.get('removed_claims',[]):
     claim=removed['text'];target=dict(option=suffix,claim=claim)
     if removed.get('reason')=='scope_not_personal' and asserted_causal_prefix(target) and not asserted_causal_prefix(dict(option=option,claim=claim)):
      rows.append(dict(cohort=job['name'],qa_id=r['qa_id'],letter=entry['letter'],claim=claim,option=option,offset=start,eligible=entry['letter'] in r.get('answer_eligible_options',[]),score=r['score'],kind=entry['kind']))
assert len(population)==len(set(population))
out=dict(questions=len(population),population=population,removed_sentence_causal_claims=rows,caveat='Read-only inventory: existing opening guard applied to suffix after punctuation. Not a safe sentence parser; quotes, abbreviations, conditionals, generated templates and universal background claims require boundary review. No runtime behavior changed.')
(base/(stem+'sentence-causal-inventory.json')).open('x').write(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(questions=len(population),matches=len(rows)),ensure_ascii=False))
for r in rows:print(json.dumps({k:r[k] for k in ['cohort','qa_id','letter','claim','kind']},ensure_ascii=False))
