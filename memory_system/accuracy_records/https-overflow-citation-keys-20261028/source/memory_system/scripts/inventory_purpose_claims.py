"""Inventory literal purpose-clause extracted claims; no scope changes or provider calls."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import choice_premises,config
from unittest.mock import patch
from replay_query_context import records
jobs=json.loads(Path('runs/https-cycle-20261011-constraint-term-forms-jobs.json').read_text())
jobs += [dict(name='cohort29_partial',run='runs/https-cycle-20261011-cohort29-control',prepared='runs/https-cycle-20261011-cohort29-input.jsonl')]
rows=[];population=[];invalid=[]
for j in jobs:
 qa={str(q['id']):q for c in records(Path(j['prepared'])) for q in c['qa']}
 for r in records(Path(j['run'])/'completed/data/results/personamem-v2-32k.jsonl'):
  population.append(dict(cohort=j['name'],qa_id=r['qa_id']))
  options={chr(65+i):o for i,o in enumerate(qa[r['qa_id']]['options'])}
  for call in r['answer_calls']:
   if call['stage']!='eval.choice_support':continue
   payload=call.get('response') or {}
   if not isinstance(payload,dict) or not isinstance(payload.get('options'),dict):
    invalid.append(dict(cohort=j['name'],qa_id=r['qa_id'],reason='malformed_support_payload'));continue
   for letter,entry in payload['options'].items():
    if letter not in options or not isinstance(entry,dict) or not isinstance(entry.get('claims'),list):
     invalid.append(dict(cohort=j['name'],qa_id=r['qa_id'],letter=letter,reason='malformed_option_entry'));continue
    option=options[letter]
    for claim in entry.get('claims',[]):
     if not isinstance(claim,dict) or not isinstance(claim.get('text'),str):
      invalid.append(dict(cohort=j['name'],qa_id=r['qa_id'],letter=letter,claim=claim));continue
     text=claim['text'];hits=list(re.finditer(re.escape(text),option,re.I))
     if len(hits)!=1:continue
     hit=hits[0];prefix=option[:hit.start()]
     if not re.search(r'\bso\s+(?:that\s+)?$',prefix,re.I):continue
     with patch.multiple(config,**r['versions']['settings']):suggestion=choice_premises.is_suggestion(text,option)
     alignment=next((x for x in r.get('choice_alignment',[]) if x['letter']==letter),{})
     retained=[x for x in alignment.get('claims',[]) if x['text']==text]
     removed=[x for x in alignment.get('removed_claims',[]) if x['text']==text]
     rows.append(dict(final_claim_statuses=[x.get('status') for x in retained],removed_reasons=[x.get('reason') for x in removed],eligible=letter in r.get('answer_eligible_options',[]),cohort=j['name'],qa_id=r['qa_id'],letter=letter,claim=text,option=option,original_suggestion=suggestion,score=r['score']))
out=dict(invalid_claim_records=invalid,population=population,questions=len(population),candidates=rows,caveat='Literal immediate so/so-that prefix only. Inventory not exhaustive, no semantic decision, partial cohort29 snapshot. Does not modify runtime or claim accuracy.')
Path('runs/https-cycle-20261011-purpose-claims-inventory.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(questions=len(population),candidates=len(rows),unrecognized=sum(not r['original_suggestion'] for r in rows))))
