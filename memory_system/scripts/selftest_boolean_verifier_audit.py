import copy,json,tempfile
from pathlib import Path
from probe_boolean_verifier import SUFFIX,schema,validate,checks_in
from probe_question_source_filter import sha
from summarize_boolean_verifier import summarize
p=json.loads(Path('runs/https-cycle-20261011-boolean-verifier-cohort25.plan.json').read_text());rows=[]
for i,c in enumerate(p['cases']):
 checks=checks_in(c['prompt'])
 for a in (p['arms'] if i%2==0 else list(reversed(p['arms']))):
  response={'checks':[dict(claim_id=x['claim_id'],**(dict(verdict='unsupported_personal') if a=='labels' else dict(asserts_personal_fact=True,all_personal_facts_supported=False))) for x in checks]};contract=schema(checks) if a=='booleans' else c['schema']
  rows.append(dict(qa_id=c['qa_id'],arm=a,prompt_sha256=sha(c['normalized_prompt']+(SUFFIX if a=='booleans' else '')),schema_sha256=sha(json.dumps(contract,sort_keys=True)),response=response,verdicts=validate(response,checks,a),status='ok',calls=0,tokens=0))
rows[0].pop('response');rows[0].pop('verdicts');rows[0].update(status='error',error_type='TimeoutError')
rows[1]['response']['checks'][0].update(asserts_personal_fact=False,all_personal_facts_supported=False);rows[1].pop('verdicts');rows[1].update(status='error',error_type='ValueError')
with tempfile.TemporaryDirectory() as tmp:
 path=Path(tmp)/'mock.jsonl';path.with_suffix('.plan.json').write_text(json.dumps(p))
 def save(r):path.write_text(''.join(json.dumps(x)+'\n' for x in r))
 save(rows);r=summarize(path);assert r['complete_pairs']==28 and len(r['errors'])==2
 rejected=[]
 for k in ['prompt','schema','verdict','order','incomplete']:
  bad=copy.deepcopy(rows)
  if k=='prompt':bad[2]['prompt_sha256']='bad'
  if k=='schema':bad[2]['schema_sha256']='bad'
  if k=='verdict':bad[2]['verdicts'][0]['verdict']='personal_supported'
  if k=='order':bad[2],bad[3]=bad[3],bad[2]
  if k=='incomplete':bad.pop()
  save(bad)
  try:summarize(path)
  except ValueError:rejected.append(k)
  else:raise AssertionError(k)
print(json.dumps(dict(status='passed',complete_pairs=28,preserved_failures=2,rejected=rejected)))
