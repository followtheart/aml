import copy,json
from pathlib import Path
from probe_boolean_verifier import validate,checks_in,normalize
checks=[dict(claim_id='A:0')]
for a,b,label in [(True,True,'personal_supported'),(True,False,'unsupported_personal'),(False,True,'no_personal_premise')]:
 assert validate({'checks':[dict(claim_id='A:0',asserts_personal_fact=a,all_personal_facts_supported=b)]},checks,'booleans')==[dict(claim_id='A:0',verdict=label)]
for a,b in [(False,False),(1,True),(True,'false')]:
 try:validate({'checks':[dict(claim_id='A:0',asserts_personal_fact=a,all_personal_facts_supported=b)]},checks,'booleans')
 except ValueError:pass
 else:raise AssertionError((a,b))
plan=json.loads(Path('runs/https-cycle-20261011-verifier-assessment-cohort25.plan.json').read_text())
for c in plan['cases']:assert checks_in(c['prompt'])==checks_in(normalize(c['prompt']))
print(json.dumps(dict(status='passed',valid_encodings=3,invalid_encodings_rejected=3,unchanged_check_batches=len(plan['cases']))))
