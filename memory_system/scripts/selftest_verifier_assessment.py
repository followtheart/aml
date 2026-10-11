import copy,json
from probe_verifier_assessment import validate,assessment_schema
checks=[dict(claim_id='C:0',sources=[dict(id='s4')],context_neighbors=[dict(id='s5')])]
valid={'checks':[dict(claim_id='C:0',evidence_assessment='User asked about this topic.',source_ids=['s4'],verdict='personal_supported')]}
assert validate(valid,checks,'assessment')==[dict(claim_id='C:0',verdict='personal_supported')]
rejected=[]
for kind in ['unknown_source','duplicate_source','long_assessment','missing_check','duplicate_check','unknown_verdict','extra_field']:
 r=copy.deepcopy(valid);c=r['checks'][0]
 if kind=='unknown_source':c['source_ids']=['s99']
 if kind=='duplicate_source':c['source_ids']=['s4','s4']
 if kind=='long_assessment':c['evidence_assessment']='x'*181
 if kind=='missing_check':r['checks']=[]
 if kind=='duplicate_check':r['checks'].append(copy.deepcopy(c))
 if kind=='unknown_verdict':c['verdict']='yes'
 if kind=='extra_field':c['answer']='C'
 try:validate(r,checks,'assessment')
 except ValueError:rejected.append(kind)
 else:raise AssertionError(kind)
assert assessment_schema(checks)['properties']['checks']['maxItems']==1
print(json.dumps(dict(status='passed',rejected=rejected)))
