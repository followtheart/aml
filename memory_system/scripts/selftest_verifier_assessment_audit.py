"""Mock complete reports and reject corrupted prompts, verdicts and identities."""
import copy,json,tempfile
from pathlib import Path
from probe_verifier_assessment import SUFFIX,checks_in,assessment_schema,validate
from probe_question_source_filter import sha
from summarize_verifier_assessment import summarize
plan=json.loads(Path('runs/https-cycle-20261011-verifier-assessment-cohort25.plan.json').read_text());rows=[]
for i,c in enumerate(plan['cases']):
 checks=checks_in(c['prompt'])
 for arm in (plan['arms'] if i%2==0 else list(reversed(plan['arms']))):
  schema=assessment_schema(checks) if arm=='assessment' else c['schema'];response={'checks':[dict(claim_id=x['claim_id'],verdict='unsupported_personal',**(dict(evidence_assessment='No assessment in synthetic test.',source_ids=[]) if arm=='assessment' else {})) for x in checks]}
  rows.append(dict(qa_id=c['qa_id'],arm=arm,prompt_sha256=sha(c['prompt']+(SUFFIX if arm=='assessment' else '')),schema_sha256=sha(json.dumps(schema,sort_keys=True)),response=response,verdicts=validate(response,checks,arm),status='ok',calls=0,tokens=0))
# Provider timeout and malformed response remain explicit failures.
rows[0].pop('response');rows[0].pop('verdicts');rows[0].update(status='error',error_type='TimeoutError')
rows[1].pop('verdicts');rows[1].update(response={},status='error',error_type='ValueError')
with tempfile.TemporaryDirectory() as tmp:
 path=Path(tmp)/'mock.jsonl';path.with_suffix('.plan.json').write_text(json.dumps(plan))
 def save(rs):path.write_text(''.join(json.dumps(r)+'\n' for r in rs))
 save(rows);result=summarize(path);assert len(result['errors'])==2 and result['complete_pairs']==len(plan['cases'])
 rejected=[]
 for kind in ['prompt','verdict','order','incomplete']:
  bad=copy.deepcopy(rows)
  if kind=='prompt':bad[2]['prompt_sha256']='bad'
  if kind=='verdict':bad[2]['verdicts'][0]['verdict']='personal_supported'
  if kind=='order':bad[2],bad[3]=bad[3],bad[2]
  if kind=='incomplete':bad.pop()
  save(bad)
  try:summarize(path)
  except ValueError:rejected.append(kind)
  else:raise AssertionError(kind)
print(json.dumps(dict(status='passed',complete_pairs=len(plan['cases']),preserved_errors=2,rejected=rejected)))
