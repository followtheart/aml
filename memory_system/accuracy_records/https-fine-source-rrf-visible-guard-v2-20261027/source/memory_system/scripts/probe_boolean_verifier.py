"""Diagnostic: normalized three-label verifier versus equivalent Boolean outputs."""
import argparse,asyncio,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget,config,llm
from probe_question_source_filter import sha
from probe_verifier_assessment import checks_in,validate as validate_labels
from summarize_verifier_assessment import summarize as audit_source
from typed_entailment_prompt import normalize

SUFFIX='''\nUse the same evidence rules above, but encode each result with two Boolean fields instead of a verdict label.\nasserts_personal_fact: true exactly when this check asserts a pre-existing personal fact. Missing evidence does NOT make this false. Proposed new activities alone are not prior personal facts.\nall_personal_facts_supported: true exactly when all personal facts asserted by this check are supported by the supplied sources under the rules above. If no personal fact is asserted, this field must be true (there is no unsupported personal fact).\nThe equivalent encodings are: personal_supported = true/true; unsupported_personal = true/false; no_personal_premise = false/true. false/false is inconsistent and must not be returned.\nReturn claim_id, asserts_personal_fact and all_personal_facts_supported for every check. Do not return verdict or commentary.\n'''

def schema(checks):
 return dict(type='object',additionalProperties=False,required=['checks'],properties=dict(checks=dict(type='array',minItems=len(checks),maxItems=len(checks),items=dict(type='object',additionalProperties=False,required=['claim_id','asserts_personal_fact','all_personal_facts_supported'],properties=dict(claim_id=dict(type='string',enum=[c['claim_id'] for c in checks]),asserts_personal_fact=dict(type='boolean'),all_personal_facts_supported=dict(type='boolean'))))))

def validate(response,checks,arm):
 if arm=='labels':return validate_labels(response,checks,'labels')
 if not isinstance(response,dict) or set(response)!={'checks'} or not isinstance(response['checks'],list):raise ValueError('Invalid root')
 mapped=[]
 for c in response['checks']:
  if not isinstance(c,dict) or set(c)!={'claim_id','asserts_personal_fact','all_personal_facts_supported'}:raise ValueError('Invalid fields')
  a,b=c['asserts_personal_fact'],c['all_personal_facts_supported']
  if type(a) is not bool or type(b) is not bool or (not a and not b):raise ValueError('Invalid/inconsistent Boolean pair')
  mapped.append(dict(claim_id=c['claim_id'],verdict='no_personal_premise' if not a else 'personal_supported' if b else 'unsupported_personal'))
 return validate_labels(dict(checks=mapped),checks,'labels')

async def main(args):
 if config.FAKE:raise ValueError('Real provider required')
 audit=audit_source(args.source)
 if audit['partial']:raise ValueError('Source run incomplete')
 sourceplan=json.loads(args.source.with_suffix('.plan.json').read_text());cases=sourceplan['cases']
 # Normalize the legacy Boolean/typed contradiction identically in both arms.
 for c in cases:c['normalized_prompt']=normalize(c['prompt'])
 plan=dict(source=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),source_plan_sha256=hashlib.sha256(args.source.with_suffix('.plan.json').read_bytes()).hexdigest(),cases=cases,arms=['labels','booleans'],suffix=SUFFIX,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),caveat='Complete28 verifier batches from previously exact-frozen-replayed cohort25. Both prompts first receive identical previously tested contract normalization. Same check objects, evidence, system and3072output cap. Only output encoding/instructions differ; Boolean pairs map to original labels. One call per arm, alternating order, failures retained. Not final-answer accuracy.')
 with args.output.open('x') as out:
  args.output.with_suffix('.plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
  for i,c in enumerate(cases):
   checks=checks_in(c['prompt'])
   for arm in (plan['arms'] if i%2==0 else list(reversed(plan['arms']))):
    prompt=c['normalized_prompt']+(SUFFIX if arm=='booleans' else '');contract=schema(checks) if arm=='booleans' else c['schema'];row=dict(qa_id=c['qa_id'],arm=arm,prompt_sha256=sha(prompt),schema_sha256=sha(json.dumps(contract,sort_keys=True)))
    try:
     with budget.scope(seconds=100,calls=1,tokens=100000) as usage:
      try:response=await llm.complete_json(prompt,schema=contract,system=c['system'],max_tokens=c['max_tokens'],attempts=1,timeout=90,stage='diagnostic.boolean_verifier.'+arm)
      finally:row.update(calls=usage.calls,tokens=usage.tokens)
     row['response']=response;row['verdicts']=validate(response,checks,arm);row['status']='ok'
    except Exception as exc:row.update(status='error',error_type=type(exc).__name__,error=str(exc)[:500])
    out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();print(json.dumps({k:row.get(k) for k in ['qa_id','arm','status','error_type']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);asyncio.run(main(p.parse_args()))
