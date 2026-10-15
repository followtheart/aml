"""Paired verifier diagnostic: labels versus brief source-grounded assessments."""
import argparse,asyncio,copy,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac,budget,config,eval_scoring,llm
from probe_question_source_filter import load_cohort,sha
from replay_query_context import recorded_response

SUFFIX='''\nFor every check, also return a short evidence_assessment (at most 180 characters): summarize what the supplied sources establish or which specific personal detail is missing. Include only source_ids from that check, or [] when none apply. Keep the existing evidence policy and distinguish topical interest from ownership, diagnosis, frequency and expertise. This brief assessment is an audit aid; never invent evidence. Return the verdict and these fields using the required schema.\n'''

def checks_in(prompt):return json.loads(prompt.split('<checks>\n',1)[1].split('\n</checks>',1)[0])
def assessment_schema(checks):
    return dict(type='object',additionalProperties=False,required=['checks'],properties=dict(checks=dict(type='array',minItems=len(checks),maxItems=len(checks),items=dict(type='object',additionalProperties=False,required=['claim_id','evidence_assessment','source_ids','verdict'],properties=dict(claim_id=dict(type='string',enum=[c['claim_id'] for c in checks]),evidence_assessment=dict(type='string',minLength=1,maxLength=180),source_ids=dict(type='array',maxItems=6,uniqueItems=True,items=dict(type='string')),verdict=dict(type='string',enum=['personal_supported','unsupported_personal','no_personal_premise']))))))

def validate(response,checks,arm):
    if not isinstance(response,dict) or set(response)!={'checks'} or not isinstance(response['checks'],list):raise ValueError('Invalid response')
    expected={c['claim_id']:c for c in checks};seen=set();normalized=[]
    for c in response['checks']:
        fields={'claim_id','verdict'}|({'evidence_assessment','source_ids'} if arm=='assessment' else set())
        if not isinstance(c,dict) or set(c)!=fields or c['claim_id'] not in expected or c['claim_id'] in seen:raise ValueError('Invalid/duplicate check')
        seen.add(c['claim_id'])
        if c['verdict'] not in ['personal_supported','unsupported_personal','no_personal_premise']:raise ValueError('Invalid verdict')
        if arm=='assessment':
            allowed={s['id'] for field in ['sources','context_neighbors'] for s in expected[c['claim_id']].get(field,[])};ids=c['source_ids']
            if not isinstance(c['evidence_assessment'],str) or not 1<=len(c['evidence_assessment'])<=180:raise ValueError('Invalid assessment length')
            if not isinstance(ids,list) or len(ids)>6 or any(not isinstance(x,str) or x not in allowed for x in ids) or len(set(ids))!=len(ids):raise ValueError('Unknown/duplicate source')
        normalized.append(dict(claim_id=c['claim_id'],verdict=c['verdict']))
    if seen!=set(expected):raise ValueError('Missing checks')
    return sorted(normalized,key=lambda c:c['claim_id'])

async def capture(cohort):
    cases=[];all_ids=[]
    for r,q,p,s,c in cohort:
        pending=copy.deepcopy(r['answer_calls']);errors=[];calls=[]
        async def frozen(prompt,*a,**kw):
            if not pending:errors.append('extra');raise ValueError('Extra call')
            old=pending.pop(0)
            if old['stage']!=kw['stage'] or old['prompt_sha256']!=sha(prompt):errors.append('mismatch');raise ValueError('Prompt mismatch')
            if kw['stage']=='eval.choice_entailment':
                checks_in(prompt);calls.append(dict(qa_id=r['qa_id'],prompt=prompt,schema=kw['schema'],system=kw.get('system'),max_tokens=kw.get('max_tokens'),archived_response=old.get('response')))
            budget.current.get().before_call();return recorded_response(old)
        with patch.multiple(config,**r['versions']['settings']),patch.object(llm,'complete_json',frozen),budget.scope(seconds=300,calls=40,tokens=200000):
            pred,score,_=await eval_scoring.evaluate(q,copy.deepcopy(p))
        if pending or errors or (pred,score)!=(r['prediction'],r['score']):raise ValueError('Frozen replay mismatch '+r['qa_id'])
        if len(calls)>1:raise ValueError('Multiple initial batches')
        cases.extend(calls);all_ids.append(r['qa_id'])
    return all_ids,cases

async def main(args):
    if config.FAKE:raise ValueError('Real provider required')
    all_ids,cases=await capture(load_cohort(args.run,args.prepared))
    plan=dict(run=str(args.run),prepared=str(args.prepared),prepared_sha256=hashlib.sha256(args.prepared.read_bytes()).hexdigest(),all_questions=all_ids,cases=cases,arms=['labels','assessment'],suffix=SUFFIX,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),caveat='Every initial verifier batch from complete cohort, regardless of gold score; all archived outcomes exactly replayed before live calls. Same source objects/system/instructions and3072-output cap; treatment adds short assessment/source IDs. Format errors retained, no fallback/retry. Diagnostic verdicts, not final reader or HTTPS accuracy. Source ID validity does not prove assessment semantics.')
    with args.output.open('x') as out:
        with args.output.with_suffix('.plan.json').open('x') as f:f.write(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
        for i,case in enumerate(cases):
            checks=checks_in(case['prompt'])
            for arm in (plan['arms'] if i%2==0 else list(reversed(plan['arms']))):
                prompt=case['prompt']+(SUFFIX if arm=='assessment' else '');schema=assessment_schema(checks) if arm=='assessment' else case['schema'];row=dict(qa_id=case['qa_id'],arm=arm,prompt_sha256=sha(prompt),schema_sha256=sha(json.dumps(schema,sort_keys=True)))
                try:
                    with budget.scope(seconds=100,calls=1,tokens=100000) as usage:
                        try:response=await llm.complete_json(prompt,schema=schema,system=case['system'],max_tokens=case['max_tokens'],attempts=1,timeout=90,stage='diagnostic.verifier_assessment.'+arm)
                        finally:row.update(calls=usage.calls,tokens=usage.tokens)
                    row['response']=response;row['verdicts']=validate(response,checks,arm);row['status']='ok'
                except Exception as exc:row.update(status='error',error_type=type(exc).__name__,error=str(exc)[:500])
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();print(json.dumps({k:row.get(k) for k in ['qa_id','arm','status','error_type']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','prepared','output']:p.add_argument('--'+n,type=Path,required=True)
    asyncio.run(main(p.parse_args()))
