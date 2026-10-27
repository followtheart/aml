import argparse,asyncio,copy,hashlib,json,re,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac,budget,config,eval_scoring,llm
from probe_question_source_filter import load_cohort,sha
from replay_query_context import recorded_response
from constraint_action_evidence import Response,pairs,transform,normalize
async def capture(cohort):
    captured=[]
    for r,qa,packet,sources,constraints in cohort:
        pending=copy.deepcopy(r['answer_calls']);failures=[];calls=[]
        async def frozen(prompt,*a,**kw):
            if not pending:failures.append('extra');raise ValueError('No live calls permitted')
            old=pending.pop(0)
            if old['stage']!=kw['stage'] or old['prompt_sha256']!=sha(prompt):failures.append(kw['stage']);raise ValueError('Archived prompt changed')
            if kw['stage']=='eval.choice_constraints':
                calls.append(dict(qa_id=r['qa_id'],prompt=prompt,schema=kw['schema'],system=kw.get('system'),max_tokens=kw.get('max_tokens'),response=old.get('response')))
            budget.current.get().before_call();return recorded_response(old)
        with patch.multiple(config,**r['versions']['settings']),patch.object(llm,'complete_json',frozen),budget.scope(seconds=300,calls=30,tokens=160000):
            prediction,score,_=await eval_scoring.evaluate(qa,copy.deepcopy(packet))
        if pending or failures or (prediction,score)!=(r['prediction'],r['score']):raise ValueError('Full frozen outcome mismatch '+r['qa_id'])
        if len(calls)>1:raise ValueError('Multiple constraint calls need explicit handling')
        captured.extend(calls)
    return captured

async def main(a):
 cohort=load_cohort(a.run,a.prepared);allcases=await capture(cohort)
 cases=[c for c in allcases if any(re.search(r'\bI\s+(?:volunteer|am volunteering)\b',p['constraint_quote'],re.I) for p in pairs(c['prompt']))]
 plan=dict(run=str(a.run),prepared=str(a.prepared),prepared_sha256=hashlib.sha256(a.prepared.read_bytes()).hexdigest(),all_questions=[r['qa_id'] for r,*_ in cohort],cases=cases,repetitions=2,selection='All existing constraint batches mentioning explicit first-person volunteering; regardless of score; full reader preflight before any live call.',rule_sha256=sha(Path(__file__).with_name('constraint_action_evidence.py').read_text()))
 a.output.with_suffix('.plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
 with a.output.open('x') as f:
  for c in cases:
   for rep in range(2):
    for arm in (['control','candidate'] if rep==0 else ['candidate','control']):
     prompt=c['prompt'] if arm=='control' else transform(c['prompt']);schema=c['schema'] if arm=='control' else Response.model_json_schema();row=dict(qa_id=c['qa_id'],arm=arm,repetition=rep+1,prompt_sha256=sha(prompt),schema_sha256=sha(json.dumps(schema,sort_keys=True)))
     try:
      with budget.scope(seconds=110,calls=1,tokens=64000):response=await llm.complete_json(prompt,schema=schema,system=c['system'],stage='diagnostic.constraint_action.'+arm,attempts=1,timeout=90,max_tokens=c['max_tokens'])
      normalized=response if arm=='control' else normalize(response,pairs(c['prompt']))
      parsed=ac.ConstraintDecisions.model_validate(normalized);expected={p['pair_id'] for p in pairs(c['prompt'])};assert len(parsed.decisions)==len(expected) and {d.pair_id for d in parsed.decisions}==expected
      row.update(status='ok',response=response,decisions=normalized['decisions'])
     except Exception as exc:row.update(status='error',error_type=type(exc).__name__)
     f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();print(json.dumps({k:v for k,v in row.items() if k not in ['response','decisions']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['run','prepared','output']:p.add_argument('--'+n,type=Path,required=True)
 asyncio.run(main(p.parse_args()))
