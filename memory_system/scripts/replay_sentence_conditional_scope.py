"""Full-cohort fixed-response probe of generic observation scope rule."""
import argparse,asyncio,copy,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac,budget,config,eval_scoring,llm
from probe_question_source_filter import load_cohort,sha
from replay_query_context import recorded_response
from sentence_conditional_scope import reason

async def main(a):
 rule_file=Path(__file__).with_name('sentence_conditional_scope.py')
 cohort=load_cohort(a.run,a.prepared);provider=llm.complete_json;original_rule=ac._nonasserted_scope_reason;rows=[]
 async def run(data,arm):
  r,q,packet,s,c=data;pending=copy.deepcopy(r['answer_calls']);live=[];live_started=False;matched=[];mismatches=[]
  def rule(claim,option):
   prior=original_rule(claim,option)
   if prior:return prior
   added=reason(claim,option,original_rule) if arm=='candidate' else None
   if added:matched.append(dict(claim=copy.deepcopy(claim),option=option))
   return added
  async def frozen(prompt,*pos,**kw):
   nonlocal live_started
   old=pending.pop(0) if pending else None
   if not live_started and old and old['stage']==kw['stage'] and old['prompt_sha256']==sha(prompt):
    budget.current.get().before_call();return recorded_response(old)
   if arm=='control' or not matched:
    mismatches.append(kw['stage']);raise RuntimeError('Unexpected pre-rule divergence')
   live_started=True;trace=dict(stage=kw['stage'],prompt=prompt,prompt_sha256=sha(prompt));live.append(trace)
   try:
    response=await provider(prompt,*pos,**kw);trace.update(status='ok',response=response);return response
   except Exception as exc:trace.update(status='error',error_type=type(exc).__name__);raise
  with patch.multiple(config,**r['versions']['settings']),patch.object(ac,'_nonasserted_scope_reason',rule),patch.object(llm,'complete_json',frozen),budget.scope(seconds=300,calls=40,tokens=200000):
   pred,score,diag=await eval_scoring.evaluate(q,copy.deepcopy(packet))
  if mismatches:raise ValueError('Unexpected calls '+str(mismatches))
  if arm=='control' and (pending or live or (pred,score)!=(r['prediction'],r['score'])):raise ValueError('Baseline mismatch')
  return dict(qa_id=r['qa_id'],arm=arm,prediction=pred,score=score,matched=matched,live_calls=live,eligible=diag.get('answer_eligible_options'),entailment_rules=diag.get('answer_entailment_rules'),calls=diag.get('answer_calls'))
 for data in cohort:rows.append(await run(data,'control'))
 for data in cohort:
  row=await run(data,'candidate');rows.append(row);print(json.dumps({k:row[k] for k in ['qa_id','arm','prediction','score','matched']}),flush=True)
 report=dict(run=str(a.run),prepared=str(a.prepared),prepared_sha256=hashlib.sha256(a.prepared.read_bytes()).hexdigest(),rule_sha256=hashlib.sha256(rule_file.read_bytes()).hexdigest(),rows=rows,scores={arm:sum(r['score'] for r in rows if r['arm']==arm) for arm in ['control','candidate']},live_calls=sum(len(r['live_calls']) for r in rows),caveat='Full archived cohort preflight, unchanged provider outputs reused only until exact-prompt divergence. Candidate extends advice-scope classification; independent whole-option verification remains active; after divergence downstream calls live. Diagnostic fixed-response propagation, not new HTTPS accuracy; single-conversation narrow rule may overfit. Default unchanged.')
 with a.output.open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(scores=report['scores'],live_calls=report['live_calls'])))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['run','prepared','output']:p.add_argument('--'+n,type=Path,required=True)
 asyncio.run(main(p.parse_args()))
