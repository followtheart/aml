"""Full-cohort runtime constraint term flag toggle on frozen HTTPS packets."""
import argparse,asyncio,copy,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac,budget,config,eval_scoring,llm
from probe_question_source_filter import load_cohort,sha
from replay_query_context import recorded_response


async def main(a):
 rule_file=Path(__file__).resolve().parents[1]/'app/constraint_term_forms.py'
 cohort=load_cohort(a.run,a.prepared);provider=llm.complete_json;rows=[]
 async def run(data,arm):
  r,q,packet,s,c=data;pending=copy.deepcopy(r['answer_calls']);live=[];live_started=False;matched=[];mismatches=[]
  async def frozen(prompt,*pos,**kw):
   nonlocal live_started
   old=pending.pop(0) if pending else None
   if not live_started and old and old['stage']==kw['stage'] and old['prompt_sha256']==sha(prompt):
    budget.current.get().before_call();return recorded_response(old)
   if arm=='control' or (not live_started and kw['stage'] not in ['eval.choice_constraints','eval.choice_select','eval.choice_select_review']):
    mismatches.append(kw['stage']);raise RuntimeError('Unexpected pre-rule divergence')
   if not live_started:matched.append(dict(from_enabled=r['versions']['settings']['CHOICE_CONSTRAINT_TERM_FORMS'],to_enabled=not r['versions']['settings']['CHOICE_CONSTRAINT_TERM_FORMS'],first_changed_stage=kw['stage']))
   live_started=True;trace=dict(stage=kw['stage'],prompt=prompt,prompt_sha256=sha(prompt));live.append(trace)
   try:
    response=await provider(prompt,*pos,**kw);trace.update(status='ok',response=response);return response
   except Exception as exc:trace.update(status='error',error_type=type(exc).__name__);raise
  settings=dict(r['versions']['settings'])
  if arm=='candidate':settings['CHOICE_CONSTRAINT_TERM_FORMS']=not settings['CHOICE_CONSTRAINT_TERM_FORMS']
  with patch.multiple(config,**settings),patch.object(llm,'complete_json',frozen),budget.scope(seconds=300,calls=40,tokens=200000):
   pred,score,diag=await eval_scoring.evaluate(q,copy.deepcopy(packet))
  if mismatches:raise ValueError('Unexpected calls '+str(mismatches))
  if arm=='candidate' and not live_started and (pending or (pred,score)!=(r['prediction'],r['score'])):
   matched.append(dict(from_enabled=r['versions']['settings']['CHOICE_CONSTRAINT_TERM_FORMS'],to_enabled=settings['CHOICE_CONSTRAINT_TERM_FORMS'],deterministic_effect=True,skipped_archived_stages=[c['stage'] for c in pending]))
  if arm=='control' and (pending or live or (pred,score)!=(r['prediction'],r['score'])):raise ValueError('Baseline mismatch')
  return dict(qa_id=r['qa_id'],arm=arm,prediction=pred,score=score,matched=matched,live_calls=live,eligible=diag.get('answer_eligible_options'),entailment_rules=diag.get('answer_entailment_rules'),calls=diag.get('answer_calls'),diagnostics=diag)
 for data in cohort:rows.append(await run(data,'control'))
 for data in cohort:
  row=await run(data,'candidate');rows.append(row);print(json.dumps({k:row[k] for k in ['qa_id','arm','prediction','score','matched']}),flush=True)
 report=dict(run=str(a.run),prepared=str(a.prepared),prepared_sha256=hashlib.sha256(a.prepared.read_bytes()).hexdigest(),rule_sha256=hashlib.sha256(rule_file.read_bytes()).hexdigest(),rows=rows,scores={arm:sum(r['score'] for r in rows if r['arm']==arm) for arm in ['control','candidate']},live_calls=sum(len(r['live_calls']) for r in rows),caveat='Full archived cohort preflight, unchanged provider outputs reused only until exact-prompt divergence. Candidate flips the recorded runtime CHOICE_CONSTRAINT_TERM_FORMS flag on the same complete archived packets. All other recorded settings fixed; original control must match exactly, first divergence restricted to constraint/selection stages; downstream calls live. Control means original recorded setting, candidate means its opposite, including reverse on-to-off replay. Diagnostic fixed-response propagation, not new HTTPS accuracy; single-conversation narrow rule may overfit. Default unchanged.')
 with a.output.open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(scores=report['scores'],live_calls=report['live_calls'])))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['run','prepared','output']:p.add_argument('--'+n,type=Path,required=True)
 asyncio.run(main(p.parse_args()))
