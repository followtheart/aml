"""Offline propagation of audited verifier labels; stop at changed downstream prompts."""
import argparse,asyncio,copy,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget,config,eval_scoring,llm
from probe_question_source_filter import load_cohort,sha
from replay_query_context import recorded_response
from summarize_boolean_verifier import summarize
class Diverged(BaseException):pass

async def main(a):
 audit=summarize(a.probe)
 if audit['partial'] or audit['errors']:raise ValueError('Complete valid verifier pairs required')
 probeplan=json.loads(a.probe.with_suffix('.plan.json').read_text());source=Path(probeplan['source']);origin=json.loads(source.with_suffix('.plan.json').read_text());cohort=load_cohort(Path(origin['run']),Path(origin['prepared']));responses={(r['qa_id'],r['arm']):r for r in map(json.loads,a.probe.read_text().splitlines())};rows=[]
 async def run(data,arm):
  r,q,packet,s,c=data;pending=copy.deepcopy(r['answer_calls']);injected=False;divergence=None
  async def frozen(prompt,*pos,**kw):
   nonlocal injected,divergence
   old=pending.pop(0) if pending else None
   if old is None or old['stage']!=kw['stage'] or old['prompt_sha256']!=sha(prompt):
    divergence=dict(stage=kw['stage'],prompt=prompt,prompt_sha256=sha(prompt),archived_stage=old['stage'] if old else None,archived_prompt_sha256=old['prompt_sha256'] if old else None)
    raise Diverged()
   budget.current.get().before_call()
   if arm!='archived' and kw['stage']=='eval.choice_entailment':
    if injected:raise ValueError('Multiple intervention batches')
    injected=True;return dict(checks=copy.deepcopy(responses[r['qa_id'],arm]['verdicts']))
   return recorded_response(old)
  with patch.multiple(config,**r['versions']['settings']),patch.object(llm,'complete_json',frozen),budget.scope(seconds=300,calls=40,tokens=200000):
   try:pred,score,diag=await eval_scoring.evaluate(q,copy.deepcopy(packet))
   except Diverged:
    if arm=='archived' or not injected:raise ValueError('Unexpected pre-intervention divergence')
    return dict(qa_id=r['qa_id'],arm=arm,status='requires_new_downstream_call',divergence=divergence)
  if arm=='archived' and (pending or (pred,score)!=(r['prediction'],r['score'])):raise ValueError('Baseline mismatch')
  if arm!='archived' and not injected:raise ValueError('No injection')
  return dict(qa_id=r['qa_id'],arm=arm,status='resolved_with_archived_downstream',prediction=pred,score=score,unused_archived_calls=len(pending),eligible=diag.get('answer_eligible_options'),blocked=diag.get('answer_blocked_options'))
 for data in cohort:await run(data,'archived')
 for data in cohort:
  for arm in ['labels','booleans']:rows.append(await run(data,arm))
 report=dict(probe=str(a.probe),probe_sha256=hashlib.sha256(a.probe.read_bytes()).hexdigest(),complete_frozen_preflight=len(cohort),rows=rows,divergent={arm:[r['qa_id'] for r in rows if r['arm']==arm and r['status']=='requires_new_downstream_call'] for arm in ['labels','booleans']},caveat='No provider calls. Inject audited initial-verifier verdicts, reuse a downstream response only when both stage and full prompt SHA exactly match archive. Stop at first changed prompt; its final score is unknown. Resolved scores are conditional counterfactuals, not new HTTPS accuracy.')
 with a.output.open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k in ['complete_frozen_preflight','divergent']}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);asyncio.run(main(p.parse_args()))
