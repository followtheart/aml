"""Offline exact provider-prompt replay for the repeated-interest candidate."""
import argparse,asyncio,contextlib,copy,hashlib,io,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import llm
from probe_question_source_filter import sha,load_cohort
from replay_query_context import recorded_response
from replay_generic_observation_scope import main as replay

async def audit(path):
 r=json.loads(path.read_text());prepared=Path(r['prepared']);rule=Path(__file__).with_name('generic_observation_scope.py')
 if hashlib.sha256(prepared.read_bytes()).hexdigest()!=r['prepared_sha256'] or hashlib.sha256(rule.read_bytes()).hexdigest()!=r['rule_sha256']:raise ValueError('Inputs changed')
 expected=[(d[0]['qa_id'],arm) for arm in ['control','candidate'] for d in load_cohort(Path(r['run']),prepared)]
 if [(x['qa_id'],x['arm']) for x in r['rows']]!=expected:raise ValueError('Population/order changed')
 pending=[copy.deepcopy(c) for x in r['rows'] for c in x['live_calls']];errors=[];calls=len(pending)
 async def frozen(prompt,*pos,**kw):
  if not pending:errors.append('extra');raise ValueError('Unexpected provider call')
  c=pending.pop(0)
  if c['stage']!=kw['stage'] or c['prompt_sha256']!=sha(prompt):errors.append('changed');raise ValueError('Prompt changed')
  return recorded_response(c)
 with tempfile.TemporaryDirectory() as tmp:
  out=Path(tmp)/'audit.json'
  with patch.object(llm,'complete_json',frozen),contextlib.redirect_stdout(io.StringIO()):await replay(SimpleNamespace(run=Path(r['run']),prepared=prepared,output=out,rule_version=r.get('rule_version','v1')))
  again=json.loads(out.read_text())
 if pending or errors:raise ValueError('Provider mismatch')
 fields=['qa_id','arm','prediction','score','matched','eligible','entailment_rules']
 if [{k:x.get(k) for k in fields} for x in r['rows']]!=[{k:x.get(k) for k in fields} for x in again['rows']]:raise ValueError('Outcome changed')
 if r['scores']!=again['scores'] or r['live_calls']!=calls:raise ValueError('Totals changed')
 controls={x['qa_id']:x for x in r['rows'] if x['arm']=='control'};changes=[dict(qa_id=x['qa_id'],control=controls[x['qa_id']]['prediction'],candidate=x['prediction'],score_before=controls[x['qa_id']]['score'],score_after=x['score'],matched=x['matched']) for x in r['rows'] if x['arm']=='candidate' and x['prediction']!=controls[x['qa_id']]['prediction']]
 return dict(status='passed',pairs=len(controls),scores=r['scores'],live_calls=calls,changes=changes,provider_errors=[c.get('error_type') for x in r['rows'] for c in x['live_calls'] if c['status']=='error'],caveat=r['caveat'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=asyncio.run(audit(a.replay));a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='caveat'}))
