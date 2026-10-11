"""Re-execute every downstream trajectory from recorded provider responses only."""
import argparse,asyncio,contextlib,copy,hashlib,io,json,tempfile,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import llm
from probe_question_source_filter import sha
from replay_query_context import recorded_response
from replay_boolean_verifier_downstream import main as replay

async def audit(path):
 original=json.loads(path.read_text());probe=Path(original['probe'])
 if hashlib.sha256(probe.read_bytes()).hexdigest()!=original['probe_sha256']:raise ValueError('Probe changed')
 pending=[copy.deepcopy(c) for r in original['rows'] for c in r.get('live_calls',[])];count=len(pending);errors=[]
 async def frozen(prompt,*pos,**kw):
  if not pending:errors.append('extra');raise ValueError('No extra provider calls permitted')
  c=pending.pop(0)
  if c['stage']!=kw['stage'] or c['prompt_sha256']!=sha(prompt):errors.append('prompt');raise ValueError('Downstream prompt changed')
  return recorded_response(c)
 with tempfile.TemporaryDirectory() as tmp:
  out=Path(tmp)/'replayed.json'
  with patch.object(llm,'complete_json',frozen),contextlib.redirect_stdout(io.StringIO()):await replay(SimpleNamespace(probe=probe,output=out))
  reproduced=json.loads(out.read_text())
 if pending or errors:raise ValueError('Provider trajectory mismatch')
 fields=['qa_id','arm','status','prediction','score','eligible','blocked','unused_archived_calls']
 if [{k:r.get(k) for k in fields} for r in original['rows']]!=[{k:r.get(k) for k in fields} for r in reproduced['rows']]:raise ValueError('Outcomes changed')
 if len(original['rows'])!=56:raise ValueError('Incomplete cohort')
 by={};
 for r in original['rows']:
  if r['arm'] in by.setdefault(r['qa_id'],{}):raise ValueError('Duplicate arm')
  by[r['qa_id']][r['arm']]=r
 if len(by)!=28 or any(set(x)!= {'labels','booleans'} for x in by.values()):raise ValueError('Incomplete pairs')
 return dict(status='passed',pairs=28,recorded_live_calls=count,provider_errors=[dict(qa_id=r['qa_id'],arm=r['arm'],error_type=c.get('error_type')) for r in original['rows'] for c in r.get('live_calls',[]) if c['status']=='error'],scores={a:sum(x[a]['score'] for x in by.values()) for a in ['labels','booleans']},wins=[q for q,x in by.items() if x['booleans']['score']>x['labels']['score']],losses=[q for q,x in by.items() if x['booleans']['score']<x['labels']['score']],prediction_changes=[q for q,x in by.items() if x['booleans']['prediction']!=x['labels']['prediction']],artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),caveat=original['caveat'])
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=asyncio.run(audit(a.replay));a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='caveat'}))
