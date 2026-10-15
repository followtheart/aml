"""Audit complete structured action diagnostic prompts, schemas and mappings offline."""
import argparse,asyncio,hashlib,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac
from probe_constraint_action_evidence import capture
from probe_question_source_filter import load_cohort,sha
from constraint_action_evidence import Response,pairs,transform,normalize
async def audit(path):
 plan=json.loads(path.with_suffix('.plan.json').read_text());rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
 assert plan['prepared_sha256']==hashlib.sha256(Path(plan['prepared']).read_bytes()).hexdigest()
 assert plan['rule_sha256']==sha(Path(__file__).with_name('constraint_action_evidence.py').read_text())
 cohort=load_cohort(Path(plan['run']),Path(plan['prepared']));assert plan['all_questions']==[r['qa_id'] for r,*_ in cohort]
 cases=await capture(cohort);cases=[c for c in cases if any(re.search(r'\bI\s+(?:volunteer|am volunteering)\b',p['constraint_quote'],re.I) for p in pairs(c['prompt']))];assert cases==plan['cases']
 expected=[(c,rep+1,arm) for c in cases for rep in range(2) for arm in (['control','candidate'] if rep==0 else ['candidate','control'])];assert len(rows)==len(expected)
 differences=[];controls={};errors=[]
 for row,(case,rep,arm) in zip(rows,expected):
  assert (row['qa_id'],row['repetition'],row['arm'])==(case['qa_id'],rep,arm)
  prompt=case['prompt'] if arm=='control' else transform(case['prompt']);schema=case['schema'] if arm=='control' else Response.model_json_schema()
  assert row['prompt_sha256']==sha(prompt) and row['schema_sha256']==sha(json.dumps(schema,sort_keys=True))
  if row['status']!='ok':errors.append(dict(qa_id=case['qa_id'],arm=arm,repetition=rep,error=row.get('error_type')));continue
  normalized=row['response'] if arm=='control' else normalize(row['response'],pairs(case['prompt']))
  assert normalized['decisions']==row['decisions'];parsed=ac.ConstraintDecisions.model_validate(normalized)
  expected_ids={p['pair_id'] for p in pairs(case['prompt'])};assert len(parsed.decisions)==len(expected_ids) and {d.pair_id for d in parsed.decisions}==expected_ids
  controls[(case['qa_id'],rep,arm)]={d.pair_id:d.violates for d in parsed.decisions}
 for c in cases:
  for rep in [1,2]:
   before=controls.get((c['qa_id'],rep,'control'));after=controls.get((c['qa_id'],rep,'candidate'))
   if before is not None and after is not None:
    differences.extend(dict(qa_id=c['qa_id'],repetition=rep,pair_id=k,control=v,candidate=after[k]) for k,v in before.items() if after[k]!=v)
 return dict(status='passed',questions=len(cohort),cases=len(cases),calls=len(rows),errors=errors,differences=differences,caveat='Exact archived control and diagnostic trace audit; relation labels remain model judgments. No downstream accuracy claim.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=asyncio.run(audit(a.replay));a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps(r))
