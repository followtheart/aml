"""Paired semantic boundary diagnostic; no answer accuracy claim."""
import asyncio,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac,budget,llm,prompts
from volunteer_constraint_scope import augment
CASES=[('Visit the community garden to watch birds.',False),('Rent a corner of the community garden for a photography workshop.',False),('Help out weeding the beds at the community garden.',True),('Resume volunteering every Saturday at the community garden.',True),('Avoid taking on unpaid work at the community garden.',False),('Offer your time to help maintain the community garden.',True)]
async def main():
 pairs=[dict(pair_id=f'case{i}',constraint_quote='Please forget that I volunteer at local community gardens.',option_quote=option) for i,(option,expected) in enumerate(CASES)]
 original=prompts.render('13_choice_constraints.txt',question='Suggest an outdoor weekend activity.',pairs=json.dumps(pairs,ensure_ascii=False))
 candidate,matched=augment(original);assert len(matched)==len(pairs)
 output=Path('runs/https-cycle-20261011-volunteer-scope-boundaries.jsonl');output.with_suffix('.plan.json').write_text(json.dumps(dict(cases=[dict(**p,expected=CASES[i][1]) for i,p in enumerate(pairs)],prompts=dict(control=original,candidate=candidate),repetitions=2),ensure_ascii=False,indent=2)+'\n')
 with output.open('x') as f:
  for rep in range(2):
   for arm in (['control','candidate'] if rep==0 else ['candidate','control']):
    prompt=original if arm=='control' else candidate;row=dict(arm=arm,repetition=rep+1,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    try:
     with budget.scope(seconds=110,calls=1,tokens=32000):response=await llm.complete_json(prompt,schema=ac.ConstraintDecisions.model_json_schema(),stage='diagnostic.volunteer_scope.'+arm,attempts=1,timeout=90)
     parsed=ac.ConstraintDecisions.model_validate(response);got={d.pair_id:d.violates for d in parsed.decisions};assert len(got)==len(parsed.decisions)==len(pairs) and set(got)=={p['pair_id'] for p in pairs}
     row.update(status='ok',response=response,errors=[p['pair_id'] for i,p in enumerate(pairs) if got[p['pair_id']]!=CASES[i][1]])
    except Exception as exc:row.update(status='error',error_type=type(exc).__name__)
    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();print(json.dumps(row),flush=True)
asyncio.run(main())
