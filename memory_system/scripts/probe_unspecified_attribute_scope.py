"""Diagnostic for open preference criteria versus concrete personal presuppositions."""
import argparse, asyncio, hashlib, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac, budget, config, llm
from probe_verification_scope import scope_prefix
GUIDANCE = (' An unspecified preference, goal, need or value used as an open selection criterion '
    'does not assert which preference, goal, need or value the user actually has. '
    'Distinguish such open criteria from identifying a particular interest, existing item, '
    'diagnosis, habit, personal event or specific circumstance. A concrete personal presupposition '
    'still requires personal_fact=true even when embedded in suggested actions. '
    'Do not erase it just because it occurs after your, favorite, choose or reflect. '
    'If the complete claim contains both an open criterion and a concrete presupposition, return true.')

def cases():
    examples=[
      ('Choose books from genres you enjoy.','genres you enjoy',False),
      ('Choose books from the historical biographies you enjoy.','the historical biographies you enjoy',True),
      ('Pick a route that fits your goals.','fits your goals',False),
      ('Pick a route that fits your goal of completing a marathon.','your goal of completing a marathon',True),
      ('Adapt the plan to your needs.','your needs',False),
      ('Adapt the plan to your insulin needs.','your insulin needs',True),
      ('Choose supplies in colors you like.','colors you like',False),
      ('Use your favorite turquoise inks.','your favorite turquoise inks',True),
      ('Read something that resonates with you.','resonates with you',False),
      ('Read something that recalls your years living abroad.','your years living abroad',True),
      ('Choose an activity aligned with your values.','your values',False),
      ('Choose an activity aligned with your lifelong commitment to animal rescue.','your lifelong commitment to animal rescue',True),
      ('Set goals that inspire you.','goals that inspire you',False),
      ('Because you do not own a bicycle, choose another activity.','you do not own a bicycle',True),
      ('Use your old incision sites as a guide when checking discomfort.','your old incision sites',True),
      ('Use your handwritten task list to record ideas.','your handwritten task list',True),
      ('Pick a book that reflects your values and your experience after surgery.','reflects your values and your experience after surgery',True),
      ('If you have a collection, choose something that complements it.','you have a collection',False),
    ]
    return [dict(claim_id='c'+str(i),option=o,claim=c,expected=e) for i,(o,c,e) in enumerate(examples)]

async def main(args):
    if config.FAKE:raise ValueError('Real provider required')
    data=cases();base=scope_prefix();schema=ac.Scopes.model_json_schema()
    plan=dict(cases=data,guidance=GUIDANCE,baseline=base,repetitions=2,arms=['baseline','open_criterion'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),caveat='Synthetic grammatical controls; labels never sent to provider; not accuracy or production validation.')
    with args.output.open('x') as out:
      with args.output.with_suffix('.plan.json').open('x') as f:f.write(json.dumps(plan,indent=2)+'\n')
      for rep in range(2):
        ordered=data if rep==0 else list(reversed(data))
        for arm in (['baseline','open_criterion'] if rep==0 else ['open_criterion','baseline']):
          prompt=base+(GUIDANCE if arm=='open_criterion' else '')+'\n'+json.dumps([{k:c[k] for k in ['claim_id','option','claim']} for c in ordered])
          row=dict(repetition=rep+1,arm=arm,prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
          try:
            with budget.scope(seconds=100,calls=1,tokens=32000) as usage:
              response=await llm.complete_json(prompt,schema=schema,stage='diagnostic.unspecified_scope.'+arm,attempts=1,timeout=90,max_tokens=2048)
            row.update(response=response,calls=usage.calls,tokens=usage.tokens)
            parsed=ac.Scopes.model_validate(response);decisions={c.claim_id:c.personal_fact for c in parsed.claims}
            if len(parsed.claims)!=len(data) or set(decisions)!={c['claim_id'] for c in data}:raise ValueError('Incomplete/duplicate scope decisions')
            row.update(status='ok',false_personal=[c['claim_id'] for c in data if not c['expected'] and decisions[c['claim_id']]],missed_personal=[c['claim_id'] for c in data if c['expected'] and not decisions[c['claim_id']]])
          except Exception as exc:row.update(status='error',error_type=type(exc).__name__,error=str(exc)[:500])
          out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();print(json.dumps({k:v for k,v in row.items() if k not in ['prompt','response']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);asyncio.run(main(p.parse_args()))
