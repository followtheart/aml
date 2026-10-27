"""Reproduce full reader archives; isolate order of source cards in citation recovery."""
import argparse,asyncio,copy,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac,budget,config,eval_scoring,llm
from probe_question_source_filter import load_cohort,sha
from replay_query_context import recorded_response


def reverse_cards(prompt):
    prefix,raw=prompt.split('\nOriginal source cards:\n',1)
    cards=json.loads(raw)
    if not isinstance(cards,list) or len({c['id'] for c in cards})!=len(cards):raise ValueError('Source cards are not unique')
    return prefix+'\nOriginal source cards:\n'+json.dumps(list(reversed(cards)),ensure_ascii=False)

async def capture(cohort):
    captured=[]
    for r,qa,packet,sources,constraints in cohort:
        pending=copy.deepcopy(r['answer_calls']);failures=[];calls=[]
        async def frozen(prompt,*a,**kw):
            if not pending:failures.append('extra');raise ValueError('No live calls permitted')
            old=pending.pop(0)
            if old['stage']!=kw['stage'] or old['prompt_sha256']!=sha(prompt):failures.append(kw['stage']);raise ValueError('Archived prompt changed')
            if kw['stage']=='eval.choice_citation_recovery':
                reverse_cards(prompt)
                calls.append(dict(qa_id=r['qa_id'],prompt=prompt,schema=kw['schema'],system=kw.get('system'),max_tokens=kw.get('max_tokens'),response=old.get('response')))
            budget.current.get().before_call();return recorded_response(old)
        with patch.multiple(config,**r['versions']['settings']),patch.object(llm,'complete_json',frozen),budget.scope(seconds=300,calls=30,tokens=160000):
            prediction,score,_=await eval_scoring.evaluate(qa,copy.deepcopy(packet))
        if pending or failures or (prediction,score)!=(r['prediction'],r['score']):raise ValueError('Full frozen outcome mismatch '+r['qa_id'])
        if len(calls)>1:raise ValueError('Multiple recovery calls need explicit handling')
        captured.extend(calls)
    return captured

async def main(args):
    if config.FAKE:raise ValueError('Real provider required')
    cohort=load_cohort(args.run,args.prepared);cases=await capture(cohort)
    plan=dict(run=str(args.run),prepared=str(args.prepared),prepared_sha256=hashlib.sha256(args.prepared.read_bytes()).hexdigest(),all_questions=[r['qa_id'] for r,*_ in cohort],cases=cases,arms=['original','reversed'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),caveat='All archived reader outcomes exactly replayed before provider calls. Every available citation-recovery batch included, regardless of score. Source order only changes; original schema/system and all text preserved. Proposal diagnostic, not final-answer accuracy.')
    with args.output.open('x') as out:
        with args.output.with_suffix('.plan.json').open('x') as f:f.write(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
        for i,c in enumerate(cases):
            targets=json.loads(c['prompt'].split('\nClaims:\n')[1].split('\nOriginal source cards:\n')[0]);expected=[x['claim_id'] for x in targets]
            for arm in (plan['arms'] if i%2==0 else list(reversed(plan['arms']))):
                prompt=c['prompt'] if arm=='original' else reverse_cards(c['prompt']);row=dict(qa_id=c['qa_id'],arm=arm,prompt_sha256=sha(prompt))
                try:
                    with budget.scope(seconds=100,calls=1,tokens=64000) as usage:
                        response=await llm.complete_json(prompt,schema=c['schema'],system=c['system'],stage='diagnostic.recovery_order.'+arm,attempts=1,timeout=90,max_tokens=c['max_tokens'])
                    row.update(response=response,calls=usage.calls,tokens=usage.tokens)
                    normalized=ac.normalize_citation_recovery_matches(response,expected)
                    row.update(status='ok',proposals=[x.model_dump() for x in normalized])
                except Exception as exc:row.update(status='error',error_type=type(exc).__name__,error=str(exc)[:500])
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();print(json.dumps({k:v for k,v in row.items() if k not in ['response','proposals']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','prepared','output']:p.add_argument('--'+n,type=Path,required=True)
    asyncio.run(main(p.parse_args()))
