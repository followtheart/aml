"""Full-cohort fixed-prefix reader pairs changing only verifier boolean wording."""
import argparse,asyncio,copy,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget,config,eval_scoring,llm,run_metadata
from probe_question_source_filter import load_cohort,sha
from replay_query_context import recorded_response
from typed_entailment_prompt import normalize,REPLACEMENTS

async def main(args):
    if config.FAKE:raise ValueError('Real provider required')
    cohort=load_cohort(args.run,args.prepared);original=llm.complete_json;captured={}
    async def run(data,arm,frozen=False):
        r,qa,packet,sources,constraints=data;calls=r['answer_calls']
        boundary=next(i for i,c in enumerate(calls) if c['stage']=='eval.choice_entailment')
        pending=copy.deepcopy(calls if frozen else calls[:boundary]);mismatches=[];live=[];entail=[]
        async def provider(prompt,*pos,**kw):
            if pending:
                old=pending.pop(0)
                if old['stage']!=kw['stage'] or old['prompt_sha256']!=sha(prompt):
                    mismatches.append(kw['stage']);raise ValueError('Frozen prompt mismatch')
                if kw['stage']=='eval.choice_entailment':
                    normalized=normalize(prompt)
                    entail.append(dict(prompt=prompt,normalized=normalized,schema=kw['schema']))
                budget.current.get().before_call()
                return recorded_response(old)
            if frozen:
                mismatches.append('unexpected_call');raise ValueError('No live calls in preflight')
            actual=normalize(prompt) if arm=='typed' and kw['stage']=='eval.choice_entailment' else prompt
            if kw['stage']=='eval.choice_entailment' and not any(c['stage']=='eval.choice_entailment' for c in live):
                if sha(prompt)!=captured[r['qa_id']]['prompt_sha256']:
                    mismatches.append('entailment_input');raise ValueError('Frozen verifier input changed')
            trace=dict(stage=kw['stage'],original_prompt_sha256=sha(prompt),prompt=actual,prompt_sha256=sha(actual));live.append(trace)
            if kw['stage']=='eval.choice_entailment':trace['original_prompt']=prompt
            try:
                response=await original(actual,*pos,**kw);trace.update(status='ok',response=response);return response
            except Exception as exc:
                trace.update(status='error',error_type=type(exc).__name__,error=str(exc)[:500]);raise
        with patch.multiple(config,**r['versions']['settings']),patch.object(llm,'complete_json',provider),budget.scope(seconds=300,calls=30,tokens=160000) as usage:
            prediction,score,diagnostics=await eval_scoring.evaluate(qa,copy.deepcopy(packet))
        if pending or mismatches:raise ValueError(dict(pending=len(pending),mismatches=mismatches))
        if frozen:
            if prediction!=r['prediction'] or score!=r['score'] or len(entail)!=1:raise ValueError('Full frozen outcome mismatch')
            e=entail[0];captured[r['qa_id']]=dict(prompt_sha256=sha(e['prompt']),typed_prompt_sha256=sha(e['normalized']),schema_sha256=sha(json.dumps(e['schema'],sort_keys=True)))
        return dict(qa_id=r['qa_id'],conversation_id=r['conversation_id'],arm=arm,packet_hash=r['packet_hash'],frozen_verified=True,fixed_prefix_calls=boundary,prediction=prediction,score=score,diagnostics=diagnostics,live_calls=live,calls=usage.calls,tokens=usage.tokens)
    for data in cohort:await run(data,'baseline',True)
    if args.inspect:
        with args.output.open('x') as out:out.write(json.dumps(dict(verified_questions=list(captured),captured=captured),indent=2)+'\n')
        print(json.dumps(dict(frozen_verified=len(captured))));return
    plan=dict(run=str(args.run),prepared=str(args.prepared),prepared_sha256=hashlib.sha256(args.prepared.read_bytes()).hexdigest(),questions=[r['qa_id'] for r,*_ in cohort],arms=['baseline','typed'],captured=captured,replacements=REPLACEMENTS,versions=run_metadata.versions(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),transform_sha256=hashlib.sha256(Path(__file__).with_name('typed_entailment_prompt.py').read_bytes()).hexdigest(),caveat='Full frozen reader outcome preflight before live calls. Only verifier wording changes; all check/source objects identical. Prior calls replayed; expanded downstream budget excludes historical time/token usage. Not HTTPS. Diagnostics store caller hashes; live_calls store actual provider prompts.')
    with args.output.open('x') as out:
        with args.output.with_suffix('.plan.json').open('x') as f:f.write(json.dumps(plan,indent=2)+'\n')
        for i,data in enumerate(cohort):
            for arm in (['baseline','typed'] if i%2==0 else ['typed','baseline']):
                row=await run(data,arm);out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();print(json.dumps({k:row[k] for k in ['qa_id','arm','prediction','score']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','prepared','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--inspect',action='store_true')
    asyncio.run(main(p.parse_args()))
