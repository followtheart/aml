"""Reproduce paired reader inputs, then compare batch versus option-group verdicts."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice, budget, config, eval_scoring, evidence_packet, llm
from replay_query_context import records, recorded_response


async def main(args):
    if config.FAKE:raise ValueError('Real verifier required')
    rows=records(args.replay)
    if len(rows)!=2 or {r['enabled'] for r in rows}!={False,True}:raise ValueError('Expected one complete pair')
    result=next(r for r in records(args.run/'completed/data/results/personamem-v2-32k.jsonl') if r['qa_id']==args.question)
    trace=next(r for r in records(args.run/'server-search-debug.jsonl') if r.get('search_id')==result['search_id'])
    qa=next(q for d in records(args.prepared) if d['conversation_id']==result['conversation_id'] for q in d['qa'] if q['id']==args.question)
    captured={}
    for row in rows:
        if row['packet_hash']!=evidence_packet.digest(trace['returned']):raise ValueError('Packet mismatch')
        pending=copy.deepcopy(row['diagnostics']['answer_calls']); failures=[]; calls=[]
        async def frozen(prompt,*positional,**kwargs):
            old=pending.pop(0) if pending else {}
            if old.get('stage')!=kwargs['stage'] or old.get('prompt_sha256')!=hashlib.sha256(prompt.encode()).hexdigest():
                failures.append(kwargs['stage']);raise ValueError('Reader prompt mismatch')
            if kwargs['stage']=='eval.choice_entailment':
                calls.append(dict(prompt=prompt,schema=kwargs['schema'],system=kwargs.get('system'),response=old.get('response')))
            budget.current.get().before_call()
            return recorded_response(old)
        with patch.object(llm,'complete_json',frozen),budget.scope(seconds=300,calls=30,tokens=160000):
            prediction,score,_=await eval_scoring.evaluate(qa,copy.deepcopy(trace['returned']))
        if pending or failures or prediction!=row['prediction'] or score!=row['score'] or len(calls)!=1:
            raise ValueError('Exact reader preflight failed')
        captured[str(row['enabled'])]=calls[0]
    def checks(c):return json.loads(c['prompt'].split('<checks>\n',1)[1].split('\n</checks>',1)[0])
    originals={k:{c['claim_id']:c for c in checks(v)} for k,v in captured.items()}
    common=set(originals['False']) & set(originals['True'])
    comparison={cid:originals['False'][cid]==originals['True'][cid] for cid in sorted(common)}
    capture=dict(qa_id=args.question,exact_reader_preflight=True,arms=captured,identical_checks=comparison)
    with args.capture.open('x') as out:json.dump(capture,out,ensure_ascii=False,indent=2)
    print(json.dumps(dict(qa_id=args.question,identical_checks=comparison)),flush=True)
    if args.inspect:return
    selected=captured['True']; all_checks=checks(selected)
    groups={letter:[c for c in all_checks if c['claim_id'].split(':',1)[0]==letter]
            for letter in dict.fromkeys(c['claim_id'].split(':',1)[0] for c in all_checks)}
    with args.output.open('x') as out:
        for rep in range(args.repeat):
            work=[('batch',all_checks),*groups.items()]
            if rep%2:work.reverse()
            for group,items in work:
                prompt=selected['prompt'].replace('<checks>\n'+json.dumps(all_checks,ensure_ascii=False)+'\n</checks>',
                    '<checks>\n'+json.dumps(items,ensure_ascii=False)+'\n</checks>',1)
                if checks(dict(prompt=prompt))!=items:raise ValueError('Check block replacement failed')
                schema=copy.deepcopy(selected['schema'])
                with budget.scope(seconds=70,calls=1,tokens=40000) as usage:
                    response=await llm.complete_json(prompt,schema=schema,system=selected['system'],
                        stage='probe.entailment_groups',attempts=1,timeout=65,max_tokens=4096)
                parsed=answer_choice.TypedEntailments.model_validate(response)
                if len(parsed.checks)!=len(items) or {c.claim_id for c in parsed.checks}!={c['claim_id'] for c in items}:
                    raise ValueError('Verifier omitted/added/duplicated checks')
                record=dict(qa_id=args.question,repetition=rep+1,group=group,prompt=prompt,response=response,
                    tokens=usage.tokens,caveat='Exact treatment check objects preserved. All versus per-option '
                    'grouping changes only check block; surrounding original verifier guidance preserved. '
                    'No gold, expected verdicts, extra evidence or final reader/HTTPS accuracy claim.')
                out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush()
                print(json.dumps(dict(qa_id=args.question,repetition=rep+1,group=group,response=response)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','replay','prepared','capture','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--question',required=True)
    p.add_argument('--repeat',type=int,choices=(1,2,3),default=2)
    p.add_argument('--inspect',action='store_true')
    asyncio.run(main(p.parse_args()))
