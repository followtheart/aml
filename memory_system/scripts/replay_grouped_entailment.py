"""Full-reader ablation of per-option entailment, with prior calls frozen."""
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
    if config.FAKE:raise ValueError('Real downstream provider required')
    archived=next(r for r in records(args.replay) if r['enabled'])
    qid=archived['qa_id']
    result=next(r for r in records(args.run/'completed/data/results/personamem-v2-32k.jsonl') if r['qa_id']==qid)
    trace=next(r for r in records(args.run/'server-search-debug.jsonl') if r.get('search_id')==result['search_id'])
    qa=next(q for d in records(args.prepared) if d['conversation_id']==result['conversation_id'] for q in d['qa'] if q['id']==qid)
    if evidence_packet.digest(trace['returned'])!=archived['packet_hash']:raise ValueError('Packet mismatch')
    calls=archived['diagnostics']['answer_calls']
    boundary=next(i for i,c in enumerate(calls) if c['stage']=='eval.choice_entailment')
    original=llm.complete_json

    async def run(enabled,frozen=False):
        pending=copy.deepcopy(calls if frozen else calls[:boundary]);mismatches=[];live=[]
        async def provider(prompt,*positional,**kwargs):
            if pending:
                old=pending.pop(0)
                if old['stage']!=kwargs['stage'] or old['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest():
                    mismatches.append(kwargs['stage']);raise ValueError('Frozen upstream mismatch')
                budget.current.get().before_call()
                return recorded_response(old)
            if frozen:
                mismatches.append('unexpected_call');raise ValueError('Unexpected call in frozen validation')
            if enabled and kwargs['stage']=='eval.choice_entailment':
                before,rest=prompt.split('<checks>\n',1);raw,after=rest.split('\n</checks>',1)
                checks=json.loads(raw)
                groups={letter:[c for c in checks if c['claim_id'].split(':',1)[0]==letter]
                        for letter in dict.fromkeys(c['claim_id'].split(':',1)[0] for c in checks)}
                combined=[]
                for letter,items in groups.items():
                    subprompt=before+'<checks>\n'+json.dumps(items,ensure_ascii=False)+'\n</checks>'+after
                    record=dict(stage=kwargs['stage'],group=letter,prompt=subprompt)
                    live.append(record)
                    response=await original(subprompt,*positional,**kwargs)
                    record['response']=response
                    parsed=answer_choice.TypedEntailments.model_validate(response)
                    if len(parsed.checks)!=len(items) or {c.claim_id for c in parsed.checks}!={c['claim_id'] for c in items}:
                        raise ValueError('Grouped verifier changed check identities')
                    combined.extend(c.model_dump() for c in parsed.checks)
                return dict(checks=combined)
            record=dict(stage=kwargs['stage'],prompt=prompt);live.append(record)
            response=await original(prompt,*positional,**kwargs);record['response']=response
            return response
        with patch.object(llm,'complete_json',provider),budget.scope(seconds=300,calls=30,tokens=160000) as usage:
            prediction,score,diagnostics=await eval_scoring.evaluate(qa,copy.deepcopy(trace['returned']))
        if pending or mismatches:raise ValueError(dict(pending=len(pending),mismatches=mismatches))
        return dict(qa_id=qid,enabled=enabled,prediction=prediction,score=score,diagnostics=diagnostics,
                    live_calls=live,calls=usage.calls,tokens=usage.tokens,packet_hash=archived['packet_hash'])
    checked=await run(False,frozen=True)
    if checked['prediction']!=archived['prediction'] or checked['score']!=archived['score']:
        raise ValueError('Full frozen reader result mismatch')
    print(json.dumps(dict(qa_id=qid,frozen_verified=True)),flush=True)
    with args.output.open('x') as out:
        for rep in range(args.repeat):
            for enabled in ([False,True] if rep%2==0 else [True,False]):
                row=await run(enabled)
                row.update(repetition=rep+1,frozen_verified=True,
                    caveat='Selected repaired trajectory, frozen packet and calls preceding entailment. '
                    'Every check and its evidence preserved; grouped mode splits only check block by option. '
                    'Extra calls charged inside expanded budget; earlier time/tokens omitted. Not fresh HTTPS.')
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
                print(json.dumps({k:row[k] for k in ['qa_id','repetition','enabled','prediction','score']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','replay','prepared','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repeat',type=int,choices=(1,2,3),default=2)
    asyncio.run(main(p.parse_args()))
