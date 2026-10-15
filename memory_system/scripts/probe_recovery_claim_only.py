"""Fixed-prefix citation recovery diagnostic: premise text without option advice."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from replay_source_vector_route import source_route
from replay_query_context import records, recorded_response
from app import budget, config, eval_scoring, evidence_packet, llm


def focus(prompt):
    marker='\nClaims:\n'; ending='\nOriginal source cards:\n'
    if prompt.count(marker)!=1 or prompt.count(ending)!=1:
        raise ValueError('Unexpected recovery prompt structure')
    prefix, rest=prompt.split(marker); payload, suffix=rest.split(ending)
    claims=json.loads(payload)
    if any(set(c)!={'claim_id','text','option'} for c in claims):
        raise ValueError('Unexpected recovery claim fields')
    return prefix+marker+json.dumps([{k:c[k] for k in ('claim_id','text')} for c in claims],ensure_ascii=False)+ending+suffix


async def main(args):
    if config.FAKE:raise ValueError('Real provider required')
    archived=next(r for r in records(args.replay) if r['qa_id']==args.question and r['arm']=='window')
    probe=json.loads(args.probe.read_text());q=next(q for q in probe['questions'] if q['qa_id']==args.question)
    trace=next(t for t in records(args.run/'server-search-debug.jsonl') if t['search_id']==q['search_id'])
    packet,_=source_route(probe,q,trace['returned'],trace['evidence_token_budget'])
    if evidence_packet.digest(packet)!=archived['packet_hash']:raise ValueError('Packet changed')
    qa=next(qa for d in records(args.prepared) for qa in d['qa'] if qa['id']==args.question)
    calls=archived['diagnostics']['answer_calls'];boundary=next(i for i,c in enumerate(calls) if c['stage']=='eval.choice_citation_recovery')
    original=llm.complete_json

    async def run(enabled,frozen=False):
        pending=copy.deepcopy(calls if frozen else calls[:boundary]);live=[];violations=[]
        async def provider(prompt,*positional,**kwargs):
            if pending:
                old=pending.pop(0)
                if old['stage']!=kwargs['stage'] or old['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest():
                    violations.append(kwargs['stage']);raise ValueError('Frozen prompt mismatch')
                budget.current.get().before_call()
                return recorded_response(old)
            if frozen:
                violations.append('unexpected_live_call');raise ValueError('Network forbidden during reproduction')
            actual=focus(prompt) if enabled and kwargs['stage']=='eval.choice_citation_recovery' else prompt
            call=dict(stage=kwargs['stage'],prompt=actual,prompt_sha256=hashlib.sha256(actual.encode()).hexdigest());live.append(call)
            response=await original(actual,*positional,**kwargs);call['response']=response
            return response
        with patch.object(config,'CHOICE_QUERY_CONTEXT',False),patch.object(llm,'complete_json',provider):
            prediction,score,diagnostics=await eval_scoring.evaluate(qa,copy.deepcopy(packet))
        if pending or violations:raise ValueError('Frozen trajectory mismatch')
        return dict(qa_id=args.question,enabled=enabled,prediction=prediction,score=score,
                    diagnostics=diagnostics,live_calls=live,packet_hash=archived['packet_hash'])

    check=await run(False,True)
    if (check['prediction'],check['score'])!=(archived['prediction'],archived['score']):
        raise ValueError('Frozen outcome mismatch')
    args.output.with_suffix('.plan.json').write_text(json.dumps(dict(qa_id=args.question,
        frozen_verified=True,intervention='Remove option field from recovery claims only; claim IDs/text and all sources unchanged.',
        caveat='Selected diagnostic, fixed prefix and packet, live recovery/downstream. No general accuracy inference.'),indent=2)+'\n')
    with args.output.open('x') as stream:
        for rep in range(getattr(args, 'repetitions', 2)):
            for enabled in ([False,True] if (rep + getattr(args, 'order_offset', 0)) % 2 == 0 else [True,False]):
                row=await run(enabled);row['repetition']=rep+1
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                print(json.dumps({k:row[k] for k in ('repetition','enabled','prediction','score')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','probe','replay','prepared','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--question',required=True)
    p.add_argument('--repetitions',type=int,default=2)
    p.add_argument('--order-offset',type=int,default=0)
    asyncio.run(main(p.parse_args()))
