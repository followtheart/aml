"""Test query-only original-source focus before support with a frozen witness."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from replay_query_context import recorded_response
from summarize_reader_replay import read_complete
from app import budget, choice_context, config, eval_scoring, evidence_packet, llm


async def main(args):
    if config.FAKE:
        raise ValueError('Real provider required')
    archived=next(r for r in read_complete(args.replay) if r['qa_id']==args.question and r['enabled'] and r['repetition']==1)
    packet=archived['packet']
    if evidence_packet.digest(packet)!=archived['packet_hash']:
        raise ValueError('Archived packet changed')
    qa=next(q for d in read_complete(args.prepared) if d['conversation_id']==archived['conversation_id'] for q in d['qa'] if q['id']==args.question)
    calls=archived['diagnostics']['answer_calls']
    boundary=next(i for i,c in enumerate(calls) if c['stage']=='eval.choice_support')
    sources={s['id']:s for s in archived['diagnostics']['answer_source_catalog']}
    original=llm.complete_json

    async def run(enabled,frozen=False):
        pending=copy.deepcopy(calls if frozen else calls[:boundary])
        live=[];mismatches=[];focus_diagnostics={};focus_text=''
        async def provider(prompt,*positional,**kwargs):
            nonlocal focus_text
            if pending:
                old=pending.pop(0)
                if old['stage']!=kwargs['stage'] or old['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest():
                    mismatches.append(kwargs['stage']);raise ValueError('Frozen stage/prompt changed')
                budget.current.get().before_call()
                return recorded_response(old)
            if frozen:
                mismatches.append('unexpected_call');raise ValueError('Unexpected call in frozen check')
            if enabled and kwargs['stage']=='eval.choice_support':
                with patch.object(config,'CHOICE_QUERY_CONTEXT',True):
                    focused=await choice_context.select(qa['question'],sources,focus_diagnostics)
                focus_text=choice_context.render(focused)
            if enabled and kwargs['stage'] in ('eval.choice_support','eval.choice_support.repair'):
                prompt+=focus_text
            call=dict(stage=kwargs['stage'],prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest());live.append(call)
            response=await original(prompt,*positional,**kwargs);call['response']=response
            return response
        with patch.object(llm,'complete_json',provider):
            prediction,score,diagnostics=await eval_scoring.evaluate(qa,copy.deepcopy(packet))
        if pending or mismatches:
            raise ValueError(dict(pending=len(pending),mismatches=mismatches))
        return dict(qa_id=args.question,enabled=enabled,prediction=prediction,score=score,
                    packet_hash=archived['packet_hash'],diagnostics=diagnostics,live_calls=live,focus=focus_diagnostics)

    check=await run(False,True)
    if (check['prediction'],check['score'])!=(archived['prediction'],archived['score']):
        raise ValueError('Frozen full-reader outcome differs')
    args.output.with_suffix('.plan.json').write_text(json.dumps(dict(question=args.question,replay=str(args.replay),
        replay_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest(),packet_hash=archived['packet_hash'],
        repeat=args.repeat,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        policy='Freeze original witness; query-only focus adds exact existing sources to support and repair prompts; all subsequent calls live.',
        caveat='Selected packet-restoration diagnostic, no cohort/HTTPS claim.'),indent=2)+'\n')
    with args.output.open('x') as out:
        for rep in range(args.repeat):
            for enabled in ([False,True] if rep%2==0 else [True,False]):
                row=await run(enabled);row.update(repetition=rep+1,frozen_verified=True)
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
                print(json.dumps({k:row[k] for k in ('qa_id','repetition','enabled','prediction','score')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',type=Path,required=True);p.add_argument('--question',required=True)
    p.add_argument('--prepared',type=Path,default=Path('data/prepared/personamem-v2-32k.jsonl'))
    p.add_argument('--repeat',type=int,choices=(1,2,3),default=2)
    p.add_argument('--output',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
