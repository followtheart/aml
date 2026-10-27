"""Ablate fused-head reservations with frozen retrieval and cross-encoder scores."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from replay_https_ranking_budget import run_ranking
from summarize_reader_replay import read_complete
from app import config, eval_scoring


async def main(args):
    if config.FAKE:
        raise ValueError('Real downstream calls required')
    requested = set(args.questions.split(','))
    results = [r for r in read_complete(args.run/'completed/data/results/personamem-v2-32k.jsonl', True)
               if r['qa_id'] in requested]
    if {r['qa_id'] for r in results} != requested:
        raise ValueError('Missing requested result')
    traces = {t['search_id']:t for t in read_complete(args.run/'server-search-debug.jsonl', True)}
    questions = {(d['conversation_id'],q['id']):q for d in read_complete(args.prepared) for q in d['qa']}
    original_reserve = config.CASCADE_FUSED_RESERVE
    if original_reserve != 3:
        raise ValueError('Expected baseline three fused reservations')
    for r in results:
        trace = traces[r['search_id']]
        check = await run_ranking(trace, trace['versions']['settings']['RERANK_MAX_PROMPT_BYTES'], frozen=True)
        if check['packet_hash'] != r['packet_hash'] or check['packet_hash'] != trace['packet_hash']:
            raise ValueError('Frozen baseline packet reproduction failed')
    plan = dict(run=str(args.run),questions=[r['qa_id'] for r in results],repetitions=args.repeat,
                intervention='CASCADE_FUSED_RESERVE: 3 -> 0',frozen_packets_verified=True,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                rationale='Selected packet-loss cases; same generic policy on every candidate, no target-source insertion.',
                caveat='Frozen retrieval/CE, live listwise and reader; upstream costs omitted. Not HTTPS accuracy.')
    args.output.with_suffix('.plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with args.output.open('x') as out:
        for repetition in range(args.repeat):
            for index,r in enumerate(results):
                trace=traces[r['search_id']]
                limit=trace['versions']['settings']['RERANK_MAX_PROMPT_BYTES']
                for enabled in ([False,True] if (index+repetition)%2==0 else [True,False]):
                    reserve=0 if enabled else original_reserve
                    with patch.object(config,'CASCADE_FUSED_RESERVE',reserve):
                        row=await run_ranking(trace,limit)
                    prediction,score,diagnostics=await eval_scoring.evaluate(questions[r['conversation_id'],r['qa_id']],row['packet'])
                    row.update(qa_id=r['qa_id'],conversation_id=r['conversation_id'],repetition=repetition+1,
                               enabled=enabled,fused_reserve=reserve,prediction=prediction,score=score,
                               diagnostics=diagnostics,baseline_packet_hash=r['packet_hash'],frozen_packet_verified=True)
                    out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
                    print(json.dumps({k:row[k] for k in ('qa_id','repetition','enabled','prediction','score')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--prepared',type=Path,default=Path('data/prepared/personamem-v2-32k.jsonl'))
    p.add_argument('--questions',required=True)
    p.add_argument('--repeat',type=int,choices=(1,2,3),default=2)
    p.add_argument('--output',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
