"""Run review-scaffolding diagnostics across an entire archived cohort."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from probe_review_scaffolding import main as probe_main
from replay_query_context import records


async def main(args):
    args.output_dir.mkdir(parents=True,exist_ok=False)
    if args.replay:
        rows=[r for r in records(args.replay) if r['enabled']]
    else:
        originals=records(args.run/'completed/data/results/personamem-v2-32k.jsonl')
        rows=[dict(qa_id=r['qa_id'],enabled=True,prediction=r['prediction'],score=r['score'],
                   packet_hash=r['packet_hash'],diagnostics=r,
                   context_enabled=r['versions']['settings'].get('CHOICE_QUERY_CONTEXT',False)) for r in originals]
    if len({r['qa_id'] for r in rows})!=len(rows):raise ValueError('Require exactly one archived trajectory per question')
    replay=args.output_dir/'input.jsonl'
    replay.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    plan=dict(run=str(args.run),input=str(replay),input_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(),
              questions=[r['qa_id'] for r in rows],repetitions=1,request_only=args.request_only,selection='Every question in input cohort, no score selection',
              probe=str(args.probe) if args.probe else None,
              scripts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__),Path(__file__).with_name('probe_review_scaffolding.py')]},
              caveat='Frozen all reader calls through first selection; only review stage rerun. Not full live reader/HTTPS.')
    (args.output_dir/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for index,row in enumerate(rows):
        await probe_main(argparse.Namespace(run=args.run,replay=replay,probe=args.probe,prepared=args.prepared,
            question=row['qa_id'],output=args.output_dir/(row['qa_id']+'.jsonl'),repeat=1,reverse=bool(index%2),request_only=args.request_only))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--replay',type=Path);p.add_argument('--probe',type=Path)
    p.add_argument('--prepared',type=Path,default=Path('data/prepared/personamem-v2-32k.jsonl'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--request-only',action='store_true')
    asyncio.run(main(p.parse_args()))
