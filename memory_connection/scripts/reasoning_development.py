"""Development only: reuse previous evaluated histories, never held-out examples."""
import asyncio
import json
import logging
import argparse
from pathlib import Path
from .optimization_benchmark import settings_from_file, PreparedQueryProvider
from .real_benchmark import append, CONTEXT, LegacyUsage
from .benchmark_metrics import common_packet
from memory_connection.app.providers import HTTPProvider
from memory_connection.app.service import MemoryService
from memory_connection.app.store import Store
from memory_connection.app.schemas import SearchRequest


async def main(cases, tag, output):
    root=Path(output)
    old=Path('memory_connection/runs/optimization-holdout-20261024')
    cfg=settings_from_file('memory_connection/.env')
    def sink(row):append(root/'development-calls.jsonl',{**CONTEXT.get(),**row})
    provider=HTTPProvider(cfg,sink)
    handler=LegacyUsage(sink);logger=logging.getLogger('aml.metrics');logger.setLevel(logging.INFO);logger.addHandler(handler)
    from memory_system.app import eval_scoring
    plan={c['conversation_id']:c for c in json.loads((old/'plan.json').read_text(encoding='utf-8'))}
    gate=asyncio.Semaphore(3)
    async def evaluate(cid,qid):
        async with gate:
            CONTEXT.set({'arm':'development/C11','phase':'search','conversation_id':cid,'qa_id':qid})
            qa=next(q for q in plan[cid]['qa'] if q['qa_id']==qid)
            prepared=json.loads((old/'query-preparations'/f'{cid}-{qid}.json').read_text(encoding='utf-8'))
            store=Store(str(old/f'{cid}-C1.db'),provider.space_id)
            try:
                response=await MemoryService(store,PreparedQueryProvider(provider,prepared),cfg).search(
                    SearchRequest(user_id=cid,query=qa['question'],top_k=10,debug=True))
                packet=common_packet(response['data'],24000)
                CONTEXT.set({'arm':'development/C11','phase':'answer_judge','conversation_id':cid,'qa_id':qid})
                prediction,score,diagnostics=await eval_scoring.evaluate(qa,[{'id':str(i),'content':json.dumps(p,ensure_ascii=False)} for i,p in enumerate(packet)])
                found={s['message_id'] for p in packet for s in p['sources']}
                row={'development_round':tag,'conversation_id':cid,'qa_id':qid,'question':qa['question'],'gold':qa['answer'],
                     'prediction':prediction,'score':score,'gold_found':sorted(set(qa['evidence'])&found),
                     'gold_ids':qa['evidence'],'diagnostics':diagnostics,'response':response,'packet':packet}
                append(root/'development.jsonl',row)
                print(json.dumps({k:v for k,v in row.items() if k not in {'response','packet','diagnostics'}},ensure_ascii=False),flush=True)
            finally:store.close()
    try:
        selected=[
            ('conv-41','88'),('conv-41','128'),('conv-41','131'),('conv-47','128'),
            ('conv-26','11'),('conv-26','36'),('conv-26','79')]
        if cases:selected=[x for x in selected if '/'.join(x) in cases]
        await asyncio.gather(*(evaluate(cid,qid) for cid,qid in selected))
    finally:await provider.aclose();logger.removeHandler(handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cases',nargs='*');parser.add_argument('--tag',default='final')
    parser.add_argument('--output',default='memory_connection/runs/optimization-v4-holdout-20261024')
    args=parser.parse_args();asyncio.run(main(args.cases,args.tag,args.output))
