"""Development on already tested LoCoMo cases; never reads held-out LME answers."""
import asyncio
import json
import logging
from pathlib import Path
from .grounding_benchmark import settings,read_and_score,ROOT
from .guarded_benchmark import PreparedQueryProvider
from .real_benchmark import CONTEXT,append,LegacyUsage
from .benchmark_metrics import common_packet
from memory_connection.app.providers import HTTPProvider
from memory_connection.app.service import MemoryService
from memory_connection.app.schemas import SearchRequest
from memory_connection.app.store import Store


async def main():
    root=Path(ROOT);old=Path('memory_connection/runs/optimization-v4-holdout-20261024')
    cfg=settings();sink=lambda x:append(root/'development-calls.jsonl',{**CONTEXT.get(),**x})
    provider=HTTPProvider(cfg,sink);handler=LegacyUsage(sink);logger=logging.getLogger('aml.metrics');logger.setLevel(logging.INFO);logger.addHandler(handler)
    plan={c['conversation_id']:c for c in json.loads((old/'plan.json').read_text(encoding='utf-8'))};gate=asyncio.Semaphore(3)
    async def case(cid,qid):
        async with gate:
            c=plan[cid];qa=next(q for q in c['qa'] if q['qa_id']==qid)
            prep=json.loads((old/'query-preparations'/f'{cid}-{qid}.json').read_text(encoding='utf-8'))
            qa={**qa,'question_date':prep['query']['reference_time']};store=Store(str(old/f'{cid}-C1.db'),provider.space_id)
            try:
                CONTEXT.set({'phase':'development_search','conversation_id':cid,'qa_id':qid})
                response=await MemoryService(store,PreparedQueryProvider(provider,prep),cfg).search(SearchRequest(user_id=cid,query=qa['question'],debug=True))
                packet=common_packet(response['data'],24000);out=[]
                for reader in ('v4','v5'):
                    CONTEXT.set({'phase':'development_reader','reader':reader,'conversation_id':cid,'qa_id':qid})
                    result=await read_and_score(provider,qa,packet,reader);out.append({'reader':reader,**result})
                append(root/'development.jsonl',{'conversation_id':cid,'qa_id':qid,'question':qa['question'],'gold':qa['answer'],'response':response,'packet':packet,'readers':out})
                print(json.dumps({'case':cid+'/'+qid,'gold':qa['answer'],'readers':[{k:r[k] for k in ('reader','prediction','score','reader_errors')} for r in out],'verification_errors':response['verification_errors']},ensure_ascii=False),flush=True)
            finally:store.close()
    try:await asyncio.gather(*(case(c,q) for c,q in [('conv-42','2'),('conv-42','22'),('conv-42','34'),('conv-42','84'),('conv-44','34'),('conv-44','47')]))
    finally:await provider.aclose();logger.removeHandler(handler)


if __name__=='__main__':asyncio.run(main())
