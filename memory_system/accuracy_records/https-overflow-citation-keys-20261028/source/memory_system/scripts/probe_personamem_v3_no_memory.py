"""Run the identical v3 proxy Answer/Judge on an empty memory packet."""
import argparse
import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import config, eval_scoring
from app.eval_data import read_records
from app.personamem_v3 import PROTOCOL

async def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=list(read_records(a.prepared))
    if any(c.get('protocol')!=PROTOCOL for c in rows):raise ValueError('Expected v3 chatbot proxy')
    with a.output.open('x') as out:
        for c in rows:
            for q in c['qa']:
                pred,score,diag=await eval_scoring.evaluate(q,[])
                out.write(json.dumps(dict(conversation_id=c['conversation_id'],qa_id=q['id'],prediction=pred,
                                          score=score,protocol=PROTOCOL,model=config.LLM_MODEL,fake=config.FAKE,
                                          condition='no_memory',**diag),ensure_ascii=False)+'\n');out.flush()
if __name__=='__main__':asyncio.run(main())
