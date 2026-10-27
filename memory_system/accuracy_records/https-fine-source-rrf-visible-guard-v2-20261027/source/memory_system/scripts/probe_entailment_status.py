"""Fixed-check ablation of proposed_status; no source or claim changes."""
import argparse,asyncio,copy,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac,config,llm

async def main(args):
    if config.FAKE:raise RuntimeError('Requires real model')
    r=next(r for r in map(json.loads,args.input.read_text().splitlines()) if r['qa_id']==args.question)
    original=next(c['prompt'] for c in r['live_calls'] if c['stage']=='eval.choice_entailment')
    match=re.search(r'(<checks>\s*)(.*?)(\s*</checks>)',original,re.S)
    checks=json.loads(match.group(2))
    with args.output.open('x') as out:
        for repetition in range(2):
            for mode in (['with_status','without_status'] if repetition==0 else ['without_status','with_status']):
                items=copy.deepcopy(checks)
                if mode=='without_status':
                    for c in items:c.pop('proposed_status',None)
                prompt=original[:match.start(2)]+json.dumps(items,ensure_ascii=False)+original[match.end(2):]
                result=await llm.complete_json(prompt,schema=ac.Entailments.model_json_schema(),stage='probe.entailment_status',attempts=1,timeout=60,max_tokens=1200)
                ac._verdicts(result,items)
                row=dict(mode=mode,repetition=repetition+1,prompt=prompt,response=result)
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();print(json.dumps({k:row[k] for k in ['mode','repetition','response']}),flush=True)

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--question',required=True)
parser.add_argument('--output',type=Path,required=True)
asyncio.run(main(parser.parse_args()))
