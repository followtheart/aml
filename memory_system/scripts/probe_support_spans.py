"""Compare support extraction with/without clause hints on frozen source cards.

No gold labels enter model prompts. This probes extraction, not final accuracy.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac, choice_premises, choice_witness, config, prompts


async def main(args):
    if config.FAKE:
        raise RuntimeError('Real support probe requires a provider')
    rows = {r['qa_id']: r for r in map(json.loads, args.results.read_text().splitlines())}
    qa = {(c['conversation_id'], q['id']): q for c in map(json.loads,args.prepared.read_text().splitlines()) for q in c['qa']}
    config.MEMORY_DEBUG_LOG = config.SEARCH_DEBUG_LOG = ''
    with args.output.open('x') as output:
        for qid in args.questions.split(','):
            row=rows[qid]; q=qa[row['conversation_id'],qid]
            sources={s['id']:s for s in row['answer_source_catalog']}
            options=q['options']
            base=prompts.render('12_choice_support.txt',question=q['question'],question_date=q.get('question_date',''),
                task_instructions=q.get('system_prompt',''),options='\n'.join(options),
                witnesses=choice_witness.render(row.get('answer_witness_prefill',[])),
                sources=json.dumps(ac._cards(sources),ensure_ascii=False))
            suffix=('\nOriginal option spans (optional exact-text references; choose span_id instead of rewriting '
                'a clause, omit text when using the ID; keep subject, negation and time from the complete option):\n'
                +json.dumps([s for letter,text in ac.option_map(options).items() for s in choice_premises.option_spans(letter,text)],ensure_ascii=False))
            for mode in ('with_spans','without_spans'):
                prompt=base+(suffix if mode=='with_spans' else '')
                schema=ac.Assessments.model_json_schema()
                schema['$defs']['Claim']['properties']['status']['enum']=['supported','unsupported']
                diagnostics={}
                result=dict(qa_id=qid,mode=mode,prompt=prompt,model=config.LLM_MODEL,
                            frozen_packet_hash=row['packet_hash'],input_sha256=hashlib.sha256(args.results.read_bytes()).hexdigest())
                try:
                    result['entries']=await ac.assess_support(prompt,schema,options,sources,diagnostics)
                except Exception as exc:
                    result.update(error_type=type(exc).__name__,error_detail=str(exc)[:1000])
                result['diagnostics']=diagnostics
                output.write(json.dumps(result,ensure_ascii=False)+'\n'); output.flush()
                print(json.dumps(dict(qa_id=qid,mode=mode,calls=len(diagnostics.get('answer_calls',[])),
                    statuses={e['letter']:e['status'] for e in result.get('entries',[])})),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--prepared',type=Path,default=Path('data/prepared/personamem-v2-32k.jsonl'))
    ap.add_argument('--questions',default='1,11,19,4,18,24')
    ap.add_argument('--output',type=Path,required=True)
    asyncio.run(main(ap.parse_args()))
