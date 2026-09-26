"""Pass recovered primary-claim proposals through the existing entailment prompt."""
import asyncio,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac,config,llm,prompts

async def main():
    config.SEARCH_DEBUG_LOG=config.MEMORY_DEBUG_LOG=''
    root=Path('runs/iteration-20260924-09')
    original={r['qa_id']:r for r in map(json.loads,Path('runs/iteration-20260924-07/completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines())}
    with (root/'recovered-citation-entailment.jsonl').open('x') as out:
        for probe in map(json.loads,(root/'claim-citation-probe.jsonl').read_text().splitlines()):
            r=original[probe['qa_id']];sources={s['id']:s for s in r['answer_source_catalog']}
            byid={c['claim_id']:c for c in probe['claims']}; proposed={m['claim_id']:m for m in probe['response']['matches']}
            entries=[]; rejected=[]
            for m in probe['checked']:
                if not m['supported']:continue
                if not m['anchored'] or not all(c['valid'] for c in m['citations']):
                    rejected.append(m['claim_id']);continue
                c=byid[m['claim_id']];letter=m['claim_id'].split(':')[0]
                entry=ac.validate_assessments({'options':[dict(letter=letter,kind='personal',claims=[dict(text=c['text'],status='supported',premise_type=m['premise_type'],reason='none',citations=proposed[m['claim_id']]['citations'])])]},[c['option']],sources)[0]
                entries.append(entry)
            options=[e['option'] for e in entries]
            checks=ac.entailment_checks(entries,sources,options)
            # Use the actual benchmark question, not a replacement task question.
            prepared=[json.loads(l) for l in Path('data/prepared/personamem-v2-32k.jsonl').read_text().splitlines()]
            q=next(q for d in prepared if d['conversation_id']==r['conversation_id'] for q in d['qa'] if q['id']==r['qa_id'])
            prompt=prompts.render('15_choice_entailment.txt',question=q['question'],question_date=q.get('question_date',''),checks=json.dumps(checks,ensure_ascii=False))
            response=await llm.complete_json(prompt,schema=ac.Entailments.model_json_schema(),stage='probe.recovered_entailment',attempts=1,timeout=60,max_tokens=1200)
            ac._verdicts(response,checks)
            row=dict(qa_id=r['qa_id'],prompt=prompt,response=response,rejected_invalid_citation_claims=rejected,
                     limitation='Focused primary-claim probe, not a full reconstructed answer flow or accuracy score.')
            out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();print(json.dumps(dict(qa_id=r['qa_id'],response=response,rejected=rejected)),flush=True)

asyncio.run(main())
