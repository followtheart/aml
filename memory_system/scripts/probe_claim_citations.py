"""Frozen-packet probe of missing-citation recovery, with no answer labels."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Literal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac, config, llm
from pydantic import BaseModel,ConfigDict,Field

class Match(BaseModel):
    model_config=ConfigDict(extra='forbid')
    claim_id:str
    premise_type:Literal['interest','ownership','condition','habit','experience','occupation','location','unknown']
    supported:bool
    citations:list[ac.Citation]=Field(max_length=3)
class Matches(BaseModel):
    model_config=ConfigDict(extra='forbid')
    matches:list[Match]

async def main(args):
    if config.FAKE:raise RuntimeError('Probe requires a provider')
    config.SEARCH_DEBUG_LOG=config.MEMORY_DEBUG_LOG=''
    results={r['qa_id']:r for r in map(json.loads,args.results.read_text().splitlines())}
    with args.output.open('x') as f:
        for qid in args.questions.split(','):
            r=results[qid];sources={s['id']:s for s in r['answer_source_catalog']}
            claims=[]
            for entry in r['choice_alignment']:
                i=entry.get('primary_claim')
                if i is None or entry['status']!='unsupported':continue
                c=entry['claims'][i]
                if c.get('citations'):continue
                claims.append(dict(claim_id=entry['letter']+':'+str(i),text=c['text'],option=entry['option']))
            if not claims:continue
            prompt='''Find original evidence for each given personal premise; do not select or rank answers, and do not extract other claims.
For each claim_id return whether the source establishes that premise, its premise_type, and up to three exact contiguous quotations with source IDs.
Read all supplied original sources. Match meaning and paraphrases, not only identical words. A described activity can establish an experience without naming it.
Preserve the subject, negation, time and strength: topical curiosity can establish interest or attention, but a question does not establish ownership, a diagnosis, a daily habit or a past event. Conditional ownership is not actual ownership. Do not turn suggestions into history or borrow third-party experiences.
Monitoring a health measurement asserts attention to the measurement, not diagnosis. A user question explicitly about their measurement can establish attention.
Only user or authoritative first-party persona facts may anchor personal premises. Assistant rewrites/advice are context, not independent evidence.
Copy citations exactly. Do not use the option as its own evidence. If no source establishes the premise, return supported=false. Return every requested claim_id exactly once.
Claims:\n'''+json.dumps(claims,ensure_ascii=False)+'\nOriginal source cards:\n'+json.dumps(ac._cards(sources),ensure_ascii=False)
            row=dict(qa_id=qid,packet_hash=r['packet_hash'],claims=claims,prompt=prompt,model=config.LLM_MODEL)
            try:
                response=await llm.complete_json(prompt,schema=Matches.model_json_schema(),stage='probe.claim_citations',max_tokens=2400,timeout=60,attempts=1)
                parsed=Matches.model_validate(response); by_id={c['claim_id']:c for c in claims}
                assert len(parsed.matches)==len(by_id) and {m.claim_id for m in parsed.matches}==set(by_id)
                checked=[]
                for m in parsed.matches:
                    refs=[ac._check_citation(c,by_id[m.claim_id]['text'],sources,m.premise_type) for c in m.citations]
                    checked.append(dict(claim_id=m.claim_id,supported=m.supported,premise_type=m.premise_type,citations=refs,
                        anchored=any(c['valid'] and c['anchor'] for c in refs)))
                row.update(response=response,checked=checked)
            except Exception as exc:row.update(error_type=type(exc).__name__,error=str(exc)[:1000])
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            print(json.dumps(dict(qa_id=qid,judgments=[{k:c[k] for k in ['claim_id','supported','premise_type','anchored']} for c in row.get('checked',[])],error=row.get('error'))),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,required=True);ap.add_argument('--questions',default='15,25,4,18')
    ap.add_argument('--output',type=Path,required=True)
    asyncio.run(main(ap.parse_args()))
