"""Synthetic option-only premise extraction probe; never supplies expected labels."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import config, llm
from pydantic import BaseModel,ConfigDict,Field

class Entry(BaseModel):
    model_config=ConfigDict(extra='forbid')
    id: str
    personal_premises: list[str]=Field(description='Exact substrings asserting existing distinguishing personal facts. Advice and conditionals are not personal history.')
class Output(BaseModel):
    model_config=ConfigDict(extra='forbid')
    entries: list[Entry]

CASES=[
 ('conditional','How can I respond thoughtfully to difficult news?', 'When a teammate shares upsetting news, listen quietly and suggest a break. Give yourself a moment to breathe afterward.',False),
 ('past_event','How can I respond thoughtfully to difficult news?', 'Because you helped your teammate through a crisis last winter, draw on that experience.',True),
 ('draft_advice','Suggest a sentence I can say to myself when I feel discouraged.', 'I remind myself that change takes time, and focus on the effort I can make today.',False),
 ('habit','Suggest a sentence I can say to myself when I feel discouraged.', 'Since you already write in your journal every night, remind yourself of the progress recorded there.',True),
 ('object','How can I decorate a desk?', 'You could add a plant to your desk and arrange a few photos so the space feels welcoming.',False),
 ('ownership','How can I decorate a desk?', 'Since you already own an antique desk, consider a lamp that matches its wood.',True),
 ('embedded_fact','What hobby could I try?', 'Try making a display of the vintage cameras you already collect.',True),
 ('future_frequency','What hobby could I try?', 'You could spend ten minutes drawing every evening before bed.',False),
 ('negated_history','What hobby could I try?', 'Because you have never tried pottery, a beginner class could be enjoyable.',True),
 ('hypothetical','How should I choose a weekend activity?', 'If you owned a kayak, you could explore the lake; otherwise try a walk.',False),
 ('medical_history','How should I choose a weekend activity?', 'Given your old knee injury, consider a gentle walk.',True),
 ('quoted_third_party','Give me a sample supportive message to send to a friend.', 'You could say: "I know how hard your recovery has been. I am here to listen."',False),
]

async def main(args):
    if config.FAKE: raise RuntimeError('Requires real model')
    config.SEARCH_DEBUG_LOG=config.MEMORY_DEBUG_LOG=''
    with args.output.open('x') as f:
        for offset in range(0,len(CASES),4):
            cases=CASES[offset:offset+4]
            prompt='''Identify only existing personal facts that each proposed answer assumes about the current user.
This is a discourse task, not evidence support or answer selection. Do not search for sources.
Read the question to tell suggested wording from a report of an existing habit. A recommended first-person sentence is not autobiography.
Conditional future scenarios, advice, intended effects, and general knowledge require no personal premises.
Do preserve ownership, habits, past events, conditions, negation, frequency, and personal facts embedded in recommendations.
An unsupported personal assumption is still a premise: never erase it because it lacks proof. Do not transfer a quoted friend's history to the user.
Return every id exactly once. Each personal_premises item must be a contiguous exact substring from that answer. Return [] only if no existing distinguishing personal fact about the user is assumed.
Inputs:\n'''+json.dumps([dict(id=c[0],question=c[1],answer=c[2]) for c in cases])
            row=dict(prompt=prompt,model=config.LLM_MODEL)
            try:
                response=await llm.complete_json(prompt,schema=Output.model_json_schema(),stage='probe.premise_boundary',attempts=1,timeout=60,max_tokens=1600)
                parsed=Output.model_validate(response)
                entries={e.id:e.personal_premises for e in parsed.entries}
                assert len(entries)==len(parsed.entries) and set(entries)=={c[0] for c in cases}
                for c in cases: assert all(p and p in c[2] for p in entries[c[0]])
                row.update(response=response,checks=[dict(id=c[0],expected_personal=c[3],actual_personal=bool(entries[c[0]]),matches=bool(entries[c[0]])==c[3]) for c in cases])
            except Exception as exc:row.update(error_type=type(exc).__name__,error=str(exc)[:1000])
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            print(json.dumps(row.get('checks',dict(error=row.get('error')))),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    asyncio.run(main(ap.parse_args()))
