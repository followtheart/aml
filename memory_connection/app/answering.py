"""Optional evidence reader, independent of the legacy system and of retrieval.

Reference validation checks provenance, not semantic truth. Missing premises are
reported; generated answers are never persisted as memories.
"""
import json
import re
from datetime import datetime
from itertools import combinations
from pydantic import Field
from .schemas import Strict
from .temporal import source_time


class GroundedAnswer(Strict):
    answer: str = Field(min_length=1,max_length=5000)
    source_refs: list[str] = Field(default_factory=list,max_length=32)
    missing_premises: list[str] = Field(default_factory=list,max_length=12)


def reading_packet(items, budget=24000):
    sources, seen, used = [], set(), 2
    for item in items:
        spans=item.get('evidence_spans',[])
        for s in item.get('sources',[]):
            key=(s.get('session_id'),s.get('message_id'),s['source_id'])
            if key in seen:continue
            entry={'ref':f's{len(sources)}','speaker':s.get('speaker_id'),'role':s.get('role'),
                   **source_time(s),'original_text':s['content'],
                   'verified_quotes':[x['quote'] for x in spans if x['source_id']==s['source_id']]}
            size=len(json.dumps(entry,ensure_ascii=False).encode())+2
            if used+size>budget:continue
            sources.append(entry);seen.add(key);used+=size
    return sources


def date_arithmetic(question,sources,budget):
    """Calculated differences of stated dates, never inferred event endpoints."""
    if not re.search(r'\b(?:how (?:many|long)|elapsed|between)\b',question,re.I):return []
    available=[s for s in sources if s['stated_at']]
    pairs=sorted(combinations(available,2),key=lambda p:-(bool(p[0]['verified_quotes'])+bool(p[1]['verified_quotes'])))
    result=[];seen=set();used=len(json.dumps(sources,ensure_ascii=False).encode())
    for a,b in pairs:
        start,end=sorted((a,b),key=lambda s:s['stated_at'])
        first=datetime.fromisoformat(start['stated_at']).date();last=datetime.fromisoformat(end['stated_at']).date()
        if first==last or (first,last) in seen:continue
        months=(last.year-first.year)*12+last.month-first.month
        item={'source_refs':[start['ref'],end['ref']],'start_stated_date':first.isoformat(),
              'end_stated_date':last.isoformat(),'calendar_month_difference':months,
              'completed_months':months-int(last.day<first.day),'elapsed_days':(last-first).days,
              'scope':'source statement dates only; bind to actual events before using'}
        size=len(json.dumps(item).encode())+2
        if used+size>budget:continue
        result.append(item);used+=size;seen.add((first,last))
        if len(result)>=32:break
    return result


async def answer(provider, question, items, question_date=None, budget=24000):
    # Reserve the JSON envelope, not just the nested source/calculation arrays.
    evidence_budget=max(0,budget-64)
    sources=reading_packet(items,evidence_budget)
    if not sources:
        return {'answer':'Unknown','source_refs':[],'missing_premises':['No verified evidence'],
                'reference_status':'no_evidence','sources':[]}
    raw=await provider.complete('answer_grounded_v1',
        {'question':question,'question_date':question_date,'sources':sources,
         'date_arithmetic':date_arithmetic(question,sources,evidence_budget)},GroundedAnswer)
    parsed=GroundedAnswer.model_validate(raw)
    refs={s['ref'] for s in sources}
    if set(parsed.source_refs)-refs:raise ValueError('unknown answer source reference')
    if parsed.answer.casefold()!='unknown' and not parsed.source_refs:raise ValueError('answer has no cited sources')
    return {**parsed.model_dump(),'reference_status':'references_valid','sources':sources}
