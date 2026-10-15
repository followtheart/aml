"""Diagnostic structured action evidence for existing constraint pairs."""
import json
from typing import Literal
from pydantic import BaseModel,ConfigDict
class Decision(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    pair_id:str
    constraint_action_quote:str
    option_action_quote:str
    relation:Literal['same_action','different_action','venue_only','warning_against','uncertain']
class Response(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    decisions:list[Decision]
INSTRUCTION = ('For every pair, quote the exact short phrase in the constraint naming the forgotten action and its object, '
 'then quote the actual action the option recommends or asserts. Use an empty option quote if no relevant action exists. '
 'Classify the relation as same_action, different_action, venue_only, warning_against, or uncertain. '
 'Matching a place is not matching the service performed there: visiting or renting a venue is distinct from contributing unpaid labor at it. '
 'Paraphrased service such as helping maintain it can match volunteering. Preserve object, time and qualifications; do not combine unrelated clauses. '
 'Use same_action only when the quoted option activity relies on or recommends the specific forgotten activity. '
 'A warning against it is warning_against. Quote literally from the original texts; never invent an action. '
 'Return decisions with pair_id, constraint_action_quote, option_action_quote, relation. Do not return violates.\n')
def pairs(prompt):return json.loads(prompt.split('<pairs>\n',1)[1].split('\n</pairs>',1)[0])
def transform(prompt):
    return prompt.split('Call emit_json_result with decisions,')[0]+INSTRUCTION

def normalize(payload,expected):
    parsed=Response.model_validate(payload);byid={p['pair_id']:p for p in expected}
    if len(parsed.decisions)!=len(byid) or {d.pair_id for d in parsed.decisions}!=set(byid):raise ValueError('Pair population changed')
    out=[]
    for d in parsed.decisions:
        p=byid[d.pair_id]
        if d.constraint_action_quote not in p['constraint_quote'] or d.option_action_quote not in p['option_quote']:raise ValueError('Action quote not literal')
        if d.relation=='same_action' and (not d.constraint_action_quote.strip() or not d.option_action_quote.strip()):raise ValueError('Positive match requires two literal witnesses')
        out.append(dict(pair_id=d.pair_id,violates=d.relation=='same_action'))
    return dict(decisions=out)
