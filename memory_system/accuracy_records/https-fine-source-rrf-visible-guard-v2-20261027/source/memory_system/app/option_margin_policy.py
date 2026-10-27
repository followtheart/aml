"""Source ranking by option-score range; invariant to option order."""
import json
import math

PREFIX='Original user message (including its outer task and qualifications):\n'


def query(question,alternative):
    return question+'\n\nRetrieval target: the following is one possible answer, not an established fact about the user. Retrieve original context that could support or contradict its details; preserve speaker, supplied-text attribution and negation.\n'+alternative


def rank(nominees,scores,option_count):
    expected=set(range(option_count))
    if option_count<2 or set(scores)!=expected:raise ValueError('Incomplete option score matrix')
    keys={n['source_view_id'] for n in nominees}
    for values in scores.values():
        if set(values)!=keys or any(type(v) not in (int,float) or not math.isfinite(v) for v in values.values()):
            raise ValueError('Incomplete or invalid source scores')
    rows=[]
    for n in nominees:
        values=[scores[i][n['source_view_id']] for i in range(option_count)]
        rows.append(dict(n,option_score_range=max(values)-min(values),max_option_score=max(values),option_scores=values))
    rows.sort(key=lambda n:(-n['option_score_range'],-n['max_option_score'],n['id'],json.dumps(n['unit']['_rank_context'],sort_keys=True)))
    return rows
