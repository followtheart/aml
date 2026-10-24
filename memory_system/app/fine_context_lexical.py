"""One bounded user-source lexical seat after the unchanged two-item reserve.

Binary-term BM25 supplies retrieval ordering, never personal-fact validation.
Original sources, CE scores and all packet byte/item limits remain intact.
"""
from collections import Counter
import math

from . import budget,personal_evidence
from .conversation_context import usable
from .fine_context_reserve import propose as original_propose

POLICY='fine_user_lexical_context_v1'


def source_scores(fine,*,user_id,query,options):
    corpus={};sources_by_id={}
    for candidate in fine:
        budget.check()
        item=candidate.get('_packet_item',{})
        sources=[s for s in item.get('sources',[]) if s.get('content')]
        score=candidate.get('_ce_score')
        if (candidate.get('view_status')=='stale' or item.get('is_constraint')
                or type(score) not in (int,float) or not math.isfinite(score)
                or not sources or any(not usable(s,user_id) or s['role'] not in ('user','assistant')
                    or s['content'] not in item.get('content','') for s in sources)):
            continue
        users=[s for s in sources if s['role']=='user']
        sources_by_id[candidate['id']]=users
        for s in users:
            key=(s['user_id'],s['request_id'],s['message_index'],s['source_event_id'],s['role'])
            corpus.setdefault(key,set()).update(personal_evidence.terms(s['content']))
    query_terms=personal_evidence.terms(' '.join([query,*options]))
    frequency=Counter(term for words in corpus.values() for term in words)
    n=len(corpus);average=sum(map(len,corpus.values()))/max(1,n)
    weights={term:math.log(1+(n-count+.5)/(count+.5)) for term,count in frequency.items()}
    scores={};details={}
    for mid,sources in sources_by_id.items():
        budget.check()
        values=[]
        for s in sources:
            words=personal_evidence.terms(s['content']);hits=words&query_terms
            value=sum(weights[t] for t in sorted(hits))*2.2/(1+1.2*(.25+.75*len(words)/max(1,average)))
            values.append(dict(source_event_id=s['source_event_id'],role='user',
                               score=value,matched_terms=sorted(hits)))
        scores[mid]=max((v['score'] for v in values),default=0)
        details[mid]=values
    return scores,details


def propose(packed,fine,*,user_id,top_k,byte_budget,query,options,reserve_bytes=4096):
    first,baseline=original_propose(packed,fine,user_id=user_id,top_k=top_k,
        byte_budget=byte_budget,reserve_bytes=reserve_bytes)
    scores,matches=source_scores(fine,user_id=user_id,query=query,options=options)
    ordered=sorted([c for c in fine if scores.get(c['id'],0)>0],key=lambda c:-scores[c['id']])
    result,extra=original_propose(first,ordered,user_id=user_id,top_k=top_k,byte_budget=byte_budget,
        reserve_bytes=reserve_bytes-baseline['added_bytes'],limit=1)
    if result[:len(first)]!=first:raise ValueError('Original reserve changed')
    trace=dict(policy=POLICY,limit=3,reserve_bytes=reserve_bytes,
        original_count=len(packed),result_count=len(result),total_bytes=extra['total_bytes'],
        proposed=baseline['proposed']+extra['proposed'],
        appended_ids=baseline['appended_ids']+extra['appended_ids'],
        added_bytes=baseline['added_bytes']+extra['added_bytes'],
        baseline_reserve_ids=baseline['appended_ids'],baseline_reserve_bytes=baseline['added_bytes'],
        lexical=dict(scores=scores,matches=matches,ordered_ids=[c['id'] for c in ordered],
                     appended_ids=extra['appended_ids'],added_bytes=extra['added_bytes']))
    return result,trace
