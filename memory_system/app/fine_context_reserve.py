"""Bounded original-source reserve from this request's eligible scored fine pool."""
import copy
import math
from . import answer_context, budget, schemas
from .conversation_context import source_key, usable

POLICY = 'fine_original_context_v1'


def full_user(source):
    if source.get('role')!='user':return False
    span=source.get('content_span')
    return not span or (span.get('start')==0 and span.get('end')==span.get('original_length')==len(source['content']))


def propose(packed,fine,*,user_id,top_k,byte_budget,limit=2,reserve_bytes=4096):
    original=copy.deepcopy(packed)
    result=list(packed)
    used=sum(len(p['content'].encode()) for p in result)+max(0,len(result)-1)
    if used>byte_budget or len(result)>top_k:raise ValueError('Baseline packet exceeds limits')
    known={source_key(s) for p in packed for s in p.get('sources',[]) if usable(s,user_id)}
    kept={p['id'] for p in packed}
    trace=dict(proposed=[],appended_ids=[],added_bytes=0,policy=POLICY,
               limit=limit,reserve_bytes=reserve_bytes)
    for candidate in fine:
        budget.check()
        mid=candidate['id'];item=candidate.get('_packet_item',{});score=candidate.get('_ce_score')
        if (mid in kept or candidate.get('view_status')=='stale' or item.get('is_constraint')
                or type(score) not in (float,int) or not math.isfinite(score)):
            continue
        sources=[s for s in item.get('sources',[]) if s.get('content')]
        if not sources or any(not usable(s,user_id) or s['role'] not in ('user','assistant') for s in sources):
            continue
        # Every included quote must already be fully visible in this request's
        # immutable prepared unit. No review-only or stored full-text expansion.
        if any(s['content'] not in item.get('content','') for s in sources):
            continue
        novel_users=[s for s in sources if full_user(s) and source_key(s) not in known]
        if not novel_users:
            continue
        addition=schemas.SearchItem.model_validate(dict(id=mid,
            content=answer_context.with_evidence('Conversation excerpt:',sources),created_at=item.get('created_at'),
            memory_type='episode',score=score,score_kind='fine_recall_context',personal_evidence=False,
            coverage_ids=[],is_constraint=False,equivalent_ids=[],sources=copy.deepcopy(sources),
            source_count=len(sources),packet_hash_version=4)).model_dump()
        cost=len(addition['content'].encode())+bool(result)
        reason=('context_limit' if len(trace['appended_ids'])>=limit else
                'top_k' if len(result)>=top_k else
                'reserve_bytes' if trace['added_bytes']+cost>reserve_bytes else
                'token_budget' if used+cost>byte_budget else 'admitted')
        trace['proposed'].append(dict(candidate_id=mid,ce_score=score,reason=reason,added_bytes=cost,
            novel_user_events=[s['source_event_id'] for s in novel_users]))
        if reason!='admitted':continue
        result.append(addition);kept.add(mid);used+=cost
        known.update(source_key(s) for s in sources)
        trace['appended_ids'].append(mid);trace['added_bytes']+=cost
    if packed!=original or result[:len(packed)]!=original:raise ValueError('Original packet changed')
    trace.update(original_count=len(packed),result_count=len(result),total_bytes=used)
    return result,trace
