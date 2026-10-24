"""Rank original evidence before the bounded verification pool, with source diversity."""
import asyncio
import json
import math
import time
import re
from .recall import recall
from .temporal import source_time
from .rerank_documents import documents as rerank_documents


def source_keys(candidate):
    event = candidate["event"]
    return {(event["session_id"], s["message_id"]) for s in event["sources"]}


def diverse_order(candidates, semantic=None):
    if not candidates:
        return []
    semantic_order = sorted(range(len(candidates)), key=lambda i: (-semantic[i], i)) if semantic is not None else []
    semantic_rank = {i: rank for rank, i in enumerate(semantic_order)}
    pending = []
    for i, candidate in enumerate(candidates):
        base = 1/(1+i/8)
        score = .95/(1+semantic_rank[i]/8)+.05*base if semantic is not None else base
        pending.append({**candidate, "coarse_rank": i+1, "score": score,
                        "semantic_score": semantic[i] if semantic is not None else None})
    result, covered = [], set()
    while pending:
        def gain(candidate):
            keys = source_keys(candidate)
            novelty = len(keys-covered)/max(1, len(keys))
            return candidate["score"]*(.55+.45*novelty)
        best = max(pending, key=lambda c: (gain(c), -c["coarse_rank"]))
        best["selection_gain"] = gain(best)
        result.append(best)
        covered.update(source_keys(best))
        pending.remove(best)
    return result


async def rank_candidates(provider, query, candidates, settings, deadline):
    metrics = {"rerank_calls": 0, "ranking_mode": "source_diversity"}
    errors = []
    if not candidates or not settings.rerank_model:
        return diverse_order(candidates), metrics, errors
    # A global query date is not the event date. Rank against the actual need.
    text = query.get('query','') if isinstance(query,dict) else str(query)
    if isinstance(query, dict) and query.get('options'):
        text += '\nCandidate answers to check against memories (not established facts):\n' + '\n'.join(query['options'])
    documents, owners = [], []
    try:
        limit = min(settings.rerank_document_bytes, settings.rerank_request_bytes-len(text.encode()))
        for index, candidate in enumerate(candidates):
            fragments = rerank_documents(candidate['event'], limit)
            documents.extend(fragments)
            owners.extend([index] * len(fragments))
    except ValueError as exc:
        return diverse_order(candidates), metrics, [{'stage': 'rerank', 'error_type': type(exc).__name__}]
    metrics['rerank_documents'] = len(documents)
    metrics['rerank_fragmented_candidates'] = sum(owners.count(i) > 1 for i in range(len(candidates)))
    batches, current, size = [], [], 0
    for document in documents:
        needed = len(document.encode())+len(text.encode())
        if current and (len(current) >= settings.rerank_batch_size or size+needed > settings.rerank_request_bytes):
            batches.append(current)
            current, size = [], 0
        current.append(document)
        size += needed
    if current:
        batches.append(current)
    scores = []
    try:
        for batch in batches:
            metrics["rerank_calls"] += 1
            values = await asyncio.wait_for(provider.rerank(text, batch), max(0, deadline-time.monotonic()))
            if len(values) != len(batch) or any(type(s) not in (float,int) or not math.isfinite(s) for s in values):
                raise ValueError("rerank alignment failure")
            scores.extend(values)
        # A relevant fragment promotes its original event; it does not become
        # evidence. Keep score ownership even across API batch boundaries.
        aggregate = [float('-inf')] * len(candidates)
        for index, value in zip(owners, scores):
            aggregate[index] = max(aggregate[index], value)
        scores = aggregate
        metrics["ranking_mode"] = "original_cross_encoder_source_diversity"
    except Exception as exc:
        scores = None
        errors.append({"stage": "rerank", "error_type": type(exc).__name__})
    return diverse_order(candidates, scores), metrics, errors


def query_anchors(query):
    """Only explicitly named query entities/quoted phrases/calendar years.

    This intentionally does not invent subquestions or use generated views.
    No useful anchor means no replacement, rather than a forced novelty slot.
    """
    skip = {'What','Which','When','Where','Who','How','Did','Does','Do','Was','Were','Is','Are',
            'Has','Have','Had','Can','Could','Would','Will','Why','The','A','An','I','My','In','On','At','Between'}
    quoted = re.findall(r'["“]([^"”]{2,100})["”]',query)
    names = [x for x in re.findall(r'\b[A-Z][a-zA-Z0-9-]+\b',query) if x not in skip]
    years = re.findall(r'\b(?:19|20)\d{2}\b',query)
    return sorted(set(x.casefold() for x in quoted+names+years))


def cover_query_anchors(candidates, query, limit, slots=2):
    anchors=query_anchors(query)
    ordered=[{**c,'pre_reservation_rank':i+1,'bridge_reserved':False,'covered_query_anchors':[
        a for a in anchors if any(re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)',s['content'],re.I)
                                 for s in c['event']['sources'])]} for i,c in enumerate(candidates)]
    if not anchors or not slots or len(ordered)<=limit or limit<2:return ordered,[]
    pool=ordered[:limit];promoted=[]
    # Replace only when explicit query coverage strictly increases without loss.
    # Gate on the same original reranker's top 2*limit, not an absolute probability.
    semantic=sorted(ordered,key=lambda c:-(c.get('semantic_score') if c.get('semantic_score') is not None else -1))
    allowed={id(c) for c in semantic[:2*limit] if c.get('semantic_score') is not None}
    for _ in range(min(slots,limit-1)):
        covered=set(a for c in pool for a in c['covered_query_anchors'])
        found=False
        for c in ordered[limit:]:
            if c in pool or id(c) not in allowed or not set(c['covered_query_anchors'])-covered:continue
            for index in range(limit-1,max(0,limit-slots)-1,-1):
                displaced=pool[index]
                remaining=set(a for j,x in enumerate(pool) if j!=index for a in x['covered_query_anchors'])
                if set(displaced['covered_query_anchors']) <= remaining|set(c['covered_query_anchors']):
                    c['bridge_reserved']=True;pool[index]=c;promoted.append(c['event']['event_id']);found=True;break
            if found:break
        if not found:break
    return pool+[c for c in ordered if c not in pool],[c['event']['event_id'] for c in pool if c['bridge_reserved']]


async def single_side_sources(events, query_views, vectors, settings, deadline):
    """Recompute both single-side candidate sets on the same indexed events.

    No new model calls, generated views or gold information. Compare source keys,
    not event IDs: an adjacent event may already expose the purported new source.
    """
    memory_only, query_only = await asyncio.wait_for(asyncio.gather(
        asyncio.to_thread(recall, events, query_views[:1], vectors[:1], settings, deadline),
        asyncio.to_thread(recall, events, query_views, vectors,
                          settings.model_copy(update={"memory_expansion": False}), deadline)),
        max(0, deadline-time.monotonic()))
    sources = set().union(*(source_keys(c) for cs,_ in (memory_only,query_only) for c in cs))
    return sources, sum(m["view_comparisons"] for _,m in (memory_only,query_only))


def reserve_exclusive(candidates, single_sources, limit, slots=1):
    ordered = [{**c, "pre_reservation_rank": i+1, "bridge_reserved": False,
                "exclusive_source_count": len(source_keys(c)-single_sources)} for i,c in enumerate(candidates)]
    if not slots or len(ordered)<=limit or limit<2:
        return ordered, []
    pool=ordered[:limit]
    # Never evict another exclusive source. Preserve all of the first limit-1.
    earlier=set().union(*(source_keys(c) for c in pool[:-1]))
    if source_keys(pool[-1])-single_sources-earlier:
        return ordered, []
    covered=set().union(*(source_keys(c) for c in pool))
    chosen=next((c for c in ordered[limit:] if source_keys(c)-single_sources-covered),None)
    if chosen is None:return ordered, []
    chosen["bridge_reserved"]=True
    displaced=pool[-1];pool[-1]=chosen
    tail=[c for c in ordered[limit:] if c is not chosen]+[displaced]
    tail.sort(key=lambda c:c["pre_reservation_rank"])
    return pool+tail,[chosen["event"]["event_id"]]
