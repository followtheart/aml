"""One bounded context seat for a near-cutoff source-query first hit.

Only already scored coarse outsiders may enter; the existing fine selector's
CE rank window also applies. Retrieval ranking never establishes personal facts.
"""
from . import budget, retrieval_queries
from .fine_context_reserve import propose as reserve
from .fine_context_lexical import propose as lexical

POLICY = 'source_query_context_v1'


def nominate(pool, *, routes, query_specs, query, options, coarse_ids, fine_ids,
             selected_ids, fine_selection):
    trace = dict(direct_queries={}, nominations=[], rejected=[], max_ce_rank=None)
    window = fine_selection.get('soft_reservation_policy', {}).get('max_ce_rank')
    initial = [r for r in routes if r.get('channel') == 'source_text' and r.get('round') == 1]
    if type(window) is not int or window < 1 or len(initial) != 1:
        trace['status'] = 'missing_rank_window_or_source_route'
        return [], trace
    trace['max_ce_rank'] = window
    direct = trace['direct_queries']
    for index, spec in enumerate(s for s in query_specs if s.get('origin') != 'grounded_followup'):
        origin = spec.get('origin')
        if origin == 'question' and spec.get('text') == query:
            direct[str(index)] = spec
        elif origin == 'option_topic':
            oi = spec.get('option_index')
            if (type(oi) is int and 0 <= oi < len(options)
                    and spec.get('text') == retrieval_queries.topic_anchor(options[oi])):
                direct[str(index)] = spec
    allowed = set(coarse_ids) - set(fine_ids) - set(selected_ids)
    ranks = {d['id']: d.get('rank') for d in fine_selection.get('decisions', [])}
    seen = set()
    for position, candidate in enumerate(initial[0].get('candidates', [])):
        budget.check()
        mid = candidate['id']
        if mid not in allowed or mid not in pool or mid in seen:
            continue
        seen.add(mid)
        hits = [key for key, rank in candidate.get('_query_ranks', {}).items()
                if type(rank) is int and rank == 1 and key in direct]
        if not hits:
            continue
        rank = ranks.get(mid)
        entry = dict(candidate_id=mid, query_indices=sorted(hits, key=int),
                     route_rank=position+1, ce_rank=rank)
        if type(rank) is not int or not 1 <= rank <= window:
            trace['rejected'].append(dict(entry, reason='outside_recorded_ce_window'))
        else:
            trace['nominations'].append(entry)
    trace['nominations'].sort(key=lambda p: (-len(p['query_indices']), p['route_rank']))
    trace['status'] = 'evaluated'
    return [pool[p['candidate_id']] for p in trace['nominations']], trace


def propose(packed, fine, *, champions, user_id, top_k, byte_budget, query, options, reserve_bytes=5120):
    first, source = reserve(packed, champions, user_id=user_id, top_k=top_k,
                            byte_budget=byte_budget, reserve_bytes=reserve_bytes, limit=1)
    for item in first[len(packed):]:
        item['score_kind'] = 'source_query_context'
    result, rest = lexical(first, fine, user_id=user_id, top_k=top_k, byte_budget=byte_budget,
                           query=query, options=options, reserve_bytes=reserve_bytes-source['added_bytes'])
    if result[:len(packed)] != packed:
        raise ValueError('Original accepted packet changed')
    return result, dict(policy=POLICY, limit=4, reserve_bytes=reserve_bytes,
        original_count=len(packed), result_count=len(result), total_bytes=rest['total_bytes'],
        proposed=source['proposed']+rest['proposed'],
        appended_ids=source['appended_ids']+rest['appended_ids'],
        added_bytes=source['added_bytes']+rest['added_bytes'], source_seat=source, remainder=rest)
