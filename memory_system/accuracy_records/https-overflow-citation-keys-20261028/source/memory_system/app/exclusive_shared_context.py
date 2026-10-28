"""One complete source carrier for an unseen shared option term.

A term must occur in at least two literal options and exactly one original
observation in the prepared coarse corpus. This is a retrieval signal only;
roles, source spans and the original packet remain intact, without fact labels.
"""
from collections import Counter
import hashlib

from . import budget, config, fine_evidence, personal_evidence, source_first_residual
from .conversation_context import usable
from .fine_context_reserve import propose as reserve

POLICY = 'exclusive_shared_option_context_v1'
INCOMPATIBLE = ('PACKET_FINE_CONTEXT_FILL', 'PACKET_SOURCE_QUERY_CONTEXT',
    'PACKET_QUESTION_CONTEXT_CE', 'PACKET_SOURCE_FIRST_RESIDUAL',
    'PACKET_MARGINAL_QUERY_CONTEXT', 'PACKET_QUERY_CONTEXT_COMBINED',
    'PACKET_CORE_GAP_CONTEXT', 'PACKET_SPARSE_SOURCE_CONTEXT')


def enabled():
    return (config.PACKET_EXCLUSIVE_SHARED_CONTEXT and config.PACKET_FINE_CONTEXT_RESERVE
        and config.PACKET_FINE_CONTEXT_5K and config.PACKET_FINE_CONTEXT_LEXICAL
        and not any(getattr(config, key) for key in INCOMPATIBLE))


def identity(source):
    return tuple(source.get(k) for k in ('user_id', 'request_id', 'message_index', 'source_event_id', 'role'))


def nominate(pool, packed, *, coarse_ids, options, user_id, top_k, byte_budget, remaining_bytes):
    coarse = [pool[mid] for mid in coarse_ids if mid in pool]
    corpus = {}
    for candidate in coarse:
        budget.check()
        for source in candidate['_packet_item'].get('sources', []):
            if usable(source, user_id) and source['role'] in ('user', 'assistant'):
                corpus.setdefault(identity(source), set()).update(personal_evidence.terms(source['content']))
    frequency = Counter(term for terms in corpus.values() for term in terms)
    option_frequency = Counter(term for option in options for term in personal_evidence.terms(option))
    represented = set().union(*(personal_evidence.terms(s['content']) for p in packed
        for s in p.get('sources', []) if usable(s, user_id)))
    eligible_terms = {w for w, n in frequency.items() if n == 1 and len(w) >= 4
        and option_frequency[w] >= 2 and w not in represented}
    kept = {p['id'] for p in packed}
    candidates, nominations = [], []
    for candidate in coarse:
        budget.check()
        mid = candidate['id']
        if mid in kept:
            continue
        group, missing = fine_evidence.closure(candidate, pool)
        if missing or not group <= kept | {mid}:
            continue
        _, trial = reserve(packed, [candidate], user_id=user_id, top_k=top_k,
            byte_budget=byte_budget, reserve_bytes=remaining_bytes, limit=1)
        if not trial['proposed']:
            continue
        anchors = []
        for source in candidate['_packet_item'].get('sources', []):
            if not source.get('content'):
                continue
            hits = eligible_terms & personal_evidence.terms(source['content'])
            if hits:
                anchors.append(dict(source_identity=list(identity(source)),
                    content_span=source.get('content_span'),
                    source_sha256=hashlib.sha256(source['content'].encode()).hexdigest(), hits=sorted(hits)))
        words = sorted(set().union(*(set(a['hits']) for a in anchors)))
        if not words:
            continue
        candidates.append(candidate)
        nominations.append(dict(candidate_id=mid, terms=words, anchors=anchors,
            ce_score=candidate['_ce_score'], cost=trial['proposed'][0]['added_bytes']))
    by_id = {n['candidate_id']: n for n in nominations}
    candidates.sort(key=lambda c: (-len(by_id[c['id']]['terms']), -c['_ce_score'], c['id']))
    return candidates, dict(corpus_observations=len(corpus), eligible_terms=sorted(eligible_terms),
        option_frequency={w: option_frequency[w] for w in sorted(eligible_terms)},
        source_frequency={w: frequency[w] for w in sorted(eligible_terms)},
        nominations=nominations, ordered_ids=[c['id'] for c in candidates])


def attach(packed, packet_hash, manifest, candidates, nomination, *, user_id, top_k):
    result, digest, updated = source_first_residual.attach(packed, packet_hash, manifest,
        candidates, nomination, user_id=user_id, top_k=top_k)
    context = dict(updated['conversation_context'])
    context['exclusive_shared'] = context.pop('source_first')
    context['policy'] = POLICY
    return result, digest, dict(updated, conversation_context=context)
