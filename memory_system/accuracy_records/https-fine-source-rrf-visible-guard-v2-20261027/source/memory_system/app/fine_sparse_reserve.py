"""Nominate unseen rare user-source terms for one extra inspection seat.

Literal retrieval matches never establish ownership, preference or support.
Only visible source fragments participate; repeated projections do not change
term frequencies. The caller enforces dependency, novelty and capacity limits.
"""
from collections import Counter
import math

from . import evidence_selection


def nominate(rows, core, requirements, visible):
    # The fine selector intentionally gives a profile quote its own coverage
    # obligation. For rarity, resolve that obligation back to original source
    # identity so extra profile projections cannot inflate document frequency.
    by_id = {c['id']: c for c in [*rows, *core]}
    canonical = {}
    for mid, pieces in visible.items():
        result = []
        for (key, role), text in pieces:
            if not key.startswith('profile-quote:'):
                result.append(((key, role), text))
                continue
            for source in by_id[mid].get('_packet_item', {}).get('sources', []):
                if source.get('role') != role or text not in evidence_selection._text(source.get('content', '')):
                    continue
                sid = source.get('source_event_id')
                if not sid and source.get('request_id') is not None and source.get('message_index') is not None:
                    sid = f"{source['request_id']}:{source['message_index']}"
                if sid:
                    result.append(((sid, role), text))
        canonical[mid] = result
    visible = canonical
    corpus = {}
    for pieces in visible.values():
        for source, text in pieces:
            corpus.setdefault(source, set()).update(evidence_selection._topic_concepts(text))
    frequency = Counter(term for terms in corpus.values() for term in terms)
    wanted = {r['id']: evidence_selection._topic_concepts(r['text']) for r in requirements}
    requirement_frequency = Counter(term for terms in wanted.values() for term in terms)
    represented = set().union(*(evidence_selection._topic_concepts(text)
        for c in core for (_, role), text in visible[c['id']] if role == 'user'))
    core_ids = {c['id'] for c in core}
    rare_limit = max(1, len(corpus) // 10)
    nominations = {}
    for row in rows:
        score = row.get('_ce_score')
        if row['id'] in core_ids or type(score) not in (int, float) or not math.isfinite(score):
            continue
        terms = set().union(*(evidence_selection._topic_concepts(text)
            for (_, role), text in visible[row['id']] if role == 'user'))
        hits = {term for term in terms - represented if len(term) >= 4
                and not term.startswith('topic_family:') and requirement_frequency[term] == 1
                and frequency[term] <= rare_limit}
        if hits:
            nominations[row['id']] = dict(terms=sorted(hits),
                requirement_ids=sorted(rid for rid, words in wanted.items() if hits & words))
    return nominations
