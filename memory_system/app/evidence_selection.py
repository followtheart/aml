"""Score-ordered selection with witnessed reservations and visible-source coverage."""
import re
import math
from . import budget, personal_evidence as pe, search_coverage

# Retrieval-only topic families. They do not imply ownership, preference, or
# entailment. Alias matching is kept out of personal-evidence validation.
_TOPIC_FAMILIES = (
    ('vinyl', 'lps', 'lp', 'pressings', 'pressing', 'phonograph'),
    ('cryptocurrency', 'cryptocurrencies', 'crypto', 'blockchain', 'stablecoins', 'stablecoin'),
    ('cholesterol', 'lipid', 'lipids'),
)


def _topic_concepts(text):
    concepts = search_coverage._concepts(text)
    words = pe.terms(text)
    for index, family in enumerate(_TOPIC_FAMILIES):
        if words.intersection(family):
            concepts.add(f'topic_family:{index}')
    return concepts


def _text(value):
    return re.sub(r'\s+', ' ', value).strip()


def fragments(row):
    """Only source text actually visible to the ranker can represent evidence."""
    body = _text(row.get('_rank_text', row.get('content', '')))
    result = []
    for source in row.get('_packet_item', {}).get('sources', []):
        text = _text(source.get('content') or '')
        if not text or text not in body:
            continue
        sid = source.get('source_event_id')
        if not sid and source.get('request_id') is not None and source.get('message_index') is not None:
            sid = f"{source['request_id']}:{source['message_index']}"
        if sid:
            result.append(((sid, source.get('role')), text))
    return result


def represented_by(row, selected, cache=None):
    visible = (lambda c: cache[c['id']]) if cache is not None else fragments
    pieces = visible(row)
    if not pieces:
        return []
    qualifiers = ('polarity', 'epistemic_status', 'resolution_status', 'valid_from', 'valid_to',
                  'temporal', 'event_time', 'state', 'knowledge_status')
    compatible = [c for c in selected if all(c.get(k) == row.get(k) for k in qualifiers)]
    # Source equality must not erase an additional recorded graph obligation.
    bridges = set(row.get('_bridge_ids', [])) | set(row.get('_relation_path_ids', []))
    seen_bridges = {mid for c in compatible for mid in
                    [c['id'], *c.get('_bridge_ids', []), *c.get('_relation_path_ids', [])]}
    if not bridges <= seen_bridges:
        return []
    owners = set()
    for source, text in pieces:
        owner = next((c['id'] for c in compatible
                      if any(key == source and text in content for key, content in visible(c))), None)
        if owner is None:
            return []
        owners.add(owner)
    return sorted(owners)


def select(rows, count, requirements, key, reserve_ids=(), *, carry_ids=(), fused_ids=(), trace=None):
    """Keep score order; defer only evidence completely represented by selected units.

    Strong source witnesses and bounded option-topic champions reserve slots.
    Option-topic matching admits evidence for inspection; it never proves a fact.
    """
    ordered = sorted(rows, key=key, reverse=True)
    visible = {c['id']: fragments(c) for c in ordered}
    count = max(0, count)
    by_id = {c['id']: c for c in ordered}
    reservations = {}
    for requirement in requirements:
        champion = next((c for c in ordered if requirement['id'] in c.get('_supported_coverage_ids', [])), None)
        if champion is not None:
            entry = reservations.setdefault(champion['id'], dict(requirement_ids=[], reasons=[]))
            entry['requirement_ids'].append(requirement['id'])
            if 'supported_coverage' not in entry['reasons']:
                entry['reasons'].append('supported_coverage')
    # One champion per option, chosen on visible source passages rather than the
    # question's CE score. No gold labels, summaries or hidden source metadata.
    # A set-cover gain lets one passage satisfy multiple option facets.
    for mid, rids in option_champions(ordered, requirements, visible, max(1, count // 2)).items():
        entry = reservations.setdefault(mid, dict(requirement_ids=[], reasons=[]))
        entry['requirement_ids'] = list(dict.fromkeys(entry['requirement_ids'] + rids))
        entry['reasons'].append('option_topic_coverage')
    for reason, members in (('unscored_rescue', reserve_ids), ('carried_reservation', carry_ids),
                            ('fused_head', fused_ids)):
        for mid in members:
            if mid in by_id:
                entry = reservations.setdefault(mid, dict(requirement_ids=[], reasons=[]))
                entry['reasons'].append(reason)
    reserved = [c for c in ordered if c['id'] in reservations]
    # Soft fused-head reserves must not crowd out option or carried witnesses.
    reserved.sort(key=lambda c: reservations[c['id']]['reasons'] == ['fused_head'])
    selected = []
    deferred = []
    for c in reserved:
        owners = represented_by(c, selected, visible)
        if owners:
            deferred.append(c)
            for owner in owners:
                entry = reservations[owner]
                for field in ('requirement_ids', 'reasons'):
                    entry[field] = list(dict.fromkeys(entry[field] + reservations[c['id']][field]))
        elif len(selected) < count:
            selected.append(c)
    for c in ordered:
        budget.check()
        if c in selected or c in reserved:
            continue
        owners = represented_by(c, selected, visible)
        if owners:
            deferred.append(c)
        elif len(selected) < count:
            selected.append(c)
    # Distinct claims from the same source remain available when capacity permits.
    for c in deferred:
        if len(selected) >= count:
            break
        selected.append(c)
    chosen_ids = {c['id'] for c in selected}
    if trace is not None:
        trace.update(policy='score_order_visible_evidence', reservations=[
            dict(candidate_id=mid, **entry, selected=mid in chosen_ids)
            for mid, entry in reservations.items()], decisions=[])
        for rank, c in enumerate(ordered, 1):
            others = [x for x in selected if x['id'] != c['id']]
            owners = represented_by(c, others, visible)
            reason = ('+'.join(reservations[c['id']]['reasons']) if c['id'] in reservations
                      else 'covered_sources' if owners else 'rank')
            trace['decisions'].append(dict(id=c['id'], rank=rank, selected=c['id'] in chosen_ids,
                reason=reason if c['id'] in chosen_ids or owners else 'capacity',
                score=c.get('_ce_score'), represented_by=owners,
                visible_sources=sorted({s[0] for s, _ in visible[c['id']]})))
    return [c for c in ordered if c['id'] in chosen_ids]


def option_champions(ordered, requirements, visible, limit):
    """Bounded per-option topical witnesses, deliberately separate from support."""
    requirements = [r for r in requirements if r['id'].startswith('option:')]
    if not requirements or not ordered:
        return {}
    passages = {c['id']: [(role, _topic_concepts(text)) for (_, role), text in visible[c['id']]]
                for c in ordered}
    corpus = {mid: set().union(*(terms for _, terms in pieces)) for mid, pieces in passages.items()}
    def weight(term):
        return 1 + math.log((len(corpus) + 1) / (1 + sum(term in words for words in corpus.values())))
    selected, covered = {}, set()
    for requirement in requirements:
        rid = requirement['id']
        if rid in covered:
            continue
        wanted = _topic_concepts(requirement['text'])
        total = sum(weight(w) for w in wanted)
        if not total:
            continue
        scores = {}
        for c in ordered:
            best = 0
            for role, terms in passages[c['id']]:
                hits = wanted & terms
                ratio = sum(weight(w) for w in hits) / total
                family_hit = any(w.startswith('topic_family:') for w in hits)
                if family_hit or (len(hits) >= min(2, len(wanted)) and ratio >= .3):
                    # For comparable topical coverage, prefer a first-party
                    # question over a long assistant essay with many keywords.
                    score = (role == 'user', family_hit, ratio)
                    best = max(best, score) if best else score
            if best:
                scores[c['id']] = best
        if not scores:
            continue
        champion = max((c for c in ordered if c['id'] in scores), key=lambda c: scores[c['id']])
        mid = champion['id']
        if mid not in selected and len(selected) >= limit:
            continue
        selected.setdefault(mid, []).append(rid)
        covered.add(rid)
    return selected


def rescue(rows, count, query, specs=()):
    """Give failed candidates bounded opportunities across distinct query facets."""
    facets = [query] + [q['text'] for q in specs if q.get('text') and q.get('origin') in
                        ('option_premise', 'option_fallback', 'sub_queries', 'grounded_followup')]
    chosen, remaining = [], list(rows)
    visible = {c['id']: fragments(c) for c in rows}
    for facet in dict.fromkeys(facets):
        if len(chosen) >= count or not remaining:
            break
        needles = pe.terms(facet)
        def rank(c):
            pieces = visible[c['id']]
            texts = [text for _, text in pieces] or [c.get('_rank_text', c['content'])]
            score = max((pe._sentence_score(s.group(), needles)[0]
                         for text in texts for s in pe._sentences(text)), default=0)
            return (score, not bool(represented_by(c, chosen, visible)),
                    bool(c.get('_selection_evidence', {}).get('user_source')),
                    c.get('_fused', 0))
        candidate = max(remaining, key=rank)
        if rank(candidate)[0] > 0:
            chosen.append(candidate)
            remaining.remove(candidate)
    # Finish in score order while preferring evidence not already represented.
    tail = sorted(remaining, key=lambda c: c.get('_fused', 0), reverse=True)
    for duplicate in (False, True):
        for c in tail:
            if len(chosen) >= count:
                return chosen
            if c not in chosen and bool(represented_by(c, chosen, visible)) == duplicate:
                chosen.append(c)
    return chosen
