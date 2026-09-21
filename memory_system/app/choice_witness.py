"""Deterministic per-option witness pre-fill for the support stage.

The support model no longer has to *find* topical evidence inside the packet:
for each option premise, the sentences of visible sources that share its topic
vocabulary are listed next to the option. These are keyword hits, never proof;
every citation still passes the same verbatim and provenance checks.
"""
import math
import re

from . import evidence_selection, personal_evidence as pe, retrieval_queries

MAX_PER_OPTION = 3
MAX_QUOTE_CHARS = 320


def _sentences(text):
    for match in pe._sentences(text):
        quote = match.group().strip()
        if 0 < len(quote) <= MAX_QUOTE_CHARS and quote in text:
            yield quote


def build(options, sources):
    """Return {letter: [witness, ...]} using only the answer source catalog."""
    cards = [c for c in sources.values() if c.get('text')]
    passages = []
    for card in cards:
        for quote in _sentences(card['text']):
            passages.append(dict(source_id=card['id'], role=card.get('declared') or card['role'],
                                 quote=quote, concepts=evidence_selection._topic_concepts(quote)))
    if not passages:
        return {}
    frequency = {}
    for p in passages:
        for term in p['concepts']:
            frequency[term] = frequency.get(term, 0) + 1
    def weight(term):
        return 1 + math.log((len(passages) + 1) / (1 + frequency.get(term, 0)))
    witnesses = {}
    for index, option in enumerate(options):
        match = re.match(r'\s*\(?([A-Z])[.)]\s*', option)
        letter = match[1] if match else chr(65 + index)
        premise = retrieval_queries.compact_option(option)
        wanted = {w for w in evidence_selection._topic_concepts(premise) if len(w) >= 3}
        total = sum(weight(w) for w in wanted)
        if not total:
            continue
        scored = []
        for p in passages:
            hits = wanted & p['concepts']
            if not hits:
                continue
            ratio = sum(weight(w) for w in hits) / total
            family_hit = any(w.startswith('topic_family:') for w in hits)
            if not (family_hit or (len(hits) >= min(2, len(wanted)) and ratio >= .3)):
                continue
            first_party = p['role'] in ('user', 'persona')
            scored.append(((first_party, family_hit, ratio, len(hits)), p, sorted(hits)))
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen, used_sources = [], set()
        # One quote per source first, so a long assistant essay cannot fill
        # every seat with near-duplicate sentences.
        for key, p, hits in scored:
            if p['source_id'] in used_sources:
                continue
            chosen.append(dict(source_id=p['source_id'], role=p['role'], quote=p['quote'],
                               matched=[h for h in hits if not h.startswith('topic_family:')] or hits,
                               first_party=key[0]))
            used_sources.add(p['source_id'])
            if len(chosen) >= MAX_PER_OPTION:
                break
        if chosen:
            witnesses[letter] = chosen
    return witnesses


def render(witnesses):
    if not witnesses:
        return '(none found)'
    lines = []
    for letter, items in witnesses.items():
        for w in items:
            lines.append(f'{letter}: [{w["source_id"]} {w["role"]}] "{w["quote"]}"')
    return '\n'.join(lines)
