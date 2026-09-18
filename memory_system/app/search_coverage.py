"""Soft topic coverage and conservative, source-local retention witnesses.

Witnesses justify a bounded retrieval reservation, not semantic entailment or a
new personal fact. Unsupported candidates remain eligible for ordinary ranking.
"""
import re
from . import personal_evidence


MAX_WITNESS_CHARS = 480
MAX_WITNESSES_PER_REQUIREMENT = 2
MAX_WITNESS_SOURCES = 32

# Small, general lexical equivalences; no query, user or dataset identifiers.
_ALIASES = {
    'pupil': 'student', 'learner': 'student', 'distressing': 'troubling',
    'distress': 'trouble', 'situation': 'circumstance', 'matter': 'circumstance', 'issue': 'circumstance',
    'disclose': 'share', 'confide': 'share',
}
_WEAK = set('experience present respond response responding sharing share stay '
            'grounded well advice tip recommendation information thing personally '
            'personal already current currently previously former past year ago '
            'childhood keeping eye'.split())
_SELF = re.compile(r'\b(?:i|we|my|our|me)\b|我(?:們|们|的)?', re.I)
_THIRD_PARTY = re.compile(r'\b(?:my|our)\s+(friend|colleague|mother|father|sister|brother|'
                          r'daughter|son|child|spouse|wife|husband)\b', re.I)
_ATTRIBUTION = re.compile(r'\b(?!I\b|We\b)([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,2})\s+'
                         r'(?:said|wrote|shared|stated)\s*[:,]')
_QUOTED_RELATION = re.compile(r'\b(?:my|our)\s+\w+\s+(?:said|wrote|shared|stated)\s*[:,]', re.I)
_FICTIONAL_FRAME = re.compile(r'^\s*(?:imagine|suppose|pretend|assume)\b|'
                              r'\b(?:fictional|imaginary|hypothetical)\s+(?:diary|story|scenario|example|person|character)\b|'
                              r'虛構|虚构|假設以下|假设以下', re.I)
_MIXED_CLAUSES = re.compile(r'\b(?:but|however|although|whereas)\b|;|'
                            r'\b(?:and|while)\s+(?:i|we|my|our|he|she|they)\b|'
                            r',\s*(?:not|never|no longer)\b', re.I)
_HYPOTHETICAL = re.compile(r'\b(?:if|suppose|supposing|imagine|hypothetically|would|could)\b|'
                          r'假如|假設|假设|如果', re.I)
_NEGATION = re.compile(r"\b(?:not|never|no longer|without)\b|n['’]t\b|不再|沒有|没有|並不|并不", re.I)
_PAST = re.compile(r'\b(?:used to|in the past|previously|formerly|childhood|history of|'
                  r'years? ago|as a child|when i was a (?:child|kid)|past)\b|過去|过去|以前|曾經|曾经', re.I)
_POSSESSION = re.compile(r'\b(?:have|has|own|owns|keep|keeps|stocked|maintain)\b|擁有|拥有', re.I)
_SELF_POSSESSION = re.compile(r"\b(?:i|we)(?:['’]ve|\s+(?:(?:already|do not|don't|don’t)\s+)?(?:have|had|own|keep|maintain|grow))\b|"
                              r'\b(?:my|our)\s+[^.!?\n]{0,70}\b(?:is|are|has|have)\b|'
                              r'我(?:們|们)?(?:有|擁有|拥有|種|种)', re.I)
_PREFERENCE = re.compile(r'\b(?:enjoys?|loves?|prefers?|passion)\b|喜歡|喜欢|熱愛|热爱', re.I)
_SELF_PREFERENCE = re.compile(r"\b(?:i|we)\s+(?:(?:used to|do not|don't|don’t)\s+)?(?:enjoy|love|prefer)\b|"
                              r'\b(?:my|our)\s+(?:passion|favorite|favourite)\b|'
                              r'我(?:們|们)?(?:喜歡|喜欢|熱愛|热爱)', re.I)


def _stem(word):
    if len(word) > 4 and word.endswith('ies'):
        word = word[:-3] + 'y'
    elif len(word) > 4 and word.endswith('s') and not word.endswith(('ss', 'us')):
        word = word[:-1]
    word = _ALIASES.get(word, word)
    if len(word) > 5 and word.endswith('ing'):
        word = word[:-3]
    elif len(word) > 4 and word.endswith('ed'):
        word = word[:-2]
    if len(word) > 4 and word.endswith('e'):
        word = word[:-1]
    return word


_WEAK_STEMS = {_stem(word) for word in _WEAK}


def _concepts(text):
    text = re.sub(r'\bkeep(?:ing)? an eye on\b', '', text, flags=re.I)
    return {_stem(word) for word in personal_evidence.terms(text)} - _WEAK_STEMS


def _local_passages(text):
    # A neighboring sentence may carry a pronoun's antecedent. Never combine
    # distant sentences or cross a paragraph boundary to manufacture a witness.
    previous = None
    for match in re.finditer(r'[^.!?。！？\n]+(?:[.!?。！？]+|(?=\n)|$)', text):
        raw = match.group()
        start = match.start() + len(raw) - len(raw.lstrip())
        end = match.end() - len(raw) + len(raw.rstrip())
        if 0 < end - start <= MAX_WITNESS_CHARS:
            yield start, end, text[start:end]
            if (previous is not None and end - previous[0] <= MAX_WITNESS_CHARS and
                    '\n' not in text[previous[1]:start]):
                yield previous[0], end, text[previous[0]:end]
        previous = start, end


def _scope_matches(text, source_text, requirement):
    wanted = requirement['text']
    if (_HYPOTHETICAL.search(text) or not _SELF.search(text) or _MIXED_CLAUSES.search(text) or
            _FICTIONAL_FRAME.search(source_text)):
        return False
    # Uncertainty about how to respond does not negate the preceding event.
    # 'Not sure whether/if I have X' remains uncertain and cannot be support.
    polarity_text = re.sub(r'\bnot\s+(?:entirely\s+)?sure\s+(?:how|what|which|where|when)\b',
                           'uncertain about', text, flags=re.I)
    if _NEGATION.search(polarity_text) is not None and _NEGATION.search(wanted) is None:
        return False
    if _NEGATION.search(wanted) is not None and _NEGATION.search(polarity_text) is None:
        return False
    if bool(_PAST.search(text)) != bool(_PAST.search(wanted)):
        return False
    # A first-person quotation can still belong to a named third party.
    if _ATTRIBUTION.search(source_text) or _QUOTED_RELATION.search(source_text):
        return False
    relation = _THIRD_PARTY.search(text)
    if relation and not re.search(r'\b' + re.escape(relation[1]) + r'\b', wanted, re.I):
        return False
    neutral = re.sub(r'\bkeep(?:ing)? an eye on\b', '', wanted, flags=re.I)
    # Personal attributes need one local assertion. A neighboring event sentence
    # may resolve 'they' for a disclosure, but must not move another assertion's
    # negation or ownership onto this target.
    if (_PREFERENCE.search(wanted) or _POSSESSION.search(neutral)) and len(
            list(re.finditer(r'[^.!?。！？\n]+(?:[.!?。！？]+|$)', text))) > 1:
        return False
    if relation:
        predicate = text[relation.end():]
        prefix = r"^\s+(?:(?:used to|does not|doesn't|doesn’t|often|usually)\s+)?"
        if _PREFERENCE.search(wanted) and not re.match(prefix + r'(?:enjoys?|loves?|prefers?)\b', predicate, re.I):
            return False
        if _POSSESSION.search(neutral) and not re.match(prefix + r'(?:has|had|owns?|keeps?|maintains?|grows?)\b', predicate, re.I):
            return False
    if not relation and _POSSESSION.search(neutral) and not _SELF_POSSESSION.search(text):
        return False
    if not relation and _PREFERENCE.search(wanted) and not _SELF_PREFERENCE.search(text):
        return False
    return True


def _witnesses(requirement, sources):
    wanted = _concepts(requirement['text'])
    if not wanted:
        return []
    found, seen = [], set()
    for source in sources[:MAX_WITNESS_SOURCES]:
        text = source.get('content')
        if (source.get('role') != 'user' or not isinstance(text, str) or not text or
                not isinstance(source.get('request_id'), str) or not source['request_id'] or
                type(source.get('message_index')) is not int or source['message_index'] < 0):
            continue
        span = source.get('content_span') or {}
        offset = span.get('start', 0)
        if type(offset) is not int or offset < 0:
            continue
        if span and (type(span.get('end')) is not int or span['end'] - offset != len(text) or
                     type(span.get('original_length')) is not int or span['original_length'] < span['end']):
            continue
        for start, end, passage in _local_passages(text):
            hits = wanted & _concepts(passage)
            if (len(hits) < min(2, len(wanted)) or len(hits) / len(wanted) < .75 or
                    not _scope_matches(passage, text, requirement)):
                continue
            key = (source['request_id'], source['message_index'], offset + start, offset + end)
            if key in seen:
                continue
            seen.add(key)
            found.append(dict(request_id=source['request_id'], message_index=source['message_index'],
                source_event_id=source.get('source_event_id'), role=source['role'], content=passage,
                span=dict(start=offset + start, end=offset + end), matched_terms=sorted(hits),
                basis='local_source_lexical_support; not entailment'))
            if len(found) >= MAX_WITNESSES_PER_REQUIREMENT:
                return found
    return found


def source_keys(sources):
    return {s.get('source_event_id') or f"{s['request_id']}:{s['message_index']}"
            for s in sources if s.get('content')}


def matches(text, query):
    wanted = personal_evidence.terms(query)
    hits = wanted & personal_evidence.terms(text)
    return bool(wanted and len(hits) >= min(2, len(wanted)) and len(hits) / len(wanted) >= .5)


def annotate(item, requirements, sources=()):
    sources = list(sources)
    text = item.get('_rank_text', item.get('content', ''))
    text += '\n' + '\n'.join(s.get('content', '') for s in sources)
    covered = set(item.get('_coverage_ids', []))
    covered.update(r['id'] for r in requirements if matches(text, r['text']))
    item['_coverage_ids'] = sorted(covered)
    witnesses = {r['id']: found for r in requirements
                 if (found := _witnesses(r, sources))}
    item['_coverage_witnesses'] = witnesses
    item['_supported_coverage_ids'] = sorted(witnesses)
    return item


def report(requirements, candidates, packed=()):
    rows = []
    for requirement in requirements:
        rid = requirement['id']
        found = [c['id'] for c in candidates if rid in c.get('_coverage_ids', [])]
        included = [c['id'] for c in packed if rid in c.get('coverage_ids', [])]
        supported = [c['id'] for c in candidates if rid in c.get('_supported_coverage_ids', [])]
        supported_included = [c['id'] for c in packed
                              if rid in c.get('supported_coverage_ids', c.get('_supported_coverage_ids', []))]
        rows.append(dict(requirement, candidate_ids=found, included_ids=included,
                         supported_candidate_ids=supported, supported_included_ids=supported_included,
                         status='packed' if included else 'candidate_only' if found else 'missing'))
    return dict(basis='query_match_or_grounded_followup; not semantic sufficiency', requirements=rows,
                missing_ids=[r['id'] for r in rows if r['status'] != 'packed'])
