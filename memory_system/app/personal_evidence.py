"""Source-aware personal evidence and pre-verification source excerpts."""
import re

_PERSONAL = re.compile(r"\b(?:the user|user's|i|my|we|our)\b", re.I)
_ASSISTANT = re.compile(r'^\s*(?:the )?assistant\b', re.I)
_NAMED_PERSONAL = re.compile(
    r'^[A-Z][a-z]+(?: [A-Z][a-z]+| and [A-Z][a-z]+){0,4}'
    r"(?:['’]s [^\n.]{0,80})?(?:, [^\n.]{0,180},)?\s+"
    r'(?:promised|asked|requested|owns|enjoys|prefers|plans|feels|felt|wants|'
    r'went|visited|attended|took|lives|lived|works|worked|adjusts|does|is|has|had)\b')
_STOP = set('the a an and or to of in on for is are was were be been with from that this it you your user users how what why some can could would should have has do does about as at by when which me my they their'.split())
_SELF_STATEMENT = re.compile(
    r"\b(?:I|we)(?:['’](?:m|ve|d|re)|\s+(?:am|have|had|was|were|live|lived|own|keep|"
    r"enjoy|love|prefer|promised|plan|feel|felt|want|went|visited|attended|took|work|teach|"
    r"do|did|can(?:not|'t)|could(?:n't)?|need|arrived|found|returned|wrote))\b|"
    r'\b(?:my|our)\s+[^.!?\n]{0,100}\b(?:is|are|has|have|was|were)\b', re.I)


def _user_declaration(sources):
    for source in sources:
        if source.get('role') != 'user':
            continue
        for sentence in re.findall(r'[^.!?\n]+[.!?]?', source.get('content', '')):
            if sentence.rstrip().endswith('?'):
                continue
            if re.search(r"\b(?:I|we)\s+(?:am|are)\s+curious\b|\b(?:want|like) to (?:know|understand)\b", sentence, re.I):
                continue
            if _SELF_STATEMENT.search(sentence):
                return True
    return False


def terms(text):
    return {w.casefold() for w in re.findall(r'[\w\u3400-\u9fff]+', text)
            if len(w) > 2 and w.casefold() not in _STOP}


def personal(candidate, sources=None):
    """Keep user-backed facts/plans too; don't promote assistant explanations.

    Legacy unsourced rows use explicit user attribution or a personal type.
    This is a retrieval view, not proof that every inferred trait is true.
    """
    text = candidate.get('content', '').split('\n[source evidence', 1)[0]
    # Model-facing prefixes aren't part of the attribution sentence.
    text = re.sub(r'^(?:\[[^\]]*\]\s*)+', '', text)
    if _ASSISTANT.match(text):
        return False
    sources = candidate.get('sources', []) if sources is None else sources
    kind = candidate.get('memory_type', candidate.get('type', 'fact'))
    attributed = bool(_PERSONAL.search(text)
                      or re.search(r'用户|使用者|用戶|我(?:的|們|们|曾|有|喜欢|喜歡)', text))
    if sources and not any(s.get('role') == 'user' for s in sources):
        return False
    # Capitalization alone cannot distinguish Daniel from Water or Japan.
    # Named summaries require a user declaration, not merely a world-knowledge
    # question in the same dialog. Unsourced legacy rows retain the fallback.
    if _NAMED_PERSONAL.search(text):
        attributed |= not sources or _user_declaration(sources)
    return kind in ('profile', 'preference', 'rule', 'plan') or attributed


def excerpt(text, query, limit, quotes=()):
    """Select one contiguous original span; never paraphrase a source quote."""
    if len(text) <= limit:
        return text, 0, len(text)
    spans = [(text.find(q), text.find(q) + len(q)) for q in quotes if q and q in text]
    if spans:
        # All explicit support from this message must survive, even if long.
        start, end = min(s[0] for s in spans), max(s[1] for s in spans)
    else:
        needles = terms(query)
        sentences = list(re.finditer(r'[^\n.!?]+(?:[.!?]+|\n|$)', text))
        if not sentences:
            return text, 0, len(text)
        best = max(sentences, key=lambda s: len(terms(s.group()) & needles))
        start, end = best.start(), best.end()
    # Expand to sentence boundaries, preserving negations and qualifications.
    left = max(text.rfind('\n', 0, start), text.rfind('. ', 0, start))
    start = left + (2 if text[left:left + 2] == '. ' else 1) if left >= 0 else 0
    tails = [p for p in (text.find('\n', end), text.find('. ', end)) if p >= 0]
    if end < len(text) and text[end - 1:end] not in '.!?\n':
        end = min(tails) + 1 if tails else len(text)
    return text[start:end], start, end


def compact_sources(candidate, sources, query, limit, max_messages):
    """Keep source references; only selected verbatim spans enter the packet.

    Select relevant spans regardless of message role: an assistant turn may
    contain a user draft or resolve a short reply. Roles remain explicit so
    the answer model can distinguish quoted context from user assertions.
    """
    out = []
    by_source = {}
    for s in sources:
        key = (s.get('request_id'), s.get('message_index'))
        by_source[key] = [e.get('quote', '') for e in candidate.get('evidence') or []
                          if e.get('message_index') == key[1] and e.get('request_id') == key[0]
                          and e.get('quote') and e['quote'] in s.get('content', '')]
    # Explicit support is mandatory, including disambiguating assistant turns.
    required = {key for key, quotes in by_source.items() if quotes}
    needles = terms(query + ' ' + candidate['content'])
    extras = sorted((s for s in sources if s.get('content') and
                     (s.get('request_id'), s.get('message_index')) not in required),
                    key=lambda s: (-len(terms(s['content']) & needles), s.get('role') != 'user'))
    chosen = required | {(s.get('request_id'), s.get('message_index'))
                         for s in extras[:max(0, max_messages - len(required))]}
    for source in sources:
        s = dict(source)
        text = s.pop('content', '')
        key = (s.get('request_id'), s.get('message_index'))
        if key not in chosen:
            s['content_omitted'] = 'source_limit'
        elif text:
            quotes = by_source[key]
            shown, start, end = excerpt(text, query + ' ' + candidate['content'], limit, quotes)
            s['content'] = shown
            s['content_span'] = {'start': start, 'end': end, 'original_length': len(text)}
            if start or end < len(text):
                s['content_excerpted'] = True
        out.append(s)
    return out
