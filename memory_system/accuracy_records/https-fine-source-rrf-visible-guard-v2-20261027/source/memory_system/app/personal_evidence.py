"""Source-aware personal evidence and verbatim source excerpts."""
import math
import re

_PERSONAL = re.compile(r"\b(?:the user|user's|i|my|we|our)\b", re.I)
_ASSISTANT = re.compile(r'^\s*(?:the )?assistant\b', re.I)
_NAMED_PERSONAL = re.compile(
    r'^[A-Z][a-z]+(?: [A-Z][a-z]+| and [A-Z][a-z]+){0,4}'
    r"(?:['’]s [^\n.]{0,80})?(?:, [^\n.]{0,180},)?\s+"
    r'(?:promised|asked|requested|owns|enjoys|prefers|plans|feels|felt|wants|'
    r'went|visited|attended|took|lives|lived|works|worked|adjusts|does|is|has|had)\b')
_STOP = set('the a an and or to of in on for is are was were be been with from that this it you your user users how what why some can could would should have has do does about as at by when which me my they their'.split())
_STOP.update('since given might consider try suggest help please like enjoy make more also really bit want things way ways'.split())
_STOP.update('where who whom whose did doing been being will shall may must could would should does not no yes we us our ours yours his her hers its them these those there here then than into out up down all any each few many much other same such very just only own both either neither if so because while'.split())
_STOP.update('什麼 什么 如何 哪些 哪裡 哪里 為何 为何 怎麼 怎么 我的 我們 我们 用戶 用户 使用 者的 的是 是否 有什 請問 请问'.split())
_EDIT_REQUEST = re.compile(r'\b(refine|polish|rewrite|rephrase|improve|proofread|revision|wording)\b', re.I)


def editorial_request(text):
    """A short editing request is context, not the body being edited."""
    return len(text) < 220 and '\n' not in text.strip() and bool(_EDIT_REQUEST.search(text))


def source_score(source, query):
    text = source.get('content') or ''
    needles = terms(query)
    best = max((_sentence_score(s.group(), needles) for s in _sentences(text)), default=(0, 0))
    # Local evidence density beats total overlap accumulated by a long persona
    # or general explanation. Role alone must not discard quoted third parties.
    return (not editorial_request(text), *best, source.get('role') == 'user')


def _sentences(text):
    return list(re.finditer(r'[^\n.!?。！？]+(?:[.!?。！？]+|\n|$)', text))


def _sentence_score(text, needles):
    words = terms(text)
    overlap = len(words & needles)
    return overlap / math.sqrt(max(8, len(words))), overlap
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
    # FTS5's default tokenizer cannot segment Chinese. Keep adjacent bigrams
    # within a run, never invent matches across punctuation or Latin words.
    latin = re.findall(r'[^\W_]+', re.sub(r'[\u3400-\u9fff]+', ' ', text.casefold()))
    cjk = [run[i:i + 2] for run in re.findall(r'[\u3400-\u9fff]+', text)
           for i in range(len(run) - 1)]
    return {w for w in latin + cjk if len(w) >= 2 and w not in _STOP}


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
        sentences = _sentences(text)
        if not sentences:
            return text, 0, len(text)
        best = max(sentences, key=lambda s: (not editorial_request(s.group()),
                                            *_sentence_score(s.group(), needles)))
        start, end = best.start(), best.end()
    # Use the available window for context, including names, qualifications and
    # nearby concrete details. A single matched sentence often hides the subject.
    # Expand enough to preserve local attribution, not to fill every per-source
    # allowance: filling 800 characters for each hit crowds out other memories.
    context_width = min(limit, 320)
    spare = max(0, context_width - (end - start))
    width = max(context_width, end - start)
    start = max(0, min(start - spare // 2, len(text) - width))
    end = max(end, min(len(text), start + width))
    # Expand to sentence boundaries, preserving negations and qualifications.
    left = max(text.rfind(mark, 0, start) for mark in ('\n', '. ', '。', '！', '？'))
    start = left + (2 if text[left:left + 2] == '. ' else 1) if left >= 0 else 0
    tails = [p for p in (text.find(mark, end) for mark in ('\n', '. ', '。', '！', '？')) if p >= 0]
    if end < len(text) and text[end - 1:end] not in '.!?。！？\n':
        end = min(tails) + 1 if tails else len(text)
    # A later sentence in "Marcus wrote: ... I ..." still belongs to Marcus.
    # Preserve the opening attribution of the same paragraph, verbatim.
    paragraph = text.rfind('\n\n', 0, start) + 2 if '\n\n' in text[:start] else 0
    if re.match(r'\s*["“]?(?:[A-Z][\w’\'-]*(?:\s+[A-Z][\w’\'-]*){0,3}\s+'
                r'(?:wrote|said|writes|says|shared|replied):|'
                r'[\u3400-\u9fff]{2,6}(?:說|说|寫道|写道|表示|提到)[：:])', text[paragraph:]):
        start = paragraph
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
    extras = sorted((s for s in sources if s.get('content') and
                     (s.get('request_id'), s.get('message_index')) not in required),
                    key=lambda s: source_score(s, query), reverse=True)
    def identity(s):
        return (s.get('content'), s.get('role'), s.get('speaker_id'), s.get('timestamp'),
                s.get('source_kind'), str(s.get('trust_scope')))
    seen = {identity(s) for s in sources if (s.get('request_id'), s.get('message_index')) in required}
    chosen = set(required)
    duplicate_sources = set()
    for s in extras:
        key = (s.get('request_id'), s.get('message_index'))
        if identity(s) in seen:
            duplicate_sources.add(key)
            continue
        if len(chosen) < max_messages:
            chosen.add(key)
            seen.add(identity(s))
    for source in sources:
        s = dict(source)
        text = s.pop('content', '')
        key = (s.get('request_id'), s.get('message_index'))
        if key not in chosen:
            s['content_omitted'] = 'duplicate_source' if key in duplicate_sources else 'source_limit'
        elif text:
            quotes = by_source[key]
            shown, start, end = excerpt(text, query, limit, quotes)
            s['content'] = shown
            s['content_span'] = {'start': start, 'end': end, 'original_length': len(text)}
            if start or end < len(text):
                s['content_excerpted'] = True
        out.append(s)
    return out
