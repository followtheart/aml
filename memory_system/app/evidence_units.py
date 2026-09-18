"""Prepare complete, traceable evidence units before either ranking or packing."""
import re
from . import answer_context, budget, config, personal_evidence as pe


def _span(source, start, end):
    result = dict(source)
    result['content'] = source['content'][start:end]
    result['content_span'] = dict(start=start, end=end, original_length=len(source['content']))
    if start or end != len(source['content']):
        result['content_excerpted'] = True
    return result


def passages(source, queries, quotes, cap):
    """Select whole sentences, retaining explicit support and attribution spans."""
    text = source.get('content', '')
    if len(text.encode('utf-8')) <= cap:
        return [dict(source)] if text else []
    sentences = pe._sentences(text)
    if not sentences:
        return []
    required, chosen = set(), set()
    for quote in quotes:
        start = text.find(quote)
        if start >= 0:
            required.update(i for i, s in enumerate(sentences)
                            if s.start() < start + len(quote) and s.end() > start)
    # An opening named quotation remains visible even when relevant text is late.
    if re.match(r'\s*["“]?(?:[A-Z][\w -]{0,80}(?:wrote|said|shared):|'
                r'[\u3400-\u9fff]{2,6}(?:說|说|寫道|写道|表示)[：:])', text):
        required.add(0)
    chosen.update(required)
    used = sum(len(sentences[i].group().encode('utf-8')) for i in chosen)
    if used > cap:
        return []  # Explicit support is atomic; report the excluded unit upstream.
    for query in queries:
        needles = pe.terms(query)
        ordered = sorted(range(len(sentences)),
                         key=lambda i: pe._sentence_score(sentences[i].group(), needles), reverse=True)
        for i in ordered:
            if not pe._sentence_score(sentences[i].group(), needles)[1]:
                continue
            if i in chosen:
                break
            # Keep local qualifications, negations, and antecedents when they fit.
            neighbors = list(range(max(0, i - 1), min(len(sentences), i + 2)))
            extra = [j for j in neighbors if j not in chosen]
            cost = sum(len(sentences[j].group().encode('utf-8')) for j in extra)
            if used + cost <= cap:
                chosen.update(extra)
                used += cost
                break
    if not chosen:
        return []
    intervals = []
    for i in sorted(chosen):
        s = sentences[i]
        if intervals and not text[intervals[-1][1]:s.start()].strip():
            intervals[-1] = (intervals[-1][0], s.end())
        else:
            intervals.append((s.start(), s.end()))
    return [_span(source, start, end) for start, end in intervals]


def build(candidate, sources, evidence, queries, body, byte_limit):
    """One fixed payload shared by the ranker and answer packet.

    Mandatory quote spans must fit together. Optional long documents are reduced
    to complete verbatim passages, with all remaining source references retained.
    """
    budget.check()
    available = byte_limit - len(body.encode('utf-8')) - 160
    if available <= 0:
        return None
    by_key = {(s['request_id'], s['message_index']): s for s in sources}
    required = {}
    for item in evidence:
        key = (item.get('request_id'), item.get('message_index'))
        if key in by_key and item.get('quote') and item['quote'] in by_key[key].get('content', ''):
            required.setdefault(key, []).append(item['quote'])
    query = ' '.join(queries)
    ordered = sorted(by_key.items(), key=lambda p: (p[0] in required, pe.source_score(p[1], query)), reverse=True)
    visible, hidden, optional = [], [], 0
    for key, source in ordered:
        budget.check()
        mandatory = key in required
        if not mandatory and optional >= max(config.SEARCH_SOURCE_MESSAGES_PER_ITEM, len(queries)):
            hidden.append(dict(source, content=''))
            continue
        # Leave capacity for the other required sources and query facets.
        remaining_required = sum(k in required for k, _ in ordered if k != key and
                                 not any((s['request_id'], s['message_index']) == k for s in visible))
        share = max(128, available // max(1, remaining_required + 1))
        rows = passages(source, queries, required.get(key, []), share)
        rendered = answer_context.with_evidence('', rows)
        cost = len(rendered.encode('utf-8'))
        if not rows or cost > available:
            if mandatory:
                return None
            hidden.append(dict(source, content=''))
            continue
        visible.extend(rows)
        available -= cost
        optional += not mandatory
    for item in hidden:
        item.pop('content', None)
        item['content_omitted'] = 'unit_budget'
    # Preserve chronological order for noncontiguous spans of the same message.
    rank = {key: i for i, key in enumerate(by_key)}
    visible.sort(key=lambda s: (rank[(s['request_id'], s['message_index'])],
                                (s.get('content_span') or {}).get('start', 0)))
    chosen = visible + hidden
    content = answer_context.with_evidence(body, chosen)
    if len(content.encode('utf-8')) > byte_limit:
        return None
    return dict(content=content, sources=chosen, _evidence_body=body)
