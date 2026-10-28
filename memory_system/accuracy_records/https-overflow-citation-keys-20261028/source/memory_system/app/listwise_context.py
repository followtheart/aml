"""Observable excerpt completeness and bounded, exact context for deletion review.

Never changes the immutable evidence packet or upgrades a support label.
"""
from . import config, personal_evidence


def identity(source):
    return source.get('request_id'), source.get('message_index'), source.get('role')


def diagnose(row):
    sources = row.get('_packet_item', {}).get('sources', [])
    rank = row.get('_rank_text', row.get('content', ''))
    reasons = []
    spans = []
    for source in sources:
        span = source.get('content_span')
        if source.get('content_omitted') or not source.get('content'):
            reasons.append('source_omitted')
        elif source['content'] not in rank:
            # Rank rendering normalizes whitespace in some historical traces.
            if ' '.join(source['content'].split()) not in ' '.join(rank.split()):
                reasons.append('source_not_in_rank_text')
        if span:
            spans.append(dict(source=list(identity(source)), **span))
            if span.get('start', 0) > 0 or span.get('end', 0) < span.get('original_length', 0):
                reasons.append('source_excerpted')
    if row.get('_rank_excerpted') or '[excerpt truncated for ranking]' in rank:
        reasons.append('rank_excerpted')
    if row.get('_body_excerpted'):
        reasons.append('body_excerpted')
    if not sources:
        reasons.append('no_original_sources')
    return dict(context_incomplete=bool(reasons), reasons=sorted(set(reasons)), spans=spans)


def prepare(row, originals, byte_limit=None):
    """Capture exact full sources or sentence windows before storage is released.

    A partial window is still incomplete and cannot authorize an unrelated
    deletion. Windows include adjacent sentences rather than arbitrary prefixes.
    """
    if byte_limit is None:
        byte_limit = config.LISTWISE_CONTEXT_MAX_BYTES
    info = diagnose(row)
    if not info['context_incomplete']:
        return info, []
    visible = row.get('_packet_item', {}).get('sources', [])
    windows, used, seen = [], 0, set()
    for source in originals:
        key = identity(source)
        if key in seen:
            continue
        seen.add(key)
        text = source.get('content', '')
        if not text or not any(identity(s) == key for s in visible):
            continue
        cap = byte_limit - used
        if len(text.encode()) <= cap:
            start, end = 0, len(text)
        else:
            matched = [s for s in visible if identity(s) == key and s.get('content')]
            spans = [s.get('content_span') for s in matched if s.get('content_span')]
            if not spans:
                continue
            lo, hi = min(s['start'] for s in spans), max(s['end'] for s in spans)
            sentences = list(personal_evidence._sentences(text))
            indices = [i for i, s in enumerate(sentences) if s.start() < hi and s.end() > lo]
            if not indices:
                continue
            start = sentences[max(0, indices[0]-1)].start()
            end = sentences[min(len(sentences)-1, indices[-1]+1)].end()
            if len(text[start:end].encode()) > cap:
                continue
        windows.append(dict(source, content=text[start:end],
                            content_span=dict(start=start, end=end, original_length=len(text))))
        used += len(text[start:end].encode())
    info['review_context_bytes'] = used
    return info, windows
