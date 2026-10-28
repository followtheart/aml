"""Conservative exact coverage of already visible source intervals."""
import json


def atom(source):
    text = source.get('content')
    if not isinstance(text, str) or not text or source.get('content_omitted'):
        return None
    if source.get('role') not in ('user', 'assistant') or not source.get('user_id') or not source.get('source_event_id'):
        return None
    if not source.get('request_id') or type(source.get('message_index')) is not int:
        return None
    span = source.get('content_span')
    if span is None:
        if source.get('content_excerpted'):
            return None
        start, end, length = 0, len(text), len(text)
    else:
        start, end, length = (span.get(k) for k in ('start', 'end', 'original_length'))
        if any(type(v) is not int for v in (start, end, length)) or not 0 <= start < end <= length:
            return None
        if end-start != len(text):
            return None
    metadata = {k: v for k, v in source.items()
                if k not in ('content', 'content_span', 'content_excerpted', 'content_omitted')}
    return dict(key=json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(',', ':')),
                start=start, end=end, original_length=length, text=text)


def visible_atoms(candidate):
    item = candidate.get('_packet_item', {})
    rank = candidate.get('_rank_text', '')
    display = item.get('content', '')
    result = []
    for source in item.get('sources', []):
        view = atom(source)
        if view is None or view['text'] not in rank or view['text'] not in display:
            return None
        result.append(view)
    return result or None


def covers(old, new):
    return (old['key'] == new['key'] and old['original_length'] == new['original_length']
            and new['start'] <= old['start'] and old['end'] <= new['end']
            and new['text'][old['start']-new['start']:old['end']-new['start']] == old['text'])


def covered(required, available):
    return required is not None and bool(required) and all(any(covers(old, new) for new in available)
                                                         for old in required)
