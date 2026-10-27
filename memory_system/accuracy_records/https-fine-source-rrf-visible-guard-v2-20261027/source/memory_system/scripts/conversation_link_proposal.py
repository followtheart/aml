"""Exact conversation structure for source-part evidence review. No new evidence."""
import copy
import json

POLICY = 'adjacent_source_links_v1'

GUIDANCE = (' The reply_to field records consecutive original user and assistant messages as '
            '[assistant_source, user_source] ID pairs, including displayed excerpts. '
            'Read linked text together for conversational context while preserving each speaker and all '
            'question, quotation, fictional, negation and hypothetical framing. A reply does not establish '
            'the truth of its premises or turn assistant material into user facts. These structural links '
            'do not indicate usefulness and grant no additional citation ownership. Judge each candidate '
            'under the original relevance and citation rules.')


def conversation_links(built):
    sources = {}
    keys = {mid: key for key, mid in built['candidate_map'].items()}
    for card in built['cards']:
        for source in card['sources']:
            ids = {built['spans'][pid]['global_source'] for pid, owner in built['owners'][keys[card['candidate_id']]].items()
                   if owner == source['source_id']}
            if len(ids) != 1:
                raise ValueError('Ambiguous original source mapping')
            sid = ids.pop()
            metadata = {k: source.get(k) for k in ('role', 'text', 'timestamp', 'request_id',
                                                  'message_index', 'source_event_id', 'content_span')}
            if sid in sources and sources[sid] != metadata:
                raise ValueError('Shared source metadata differs')
            sources[sid] = metadata
    def usable(s):
        return (isinstance(s['request_id'], str) and bool(s['request_id'])
                and isinstance(s['source_event_id'], str) and bool(s['source_event_id'])
                and type(s['message_index']) is int and s['message_index'] >= 0)
    coordinates, events = {}, {}
    for s in sources.values():
        if usable(s):
            coordinate = (s['request_id'], s['message_index'])
            coordinates.setdefault(coordinate, set()).add((s['source_event_id'], s['role']))
            events.setdefault(s['source_event_id'], set()).add((*coordinate, s['role']))
    def unambiguous(s):
        return (usable(s) and len(coordinates[s['request_id'], s['message_index']]) == 1
                and len(events[s['source_event_id']]) == 1)
    return [dict(user_source=uid, assistant_source=aid)
            for uid, user in sources.items() if user['role'] == 'user' and unambiguous(user)
            for aid, assistant in sources.items() if assistant['role'] == 'assistant' and unambiguous(assistant)
            and assistant['request_id'] == user['request_id']
            and assistant['message_index'] == user['message_index'] + 1]


def proposal(request, built):
    result = copy.deepcopy(request)
    links = conversation_links(built)
    if links:
        prefix, raw = result['prompt'].rsplit('\n', 1)
        data = json.loads(raw)
        if 'reply_to' in data:
            raise ValueError('Links already present')
        pairs = [[edge['assistant_source'], edge['user_source']] for edge in links]
        # Preserve every original display byte; compact only the new metadata.
        if not raw.endswith('}'):
            raise ValueError('Unexpected structured prompt boundary')
        result['prompt'] = prefix + '\n' + raw[:-1] + ',"reply_to":' + json.dumps(pairs,separators=(',',':')) + '}'
        result['system'] += GUIDANCE
        result['reservation'] += len(result['prompt'].encode()) - len(request['prompt'].encode()) + len(GUIDANCE.encode())
    return result, links
