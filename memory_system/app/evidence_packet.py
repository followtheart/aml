"""Immutable, whole-item evidence packing shared with answer consumption.

UTF-8 bytes provide a conservative token upper bound for byte-based tokenizers;
no text or supporting quote is silently clipped after verification.
"""
import hashlib
import json


def digest(items):
    payload = [{'id': x['id'], 'content': x['content']} for x in items]
    versions = {x.get('packet_hash_version') for x in items}
    if versions == {2}:
        payload = [dict(p, memory_type=x.get('memory_type', 'fact'), sources=x.get('sources', []),
                        personal_evidence=x.get('personal_evidence'))
                   for p, x in zip(payload, items)]
    elif versions - {None, 1}:
        raise ValueError('Mixed or unsupported evidence packet hash versions')
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def pack(items, top_k, token_budget, core_budget=None):
    selected, omitted, used, core_used = [], [], 0, 0
    seen_sources = {}
    for item in items:
        from . import answer_context
        item = dict(item)
        sources = [dict(s) for s in item.get('sources', [])]
        for source in sources:
            key = (source['request_id'], source['message_index'])
            span = source.get('content_span')
            interval = (span['start'], span['end']) if span else (0, float('inf'))
            if source.get('content') and any(a <= interval[0] and b >= interval[1]
                                           for a, b in seen_sources.get(key, [])):
                source.pop('content', None)
                source['content_omitted'] = 'duplicate'
        if sources:
            item['content'] = answer_context.with_evidence(item['content'].split(answer_context.SOURCE_MARKER, 1)[0], sources)
            item['sources'] = sources
        cost = len(item['content'].encode('utf-8')) + bool(selected)
        is_core = item.pop('_core_injected', False)
        if is_core and core_budget is not None and core_used + cost > core_budget:
            omitted.append({'id': item['id'], 'reason': 'core_budget', 'cost': cost})
            continue
        if len(selected) >= top_k or used + cost > token_budget:
            omitted.append({'id': item['id'], 'reason': 'top_k' if len(selected) >= top_k else 'token_budget', 'cost': cost})
            continue
        selected.append(dict(item, packet_hash_version=2))
        for s in sources:
            if not s.get('content'):
                continue
            span = s.get('content_span')
            seen_sources.setdefault((s['request_id'], s['message_index']), []).append(
                (span['start'], span['end']) if span else (0, float('inf')))
        used += cost
        if is_core:
            core_used += cost
    packet_hash = digest(selected)
    for item in selected:
        item['packet_hash'] = packet_hash
    return selected, packet_hash, {'included_ids': [x['id'] for x in selected],
                                    'omitted': omitted, 'token_upper_bound': used,
                                    'token_budget': token_budget, 'estimator': 'utf8_bytes',
                                    'core_bytes': core_used, 'core_budget': core_budget}
