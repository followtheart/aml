"""Diagnostic CE-only view: atomic persona fields retain their own quoted context.

The original ranking/packing unit and source catalog stay intact. This removes
only separate optional persona passages from the CE scoring document.
"""
from app import answer_context


def document(candidate):
    original = candidate.get('_rank_text', candidate.get('content', ''))
    unit = candidate.get('_packet_item', {})
    evidence = candidate.get('evidence') or []
    sources = unit.get('sources') or []
    body = unit.get('_evidence_body')
    if (candidate.get('type') not in ('profile', 'preference') or not evidence
            or not isinstance(body, str) or '[support:' in body
            or candidate.get('_bridge_ids') or candidate.get('_equivalent_ids')
            or any(e.get('source_role') != 'persona' or not e.get('quote') for e in evidence)
            or any(s.get('content') and (s.get('source_role') != 'persona' or s.get('role') != 'user')
                   for s in sources)):
        return original
    keep = []
    witnessed = set()
    for source in sources:
        matches = [index for index, e in enumerate(evidence)
                   if (e.get('request_id'), e.get('message_index')) ==
                      (source.get('request_id'), source.get('message_index'))
                   and e['quote'] in source.get('content', '')]
        if matches:
            keep.append(source)
            witnessed.update(matches)
    if len(witnessed) != len(evidence) or len(keep) == len(sources):
        return original
    result = answer_context.with_evidence(body, keep)
    return result if len(result.encode()) < len(original.encode()) else original
