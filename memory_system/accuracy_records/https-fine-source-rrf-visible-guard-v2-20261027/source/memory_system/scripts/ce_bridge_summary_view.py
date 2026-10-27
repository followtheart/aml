"""CE-only view of retrieval bridge summaries backed by complete visible sources.

Keep the primary body, all labels, declared dependencies and every source byte.
Only a retrieval-only support summary can become `Conversation excerpt:` when
all of its original source messages are already visible intact in this unit.
This is a scoring experiment, not a change to displayed evidence or support.
"""
import hashlib
from app import answer_context

PLACEHOLDER = 'Conversation excerpt:'


def identity(source):
    return source.get('request_id'), source.get('message_index')


def source_complete(original, visible):
    return any(identity(source) == identity(original)
               and source.get('role') == original.get('role')
               and source.get('timestamp') == original.get('timestamp')
               and original.get('content') and original['content'] in source.get('content', '')
               for source in visible)


def view(candidate, record, originals):
    old = candidate.get('_rank_text', candidate.get('content', ''))
    unit = candidate.get('_packet_item') or {}
    body = unit.get('_evidence_body')
    sources = unit.get('sources') or []
    def unchanged(reason):
        return old, dict(changed=False, reason=reason)
    if not candidate.get('_bridge_ids'):
        return unchanged('no_retrieval_bridge')
    if record is None or record['candidate_id'] != candidate['id']:
        raise ValueError('Missing authenticated bridge provenance')
    if (candidate.get('type') == 'rule' or candidate.get('_user_rule') or unit.get('is_constraint')):
        return unchanged('protected_primary')
    if candidate.get('_relation_path_ids') or candidate.get('_coverage_proofs'):
        return unchanged('recorded_path_or_coverage_proof')
    if (candidate.get('_body_excerpted') or candidate.get('_rank_excerpted')
            or not record['body_exact'] or not record['rank_exact'] or not isinstance(body, str)):
        return unchanged('unavailable_intact_view')
    if answer_context.with_evidence(body, sources) != old:
        raise ValueError('Scoring view does not match authenticated sources')
    supports = record['supports']
    body_bytes = body.encode()
    primary_end = len(body_bytes) - sum(s['block_bytes'] for s in supports)
    if primary_end <= 0:
        raise ValueError('Invalid support byte boundaries')
    proposed = [body_bytes[:primary_end]]
    changed, skipped, offset = [], [], primary_end
    for support in supports:
        block = body_bytes[offset:offset + support['block_bytes']]
        if (hashlib.sha256(block).hexdigest() != support['block_sha256']
                or not block.startswith(('\n[support: ' + support['id'] + '] ').encode())):
            raise ValueError('Unauthenticated support block')
        summary_bytes = support['summary_bytes']
        if not 0 < summary_bytes < len(block):
            raise ValueError('Invalid support summary boundary')
        boundary = len(block) - summary_bytes
        block[:boundary].decode('utf-8')
        block[boundary:].decode('utf-8')
        original_sources = originals.get(support['id'], [])
        reason = None
        if support['origin'] != 'retrieval_bridge':
            reason = 'declared_dependency'
        elif support['protected']:
            reason = 'protected_support'
        elif not original_sources or not all(source_complete(s, sources) for s in original_sources):
            reason = 'incomplete_original_support_sources'
        elif summary_bytes <= len(PLACEHOLDER.encode()):
            reason = 'no_byte_saving'
        if reason:
            proposed.append(block)
            skipped.append(dict(id=support['id'], reason=reason))
        else:
            proposed.append(block[:boundary] + PLACEHOLDER.encode())
            changed.append(dict(id=support['id'], start=offset + boundary,
                                end=offset + len(block), source_count=len(original_sources)))
        offset += len(block)
    if offset != len(body_bytes):
        raise ValueError('Incomplete support reconstruction')
    if not changed:
        return old, dict(changed=False, reason='no_eligible_bridge_summary', skipped=skipped)
    suffix = old[len(body):]
    new = b''.join(proposed).decode('utf-8') + suffix
    return new, dict(changed=True, reason='retrieval_bridge_summary_replaced',
                     spans=changed, skipped=skipped, primary_bytes=primary_end,
                     source_suffix_bytes=len(suffix.encode()), saved_bytes=len(old.encode())-len(new.encode()))
