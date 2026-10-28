"""CE-only ablation of inferred primary assertions, with intact source suffix."""
import re
from app import answer_context

INFERRED = '[inferred; not stated by the user] '
PLACEHOLDER = 'Conversation excerpt:'
LABEL_PREFIX = re.compile(r'(?:\[[^\]\n]+\] +)+\Z')


def view(candidate):
    original = candidate.get('_rank_text', candidate.get('content', ''))
    unit = candidate.get('_packet_item') or {}
    body = unit.get('_evidence_body')
    primary = candidate.get('content')
    sources = unit.get('sources') or []
    reason = None
    if candidate.get('epistemic_status') != 'inferred':
        reason = 'not_inferred'
    elif (candidate.get('type') == 'rule' or candidate.get('_user_rule')
            or unit.get('is_constraint')):
        reason = 'protected_rule_or_constraint'
    elif (candidate.get('_body_excerpted') or candidate.get('_rank_excerpted')
            or not isinstance(body, str) or not isinstance(primary, str) or not primary):
        reason = 'unavailable_intact_primary'
    elif not any(s.get('content') for s in sources):
        reason = 'no_visible_sources'
    elif answer_context.with_evidence(body, sources) != original:
        reason = 'unauthenticated_source_suffix'
    if reason:
        return original, dict(changed=False, reason=reason)
    start = body.find(primary)
    prefix = body[:start] if start >= 0 else ''
    tail = body[start + len(primary):] if start >= 0 else ''
    if (start < 0 or INFERRED not in prefix or not LABEL_PREFIX.fullmatch(prefix)
            or tail and not tail.startswith('\n[support: ')):
        return original, dict(changed=False, reason='ambiguous_primary_boundary')
    replacement = prefix + PLACEHOLDER + tail
    proposed = replacement + original[len(body):]
    if len(proposed.encode()) >= len(original.encode()):
        return original, dict(changed=False, reason='no_byte_saving')
    return proposed, dict(changed=True, reason='inferred_primary_removed',
        primary_bytes=len(primary.encode()), saved_bytes=len(original.encode())-len(proposed.encode()),
        source_suffix_bytes=len(original[len(body):].encode()))


def document(candidate):
    return view(candidate)[0]
