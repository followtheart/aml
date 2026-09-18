"""Compact CE questions: alternatives are hypotheses, never new user facts."""
import re


MAX_AUXILIARY_BYTES = 1800
MAX_PREMISE_CHARS = 240
MAX_PREMISES = 8
MAX_SUBQUESTIONS = 3
_GUIDANCE = ('Verify these alternative premises against the evidence; they are not established facts. '
             'Distinguish the user from third parties, preserve negation, and separate historical, '
             'current and hypothetical claims.')
_PREFIX = re.compile(r'^(?:since|given|because|as|if|when)\b|^(?:既然|因為|因为|由於|由于|如果)', re.I)
_ADVICE = re.compile(r'[,;—.，；。]\s*(?=(?:'
                     r'(?:you|we)\s+(?:could|might|should|can)\b|'
                     r'(?:i|we)(?:[\x27’]d|\s+would)?\s+(?:recommend|suggest)\b|'
                     r'(?:consider|try)\b|你(?:可以|可|應該|应该)|建議|建议))', re.I)


def _premise(option):
    if not isinstance(option, str):
        return ''
    original = re.sub(r'^\s*[A-Z0-9]+[.)、]\s*', '', option).strip()
    if not _PREFIX.search(original):
        return ''
    # Cut only at an explicit recommendation. A second 'you' clause may carry
    # 'no longer', an injury or another qualification that must stay readable.
    boundary = _ADVICE.search(original)
    value = (original[:boundary.start()] if boundary else original).strip(' ,;，；')
    if re.search(r'\b(?:could|might|should|recommend|suggest|consider|try)\b', value, re.I):
        return ''
    if not boundary and (len(original) > MAX_PREMISE_CHARS or re.search(r'[,;，；]', value)):
        return ''  # An ambiguous recommendation tail supplies no safe premise.
    return value if len(value) <= MAX_PREMISE_CHARS else ''


def build(req, plan):
    """Keep the original question intact, adding only bounded premise clauses.

    Use the actual alternatives rather than a planner's rewritten advice. The
    no-context path deliberately returns the original string byte for byte.
    """
    lines, seen = [], {req.query.casefold().strip()}
    for index, option in enumerate(req.options or []):
        value = _premise(option)
        if not value or value.casefold() in seen:
            continue
        if len(lines) >= MAX_PREMISES:
            break
        seen.add(value.casefold())
        lines.append(f'Premise {index + 1}: {value}')
    specs = plan.get('_query_specs')
    if not isinstance(specs, list):
        subqueries = plan.get('sub_queries')
        specs = [dict(text=text, origin='sub_queries') for text in
                 (subqueries if isinstance(subqueries, list) else []) if isinstance(text, str)]
    subquestions = 0
    for spec in specs:
        if not isinstance(spec, dict) or spec.get('origin') not in ('sub_queries', 'grounded_followup'):
            continue
        value = spec.get('text')
        if not isinstance(value, str):
            continue
        value = ' '.join(value.split())
        if not value or len(value) > MAX_PREMISE_CHARS or value.casefold() in seen:
            continue
        if subquestions >= MAX_SUBQUESTIONS:
            break
        seen.add(value.casefold())
        lines.append('Subquestion: ' + value)
        subquestions += 1
    if not lines:
        return req.query
    suffix = '\n\n' + _GUIDANCE
    kept = 0
    for line in lines:
        addition = '\n' + line
        if len((suffix + addition).encode('utf-8')) > MAX_AUXILIARY_BYTES:
            continue
        suffix += addition
        kept += 1
    return req.query + suffix if kept else req.query
