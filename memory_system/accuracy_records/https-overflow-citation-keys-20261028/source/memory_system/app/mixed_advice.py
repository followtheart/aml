"""Candidate prompt routing only; preserve all check text and source evidence."""
import re

_ACTION = re.compile(
    r'^(?:you\s+(?:could|might|can|should|may)\s+|'
    r'(?:use|try|consider|focus|draw|blend|lean|incorporate|explore|turn|channel|'
    r'choose|practice|start|keep|add|create|build|plan|take|make|find|embrace)\b)', re.I)
_PERSONAL = re.compile(r"\byour\b|\byou(?:\s+are|['’]re)\b", re.I)


def mixed_advice_checks(checks):
    ids = []
    for check in checks:
        if check.get('check_type') != 'premise':
            continue
        claim = check.get('claim', '').strip()
        if (not _ACTION.match(claim) or not _PERSONAL.search(claim)
                or re.match(r'^use\s+of\b', claim, re.I)):
            continue
        ids.append(check['claim_id'])
    return ids
