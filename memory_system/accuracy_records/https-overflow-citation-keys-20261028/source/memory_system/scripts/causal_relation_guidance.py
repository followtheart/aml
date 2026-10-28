"""Diagnostic prompt clarification; never grants or removes a fact locally."""
import json
import re

CAUSAL = re.compile(r'\b(?:rooted in|stems? from|driven by|due to|because of|trigger(?:ed|s)? by)\b', re.I)
NOTE = (
    '\nFor an asserted personal causal relationship, verify the relationship itself, '
    'not just the two facts mentioned. Separate evidence of an identity, interest, '
    'condition or past event does not by itself establish that it caused the current '
    'emotion, experience or behavior. In particular, a demographic profile and a '
    'cultural interest alone do not explain why this user feels grief, shame or guilt. '
    'Accept a relationship when the original first-party evidence establishes that '
    'connection, including clear semantic paraphrases. Otherwise keep the personal '
    'causal claim unsupported; do not relabel it as generic. Preserve the distinction '
    'between an asserted cause in the user history and a hypothetical, conditional '
    'or proposed future benefit, which need not assert a prior personal fact. '
    'Do not change source attribution, add evidence or infer a diagnosis.')


def augment(prompt):
    if prompt.count('<checks>') != 1 or prompt.count('</checks>') != 1:
        return prompt, []
    try:
        checks = json.loads(prompt.split('<checks>', 1)[1].split('</checks>', 1)[0])
    except (ValueError, IndexError):
        return prompt, []
    if not isinstance(checks, list):
        return prompt, []
    ids = [c['claim_id'] for c in checks if isinstance(c, dict) and isinstance(c.get('claim_id'), str)
           and CAUSAL.search(str(c.get('claim', c.get('option', ''))))]
    return (prompt + NOTE, ids) if ids else (prompt, [])
