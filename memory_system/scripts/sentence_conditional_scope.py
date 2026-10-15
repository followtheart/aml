"""Diagnostic extension of existing conditional-interest scope to later sentences."""
import copy
import re

BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=If\s+you\b)', re.I)
FRAME = re.compile(r'\b(?:quote|quoted|write|draft|say|said|tell|told|message|reply|example)\b', re.I)
ABBREVIATION = re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc|e\.g|i\.e)\.$', re.I)


def reason(claim, option, existing_rule):
    # Reuse the existing predicate and nested-fact exclusions. Never extend its
    # lexical interest vocabulary or remove claims outside the antecedent.
    if re.search(r'["“”«»()\[\]]', option) or FRAME.search(option):
        return None
    span = claim.get('option_span') or {}
    start, end = span.get('start'), span.get('end')
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(option):
        return None
    if option[start:end] != claim.get('text'):
        return None
    for boundary in BOUNDARY.finditer(option):
        offset = boundary.end()
        prefix = option[:boundary.start()]
        if start < offset or prefix.endswith('..') or ABBREVIATION.search(prefix):
            continue
        shifted = copy.deepcopy(claim)
        shifted['option_span'] = {'start': start - offset, 'end': end - offset}
        if existing_rule(shifted, option[offset:]) == 'conditional_interest_not_asserted':
            return 'conditional_interest_not_asserted'
    return None
