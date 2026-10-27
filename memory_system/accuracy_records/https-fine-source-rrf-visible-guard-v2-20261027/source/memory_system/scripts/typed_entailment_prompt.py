"""Diagnostic-only normalization of obsolete boolean verifier instructions."""
import re

REPLACEMENTS = [
 ('Return claim_id and entailed for EVERY check.', 'Return claim_id and verdict for EVERY check.'),
 ('True means ALL pre-existing personal premises in the entire option are\nestablished by the supplied original sources, or the option has NO such premises.',
  'personal_supported means the option asserts prior personal premises and ALL are\nestablished by the supplied original sources. no_personal_premise means it asserts NONE.'),
 ('False means ANY personal premise is unsupported,', 'unsupported_personal means ANY personal premise is unsupported,'),
 ('qualification, entailed must be false.', 'qualification, verdict must be unsupported_personal.'),
 ('personal premise from these sources alone, use false.', 'personal premise from these sources alone, use unsupported_personal.'),
 ('return true even when sources are empty, unrelated, or the advice is less useful.',
  'return no_personal_premise even when sources are empty, unrelated, or the advice is less useful.'),
 ('For a premise check, unrelated or empty sources must result in false.',
  'For a premise asserting a prior personal fact, unrelated or empty sources must result in unsupported_personal.'),
 ('return true only when those anchors establish the actual premise.',
  'return personal_supported only when those anchors establish the actual premise.'),
]

def normalize(prompt):
    before, remainder = prompt.split('<checks>\n', 1)
    checks, after = remainder.split('\n</checks>', 1)
    for old, new in REPLACEMENTS:
        if before.count(old) != 1: raise ValueError('Expected one legacy instruction: ' + old)
        before = before.replace(old, new, 1)
    old = 'a mistakenly extracted recommendation must be false for a premise check.'
    if old in before:
        before = before.replace(old, 'a mistakenly extracted recommendation has verdict no_personal_premise.', 1)
    if re.search(r'\b(?:true|false|entailed)\b', before, re.I):
        raise ValueError('Unconverted boolean instruction')
    old = 'Output verdict instead of entailed.'
    if after.count(old) != 1: raise ValueError('Typed contract suffix changed')
    after = after.replace(old, 'Output verdict.', 1)
    return before + '<checks>\n' + checks + '\n</checks>' + after
