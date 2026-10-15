"""Diagnostic scope rule: an unspecified preferred object inside new advice."""
import re
from app.choice_premises import is_suggestion

OPEN = re.compile(r'(?:[a-z]+ing|listening to)\s+(?:something|anything)\s+(?:that\s+)?you\s+(?:enjoy|like|find relaxing)', re.I)
QUALIFIER = re.compile(r'\b(?:since|because|given|already|own|owned|have|had|has|daily|weekly|usually|always|previously|before|after|last|your|used to)\b', re.I)

def supports(claim, option):
    if not OPEN.fullmatch(claim.strip()):
        return False
    occurrences=list(re.finditer(re.escape(claim),option,re.I))
    if len(occurrences)!=1:
        return False
    hit=occurrences[0]
    remainder=re.split(r'[.!?;\n]',option[hit.end():],maxsplit=1)[0]
    if QUALIFIER.search(remainder):
        return False
    # Replace only the open selector for the established advice-scope check.
    # The original option and independent whole-option verification stay intact.
    shorter=re.sub(r'\s+(?:that\s+)?you\s+(?:enjoy|like|find relaxing)$','',claim,flags=re.I)
    rewritten=option[:hit.start()]+shorter+option[hit.end():]
    return is_suggestion(shorter,rewritten)
