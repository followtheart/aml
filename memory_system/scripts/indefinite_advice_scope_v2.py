"""Conservative diagnostic extension: require a direct advice/list complement."""
import re
from indefinite_advice_scope import supports as original_supports

DIRECT = re.compile(r'\b(?:try|consider|start|suggest|recommend)\s*$', re.I)
LIST = re.compile(r'\b(?:like|including|such as)\s+(?P<items>[^.!?;]*)$', re.I)
HISTORY = re.compile(r'\b(?:remember\w*|recall\w*|recollect\w*|reflect\w*|'
                     r'yesterday|ago|earlier|formerly|during|helped|bought|owned|'
                     r'visited|learned|learnt|grew|was|were|did|used|had)\b', re.I)
ASSERTION = re.compile(r'\b(?:you|your|my|our|his|her|their|since|because|given|when|if|after|before)\b', re.I)

def supports(claim, option):
    if not original_supports(claim,option):return False
    hit=re.search(re.escape(claim),option,re.I)
    start=max(option.rfind(mark,0,hit.start()) for mark in ('. ',';','\n'))
    prefix=option[start+1:hit.start()]
    tail=re.split(r'[.!?;\n]',option[hit.end():],maxsplit=1)[0]
    if HISTORY.search(prefix+claim+tail):return False
    if DIRECT.search(prefix):return True
    match=LIST.search(prefix)
    # A coordinated example list can precede this object; avoid crossing an
    # intervening personal/temporal clause to reach a distant advice marker.
    return bool(match and not ASSERTION.search(match['items']) and
                re.search(r'\b(?:try|consider|could|should|might|suggest|recommend)\b',prefix[:match.start()],re.I))
