"""REJECTED diagnostic: direct purpose grammar can hide factual presuppositions."""
import re
ADVICE = re.compile(r'^(?:[A-Z][.)]\s*)?(?:it\s+(?:might|may|could)\s+help\s+to\s+|you\s+(?:could|should|might)\s+|(?:try|consider)\s+)?(?:create|build|establish|set|arrange|plan|keep|take|begin|connect|pair)\b',re.I)
HISTORY = re.compile(r"\b(?:already|previously|yesterday|ago|before|since|because|given|used to|remember|recall|diagnos\w*|disease|condition|medication|degree|credential|certificate|owned|bought|had|was|were|did)\b",re.I)
PRESUPPOSED = re.compile(r"\b(?:your|my|our|his|her|their|those|these|existing|usual|regular|still|again|always|daily|weekly|keep|continue)\b",re.I)
RESULT = re.compile(r'you\s+(?:(?:can|will)\s+)?(?:feel\b|(?:know\s+you\s+)?have\s+(?:a|an)\b)',re.I)

def supports(claim, option):
    hits=list(re.finditer(re.escape(claim),option,re.I)) if claim else []
    if len(hits)!=1 or not RESULT.match(claim.strip()):return False
    hit=hits[0]
    start=max(option.rfind(mark,0,hit.start()) for mark in ('. ', ';', '\n'))
    prefix=option[start+1:hit.start()].lstrip()
    # Do not erase a main-clause result introduced by causal "so" after a fact.
    if not re.search(r'\bso\s+(?:that\s+)?$',prefix,re.I) or not ADVICE.match(prefix):return False
    tail=re.split(r'[.!?;\n]',option[hit.end():],maxsplit=1)[0]
    if HISTORY.search(prefix+claim+tail) or PRESUPPOSED.search(claim+tail):return False
    return True
