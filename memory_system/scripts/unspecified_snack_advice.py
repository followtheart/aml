"""Diagnostic scope rule for an unspecified snack choice in proposed advice."""
import re
CLAIM=re.compile(r'your (?:favorite|favourite|preferred) snacks?',re.I)
ADVICE=re.compile(r'\b(?:you (?:could|can|might)|consider|try|suggest|recommend)\b',re.I)
HISTORY=re.compile(r'\b(?:since|because|given|already|always|usually|remember|used to|bought|owned|last|yesterday|ago)\b',re.I)
FRAME=re.compile(r'\b(?:write|say|quote|draft|reply|message)\b|["“”]',re.I)
def supports(claim,option):
    if not CLAIM.fullmatch(claim.strip()):return False
    hits=list(re.finditer(re.escape(claim),option,re.I))
    if len(hits)!=1:return False
    hit=hits[0];start=max(option.rfind('.',0,hit.start()),option.rfind('!',0,hit.start()),option.rfind('?',0,hit.start()))+1
    before=option[start:hit.start()]
    if not ADVICE.search(before) or HISTORY.search(before) or FRAME.search(option[:hit.start()]):return False
    if not re.search(r'(?:with|bring|pack|packing|choose|choosing|include|including)\s+$',before,re.I):return False
    after=option[hit.end():]
    if re.match(r'\s*,?\s*(?:that|which|from|because|since|given|you)\b',after,re.I):return False
    return bool(re.match(r'(?:\s*[,.;!?]|\s+(?:and|or)\b|\s*$)',after,re.I))
