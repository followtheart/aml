"""Diagnostic: bounded first-person stance over a general proposition."""
import re
OPEN=re.compile(r"^I(?:['’]ve| have)\s+(?:also\s+)?found that\s+(.+)$",re.I)
PERSON=re.compile(r"\b(?:you|your|yours|I|my|mine|we|our|ours|he|his|she|her|hers|they|their|theirs)\b",re.I)
HISTORY=re.compile(r"\b(?:already|previously|yesterday|ago|last|used to|owned|bought|had|was|were|became|diagnosed|remember|recall)\b|\d",re.I)
FRAME=re.compile(r"\b(?:say|write|draft|quote|message|tell|reply|replied|said|wrote)\b",re.I)
def supports(claim,option):
    match=OPEN.fullmatch(claim.strip())
    if not match:return False
    body=match[1]
    if PERSON.search(body) or HISTORY.search(body) or re.search(r'\b[A-Z][a-z]+\b',body):return False
    if re.match(r'(?:the|this|that|these|those)\b',body,re.I):return False
    if not re.search(r'\b(?:can|helps?|works?|is|are)\b',body,re.I):return False
    hits=list(re.finditer(re.escape(claim),option,re.I))
    if len(hits)!=1:return False
    hit=hits[0];prefix=option[:hit.start()]
    if FRAME.search(prefix) or re.search(r'["“”]',prefix):return False
    if prefix.strip() and not re.search(r'(?:[.!?]\s+|^[A-Z][.)]\s*)$',prefix):return False
    tail=option[hit.end():]
    return not tail.strip() or bool(re.match(r'\s*[.!?](?:\s|$)',tail))
