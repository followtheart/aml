"""Diagnostic-only literal repeated-inquiry support for a narrow interest claim."""
import re

# Exact claim frame only: do not consume coordinated or qualified personal claims.
CLAIM=re.compile(r'^(?:your interest in|you are interested in|you[’\']re interested in) ([\w -]+)[.!?,]?$',re.I)
EXCLUDED=re.compile(r'\b(?:and|or|with|own\w*|collect\w*|expert\w*|professional\w*|daily|weekly|always|never|favorite|favourite|not|diagnos\w*|treat\w*)\b',re.I)
CONTEXT=re.compile(r'\b(?:if|hypothetical\w*|imagine|pretend|roleplay|fiction\w*|character|colleague|friend|client|student|teacher|assignment|homework|survey|research project|on behalf|not interested|don.t like)\b',re.I)
QUESTION=re.compile(r'^(?:why\b|how\b|what\b|can you (?:give|show|recommend|suggest|explain)\b|could you (?:give|show|recommend|suggest|explain)\b)',re.I)

def words(text):
    tokens=re.findall(r'\b\w+\b',text.casefold())
    # Minimal final plural agreement, not semantic expansion.
    return [t[:-1] if len(t)>4 and t.endswith('s') and not t.endswith('ss') else t for t in tokens]

def supports(check):
    if check.get('check_type')!='premise' or check.get('premise_type')!='interest':return False
    m=CLAIM.fullmatch(check.get('claim','').strip())
    if not m or EXCLUDED.search(m[1]):return False
    topic=words(m[1])
    if not 2<=len(topic)<=6:return False
    witnesses={}
    for s in check.get('sources',[]):
        quote=s.get('quote','').strip();context=s.get('context','')
        if s.get('role')!='user' or not quote or quote not in context or CONTEXT.search(context):continue
        if not QUESTION.search(quote) or not quote.endswith('?'):continue
        tokens=words(quote)
        if not any(tokens[i:i+len(topic)]==topic for i in range(len(tokens)-len(topic)+1)):continue
        witnesses.setdefault(' '.join(tokens),set()).add(s.get('id'))
    return len(witnesses)>=2 and len(set().union(*witnesses.values()))>=2 if witnesses else False
