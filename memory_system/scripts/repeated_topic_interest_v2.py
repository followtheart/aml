"""Require full direct questions; never infer interest from extracted task quotations."""
import re
from repeated_topic_interest import supports as original_supports
TASK_OR_NEGATION=re.compile(r'\b(?:translat\w*|exam(?:s|ination)?|copy|copied|verbatim|quiz|test question|zero interest|no interest|not|never|uninterest\w*|disinterest\w*|hate\w*|dislike\w*|avoid\w*)\b',re.I)

def supports(check):
    sources=[]
    for s in check.get('sources',[]):
        quote=s.get('quote','').strip();context=s.get('context','').strip()
        if quote!=context or quote.count('?')!=1 or TASK_OR_NEGATION.search(context):continue
        sources.append(s)
    return original_supports(dict(check,sources=sources))
