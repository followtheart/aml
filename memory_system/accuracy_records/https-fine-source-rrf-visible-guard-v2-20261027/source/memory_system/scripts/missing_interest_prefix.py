"""Diagnostic only: preserve omitted explicit introductory preference premises."""
import copy,re
from app import answer_choice as ac
OPEN=re.compile(r'^\s*(?:[A-Z][.)]\s*)?(Given your (?:love for|interest in|preference for) [^,;.!?\n]+[,;])',re.I)
UNCERTAIN=re.compile(r'\b(?:if|unless|might|could|may|would|imagine|suppose|hypothetical|fictional)\b',re.I)
def prefix(option):
    m=OPEN.match(option)
    return m[1] if m and not UNCERTAIN.search(m[1]) else None

def apply(entries,sources,diagnostics):
    output=copy.deepcopy(entries)
    for i,e in enumerate(output):
        text=prefix(e['option'])
        if not text or e['kind']!='generic' or e.get('claims') or e.get('validation_status')!='valid':continue
        payload={'options':[{'letter':e['letter'],'kind':'personal','claims':[{'text':text,'status':'unsupported','premise_type':'interest','reason':'no_source','citations':[]}]}]}
        checked=ac.validate_assessments(payload,[e['option']],sources,expected={e['letter']:e['option']})[0]
        assert checked['claims'] and checked['claims'][0]['status']=='unsupported'
        output[i]=checked
        diagnostics.setdefault('missing_interest_prefix',[]).append({'letter':e['letter'],'text':text})
    return output
