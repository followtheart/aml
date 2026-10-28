"""Offline noun-sense normalization for original-source lexical retrieval.

Use WordNet 3.0's first noun sense and standard noun morphology, not a
dataset-specific equivalence list. No facts or preference labels are inferred.
"""
from collections import Counter
import hashlib
import math
import zipfile

from app import personal_evidence
from app.fine_context_lexical import source_scores

WORDNET_SHA256='cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59'


class NounConcepts:
    def __init__(self,path):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=WORDNET_SHA256:
            raise ValueError('Unexpected WordNet corpus identity')
        with zipfile.ZipFile(path) as z:
            self.index={}
            for line in z.read('wordnet/index.noun').decode().splitlines():
                if not line or line.startswith(' '):continue
                p=line.split();self.index[p[0]]=p[6+int(p[3])]
            self.exceptions={p[0]:p[1:] for p in (l.split() for l in z.read('wordnet/noun.exc').decode().splitlines())}

    def lemma(self,word):
        if word in self.index:return word
        forms=self.exceptions.get(word,[])
        if not forms:
            forms=[word[:-len(old)]+new for old,new in (
                ('s',''),('ses','s'),('ves','f'),('xes','x'),('zes','z'),
                ('ches','ch'),('shes','sh'),('men','man'),('ies','y')) if word.endswith(old)]
        return next((w for w in forms if w in self.index),word)

    def terms(self,text):
        result=set()
        for word in personal_evidence.terms(text):
            lemma=self.lemma(word)
            result.add('wn_noun:'+self.index[lemma] if lemma in self.index else word)
        return result


def scores(fine,*,user_id,query,options,concepts,novel_from=None):
    eligible,_=source_scores(fine,user_id=user_id,query=query,options=options)
    corpus={};members={}
    for candidate in fine:
        if candidate['id'] not in eligible:continue
        users=[s for s in candidate['_packet_item']['sources'] if s.get('role')=='user' and s.get('content')]
        members[candidate['id']]=users
        for s in users:
            key=(s['user_id'],s['request_id'],s['message_index'],s['source_event_id'],s['role'])
            corpus.setdefault(key,set()).update(concepts.terms(s['content']))
    wanted=concepts.terms(' '.join([query,*options]))
    if novel_from is not None:
        represented=set().union(*(concepts.terms(s['content']) for p in novel_from
            for s in p.get('sources',[]) if s.get('role')=='user' and s.get('content')))
        wanted={term for term in wanted-represented if term.startswith('wn_noun:')}
    frequency=Counter(t for words in corpus.values() for t in words)
    n=len(corpus);avg=sum(map(len,corpus.values()))/max(1,n)
    weights={t:math.log(1+(n-count+.5)/(count+.5)) for t,count in frequency.items()}
    result={};details={}
    for mid,users in members.items():
        values=[]
        for s in users:
            words=concepts.terms(s['content']);hits=words&wanted
            value=sum(weights[t] for t in sorted(hits))*2.2/(1+1.2*(.25+.75*len(words)/max(1,avg)))
            values.append(dict(source_event_id=s['source_event_id'],role='user',score=value,matched_terms=sorted(hits)))
        result[mid]=max((v['score'] for v in values),default=0);details[mid]=values
    return result,details
