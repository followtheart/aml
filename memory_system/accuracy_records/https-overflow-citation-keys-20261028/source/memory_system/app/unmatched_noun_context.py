"""One whole candidate for question noun matches on novel complete user sources.

The pinned vocabulary supplies retrieval signals, never personal-fact labels.
Original source roles, packet prefix and the original 5 KiB budget are kept.
"""
from collections import Counter
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import zipfile

from . import budget, config, fine_evidence, personal_evidence, source_first_residual
from .conversation_context import usable
from .fine_context_lexical import source_scores
from .fine_context_reserve import full_user, propose as reserve

POLICY = 'unmatched_question_noun_context_v1'
LEXICON = Path(__file__).resolve().parents[1]/'data/lexicons/wordnet.zip'
LEXICON_SHA256 = 'cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59'
INCOMPATIBLE = ('PACKET_FINE_CONTEXT_FILL','PACKET_SOURCE_QUERY_CONTEXT',
    'PACKET_QUESTION_CONTEXT_CE','PACKET_SOURCE_FIRST_RESIDUAL','PACKET_MARGINAL_QUERY_CONTEXT',
    'PACKET_QUERY_CONTEXT_COMBINED','PACKET_CORE_GAP_CONTEXT','PACKET_SPARSE_SOURCE_CONTEXT',
    'PACKET_EXCLUSIVE_SHARED_CONTEXT')


def enabled():
    return (config.PACKET_UNMATCHED_NOUN_CONTEXT and config.PACKET_FINE_CONTEXT_RESERVE
        and config.PACKET_FINE_CONTEXT_5K and config.PACKET_FINE_CONTEXT_LEXICAL
        and not any(getattr(config,k) for k in INCOMPATIBLE))


class NounConcepts:
    def __init__(self,path=LEXICON):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=LEXICON_SHA256:
            raise ValueError('Unexpected WordNet identity')
        with zipfile.ZipFile(path) as z:
            self.index={}
            for line in z.read('wordnet/index.noun').decode().splitlines():
                if not line or line.startswith(' '):
                    continue
                p=line.split();self.index[p[0]]=p[6+int(p[3])]
            self.exceptions={p[0]:p[1:] for p in (line.split() for line in z.read('wordnet/noun.exc').decode().splitlines())}

    def lemma(self,word):
        if word in self.index:
            return word
        forms=self.exceptions.get(word,[])
        if not forms:
            forms=[word[:-len(old)]+new for old,new in (
                ('s',''),('ses','s'),('ves','f'),('xes','x'),('zes','z'),('ches','ch'),
                ('shes','sh'),('men','man'),('ies','y')) if word.endswith(old)]
        return next((w for w in forms if w in self.index),word)

    def terms(self,text):
        result=set()
        for word in personal_evidence.terms(text):
            lemma=self.lemma(word)
            result.add('wn_noun:'+self.index[lemma] if lemma in self.index else word)
        return result


@lru_cache(maxsize=1)
def lexicon():
    return NounConcepts()


def identity(source):
    return tuple(source.get(k) for k in
        ('user_id','request_id','message_index','source_event_id','role','content'))


def nominate(pool,packed,*,coarse_ids,query,user_id,top_k,byte_budget,remaining_bytes):
    try:
        concepts=lexicon()
    except (OSError,ValueError,zipfile.BadZipFile,KeyError):
        return [],dict(status='lexicon_unavailable',nominations=[],ordered_ids=[])
    corpus=[pool[mid] for mid in coarse_ids if mid in pool]
    eligible,_=source_scores(corpus,user_id=user_id,query=query,options=[])
    known={identity(s) for p in packed for s in p.get('sources',[]) if usable(s,user_id) and full_user(s)}
    cache={}
    def terms(text):
        if text not in cache:
            cache[text]=concepts.terms(text)
        return cache[text]
    documents={};members={}
    for c in corpus:
        budget.check()
        if c['id'] not in eligible:
            continue
        sources=[s for s in c['_packet_item']['sources'] if s.get('role')=='user' and s.get('content')]
        members[c['id']]=sources
        for s in sources:
            documents.setdefault(identity(s)[:-1],set()).update(terms(s['content']))
    wanted=terms(query);literal_query=personal_evidence.terms(query)
    frequency=Counter(w for words in documents.values() for w in words)
    n=len(documents);average=sum(map(len,documents.values()))/max(1,n)
    weights={w:math.log(1+(n-count+.5)/(count+.5)) for w,count in frequency.items()}
    kept={p['id'] for p in packed};candidates=[];nominations=[]
    for c in corpus:
        budget.check()
        mid=c['id']
        if mid not in members:
            continue
        matches=[]
        for s in members[mid]:
            if not full_user(s) or identity(s) in known:
                continue
            words=terms(s['content'])
            shared=personal_evidence.terms(s['content'])&literal_query
            explained=terms(' '.join(sorted(shared)))
            hits={w for w in (words&wanted)-explained if w.startswith('wn_noun:')}
            if not hits:
                continue
            value=sum(weights[w] for w in sorted(hits))*2.2/(1+1.2*(.25+.75*len(words)/max(1,average)))
            matches.append(dict(source_event_id=s['source_event_id'],source_identity=list(identity(s)[:-1]),
                source_sha256=hashlib.sha256(s['content'].encode()).hexdigest(),score=value,
                matched_terms=sorted(hits),literal_shared_terms=sorted(shared)))
        if not matches:
            continue
        group,missing=fine_evidence.closure(c,pool)
        if missing or not group<=kept|{mid}:
            continue
        _,trial=reserve(packed,[c],user_id=user_id,top_k=top_k,
            byte_budget=byte_budget,reserve_bytes=remaining_bytes,limit=1)
        if not trial['appended_ids']:
            continue
        candidates.append(c)
        nominations.append(dict(candidate_id=mid,score=max(m['score'] for m in matches),
            ce_score=c['_ce_score'],cost=trial['added_bytes'],matches=matches))
    by_id={n['candidate_id']:n for n in nominations}
    candidates.sort(key=lambda c:(-by_id[c['id']]['score'],-c['_ce_score'],c['id']))
    return candidates,dict(status='evaluated',lexicon_sha256=LEXICON_SHA256,
        corpus_observations=len(documents),nominations=nominations,ordered_ids=[c['id'] for c in candidates])


def attach(packed,packet_hash,manifest,candidates,nomination,*,user_id,top_k):
    result,digest,updated=source_first_residual.attach(packed,packet_hash,manifest,
        candidates,nomination,user_id=user_id,top_k=top_k)
    context=dict(updated['conversation_context'])
    context['unmatched_noun']=context.pop('source_first');context['policy']=POLICY
    return result,digest,dict(updated,conversation_context=context)
