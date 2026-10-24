"""One novel source for question noun equivalences missed by literal matching.

Use the pinned general WordNet first-noun map. A matching concept contributes
only when no exact token shared by question and that source accounts for it.
Non-noun and already literal matches contribute zero. Require the scoring
message itself to be complete and not already delivered. Whole candidates,
corpus statistics, original packet and every capacity remain unchanged.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import probe_question_lexical_context as factorial
import probe_novel_source_lexical_context as novel
from probe_source_rank_residual import main as score
from app import personal_evidence


def expanded_noun_scores(corpus,*,user_id,query,options,concepts):
    if options:
        raise ValueError('Original question only')
    eligible,_=factorial.source_scores(corpus,user_id=user_id,query=query,options=[])
    documents={};members={}
    for c in corpus:
        if c['id'] not in eligible:
            continue
        sources=[s for s in c['_packet_item']['sources'] if s.get('role')=='user' and s.get('content')]
        members[c['id']]=sources
        for s in sources:
            key=tuple(s.get(k) for k in ('user_id','request_id','message_index','source_event_id','role'))
            documents.setdefault(key,set()).update(concepts.terms(s['content']))
    wanted=concepts.terms(query);literal_query=personal_evidence.terms(query)
    frequency=Counter(w for words in documents.values() for w in words)
    n=len(documents);average=sum(map(len,documents.values()))/max(1,n)
    weights={w:math.log(1+(n-count+.5)/(count+.5)) for w,count in frequency.items()}
    scores={};matches={}
    for mid,sources in members.items():
        values=[]
        for s in sources:
            words=concepts.terms(s['content'])
            shared=personal_evidence.terms(s['content'])&literal_query
            explained=concepts.terms(' '.join(sorted(shared)))
            hits={w for w in words&wanted-explained if w.startswith('wn_noun:')}
            value=sum(weights[w] for w in sorted(hits))*2.2/(1+1.2*(.25+.75*len(words)/max(1,average)))
            values.append(dict(source_event_id=s['source_event_id'],role='user',score=value,
                matched_terms=sorted(hits),literal_shared_terms=sorted(shared)))
        scores[mid]=max((v['score'] for v in values),default=0.0);matches[mid]=values
    return scores,matches


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wordnet',type=Path,default=Path('data/lexicons/wordnet.zip'))
    p.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'))
    args=p.parse_args();concepts=factorial.NounConcepts(args.wordnet)
    with patch('socket.socket.connect',side_effect=AssertionError('Offline diagnostic')),patch.object(
            factorial,'noun_scores',expanded_noun_scores),patch.object(factorial,'MODES',('question_noun',)):
        score(args,preparer=novel.preparer(concepts),modes=('question_noun',),limitation=__doc__,script_path=Path(__file__))
    result=json.loads(args.output.read_text())
    result['dependencies_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (args.wordnet,Path(factorial.__file__),Path(novel.__file__),Path(__file__).with_name('wordnet_context_diagnostic.py'))}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
