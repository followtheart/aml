"""Require the scoring source itself to be a novel complete user message.

Reuse the question/all-target and literal/noun factorial exactly. Candidate
novelty alone can be satisfied by an unrelated source while its score comes
from already-delivered text. Mask those source scores after computing the
unchanged corpus statistics. No novel-term requirement: repeat vocabulary in
a distinct original source remains eligible. Preserve full candidates/budgets.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import probe_question_lexical_context as factorial
from probe_source_rank_residual import main as score
from app.conversation_context import usable
from app.fine_context_reserve import full_user


def identity(source):
    return tuple(source.get(k) for k in
        ('user_id','request_id','message_index','source_event_id','role','content'))


def score_filter(original,known):
    def filtered(corpus,**kwargs):
        scores,details=original(corpus,**kwargs)
        by_id={c['id']:c for c in corpus}
        for mid,matches in details.items():
            sources=[s for s in by_id[mid]['_packet_item']['sources']
                if s.get('role')=='user' and s.get('content')]
            if len(sources)!=len(matches):
                raise ValueError('Source score attribution differs')
            for source,match in zip(sources,matches,strict=True):
                if match['source_event_id']!=source['source_event_id']:
                    raise ValueError('Source identity differs')
                match['novel_complete']=full_user(source) and identity(source) not in known
                match['unfiltered_score']=match['score']
                if not match['novel_complete']:
                    match['score']=0.0
            scores[mid]=max((m['score'] for m in matches),default=0.0)
        return scores,details
    return filtered


def preparer(concepts):
    original=factorial.preparer(concepts)
    literal,noun=factorial.source_scores,factorial.noun_scores
    def prepare(t):
        known={identity(s) for p in t['returned'] for s in p.get('sources',[])
            if usable(s,t['user_id']) and full_user(s)}
        with patch.object(factorial,'source_scores',score_filter(literal,known)),patch.object(
                factorial,'noun_scores',score_filter(noun,known)):
            return original(t)
    return prepare


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wordnet',type=Path,default=Path('data/lexicons/wordnet.zip'))
    p.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'))
    args=p.parse_args();concepts=factorial.NounConcepts(args.wordnet)
    with patch('socket.socket.connect',side_effect=AssertionError('Offline diagnostic')):
        score(args,preparer=preparer(concepts),modes=factorial.MODES,limitation=__doc__,script_path=Path(__file__))
    result=json.loads(args.output.read_text())
    result['dependencies_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (args.wordnet,Path(factorial.__file__),Path(__file__).with_name('wordnet_context_diagnostic.py'))}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
