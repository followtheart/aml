"""Offline source-only lexical ordering for bounded reserve seats.

The second-seat mode keeps the first CE seat; extra modes keep both original
seats. The optional coarse pool changes only the extra seat's suppliers.
Score visible original source messages with
binary-term BM25 (k1=1.2, b=.75), using question/options as retrieval terms.
Source identity is deduplicated for IDF; summaries and gold never enter ranking.
Topic matching is not entailment and does not change any source or fact label.
"""
import argparse
import ast
from collections import Counter
import copy
import csv
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import config,evidence_packet,personal_evidence,search_pipeline
from app.conversation_context import source_key,usable
from app.fine_context_reserve import propose
import audit_https_funnel as audit
from evaluation_model_guard import check_versions


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def source_scores(fine,*,user_id,query,options,user_only=False,novel_terms_from=None):
    corpus={};members={}
    for candidate in fine:
        item=candidate.get('_packet_item',{})
        sources=[s for s in item.get('sources',[]) if s.get('content')]
        score=candidate.get('_ce_score')
        if (candidate.get('view_status')=='stale' or item.get('is_constraint')
                or type(score) not in (int,float) or not math.isfinite(score)
                or not sources or any(not usable(s,user_id) or s['role'] not in ('user','assistant')
                    or s['content'] not in item.get('content','') for s in sources)):
            continue
        mids=[]
        for s in sources:
            if user_only and s['role']!='user':continue
            key=(s['user_id'],s['request_id'],s['message_index'],s['source_event_id'],s['role'])
            corpus.setdefault(key,set()).update(personal_evidence.terms(s['content']))
            mids.append(key)
        members[candidate['id']]=mids
    query_terms=personal_evidence.terms(' '.join([query,*options]))
    frequency=Counter(term for words in corpus.values() for term in words)
    n=len(corpus);average=sum(map(len,corpus.values()))/max(1,n)
    if novel_terms_from is not None:
        covered=set().union(*(personal_evidence.terms(s.get('content',''))
            for p in novel_terms_from for s in p.get('sources',[]) if s.get('role')=='user'))
        query_terms={t for t in query_terms-covered if len(t)>=4 and frequency[t]<=max(1,n//10)}
    weights={term:math.log(1+(n-count+.5)/(count+.5)) for term,count in frequency.items()}
    scores={};details={}
    for candidate in fine:
        item=candidate.get('_packet_item',{})
        values=[]
        if candidate['id'] not in members:continue
        for s in item.get('sources',[]):
            if not s.get('content'):continue
            if user_only and s['role']!='user':continue
            words=personal_evidence.terms(s['content'])
            hits=words&query_terms
            value=sum(weights[t] for t in sorted(hits))*2.2/(1+1.2*(.25+.75*len(words)/max(1,average)))
            values.append(dict(source_event_id=s['source_event_id'],role=s['role'],
                               score=value,matched_terms=sorted(hits)))
        scores[candidate['id']]=max((v['score'] for v in values),default=0)
        details[candidate['id']]=values
    return scores,details


def proposal(base,fine,*,user_id,top_k,byte_budget,query,options,reserve_bytes=5120,mode='second',extra_pool=None):
    extra=mode.startswith('extra_')
    first,first_trace=propose(base,fine,user_id=user_id,top_k=top_k,byte_budget=byte_budget,
                              reserve_bytes=reserve_bytes,limit=2 if extra else 1)
    suppliers=fine if extra_pool is None else extra_pool
    scores,details=source_scores(suppliers,user_id=user_id,query=query,options=options,user_only=mode.startswith('extra_user'),
                                novel_terms_from=first if mode=='extra_user_novel' else None)
    ordered=sorted([c for c in suppliers if not extra or scores.get(c['id'],0)>0],
                   key=lambda c:-scores.get(c['id'],0))
    result,second=propose(first,ordered,user_id=user_id,top_k=top_k,byte_budget=byte_budget,
        reserve_bytes=reserve_bytes-first_trace['added_bytes'],limit=1)
    return result,dict(appended_ids=first_trace['appended_ids']+second['appended_ids'],
        added_bytes=first_trace['added_bytes']+second['added_bytes'],first=first_trace,second=second,
        source_scores=scores,source_matches=details)


def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:gold=list(csv.DictReader(f))
    rows=[]
    for run in args.runs:
        if (run/'exit-code').read_text().strip()!='0':raise ValueError('Incomplete run')
        rs={r['search_id']:r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
        raw={}
        for e in records(run/'server-memory-debug.jsonl'):
            raw.setdefault(e['user_id'],{}).update({audit.identity(s):s for s in e['source_messages']})
        for t in records(run/'server-search-debug.jsonl'):
            check_versions(t['versions'],'source trace')
            if evidence_packet.digest(t['returned'])!=t['packet_hash']:raise ValueError('Final packet mismatch')
            ctx=t['cascade']['conversation_context']
            base=[dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
            if evidence_packet.digest(base)!=ctx['base_packet_hash']:raise ValueError('Base packet mismatch')
            with patch.multiple(config,**t['versions']['settings']):
                pool={c['id']:c for c in search_pipeline._foresight_filter(t['ranked'],copy.deepcopy(t['plan']))
                      if c.get('view_status')!='stale' and '_packet_item' in c}
                fine=[pool[mid] for mid in t['cascade']['fine']['selected_ids']
                      if mid in pool and mid not in t['selection']['selected_ids']]
                kwargs=dict(user_id=t['user_id'],top_k=t['top_k'],byte_budget=t['coverage_manifest']['token_budget'],reserve_bytes=5120)
                original,old=propose(base,fine,**kwargs)
                suppliers=([pool[mid] for mid in t['cascade']['coarse']['selected_ids']
                    if mid in pool and mid not in t['selection']['selected_ids']] if args.pool=='coarse' else None)
                revised,new=proposal(base,fine,query=t['query'],options=t['options'],mode=args.mode,extra_pool=suppliers,**kwargs)
            if revised[:len(base)]!=base:raise ValueError('Original evidence changed')
            if original[len(base):len(base)+1]!=revised[len(base):len(base)+1]:raise ValueError('First seat changed')
            if args.mode.startswith('extra_') and revised[:len(original)]!=original:raise ValueError('Original reserve displaced')
            comparisons=[('original',original),('source_order',revised)]
            lexical_trace=None
            if args.mode=='extra_all_roles':
                with patch.multiple(config,**t['versions']['settings']):
                    lexical,lexical_trace=proposal(base,fine,query=t['query'],options=t['options'],mode='extra_user',extra_pool=suppliers,**kwargs)
                comparisons.append(('user_lexical',lexical))
            qa=rs[t['search_id']]['qa_id'];snippet=gold[int(qa)]['related_conversation_snippet']
            try:snippet=json.loads(snippet)
            except ValueError:snippet=ast.literal_eval(snippet)
            scores={}
            for mode,packet in comparisons:
                cards=[dict(s,text=s.get('content',''),id=f'{i}:{j}') for i,p in enumerate(packet) for j,s in enumerate(p['sources'])]
                scores[mode]={role:[v['full'] for v in audit.turn_coverage(
                    [audit.norm(s['content']) for s in snippet if s['role']==role],cards,raw[t['user_id']],role=role)]
                    for role in ('user','assistant')}
            rows.append(dict(run=str(run),qa_id=qa,scores=scores,original=old,proposal=new,user_lexical=lexical_trace))
    summaries=[]
    for run in args.runs:
        selected=[r for r in rows if r['run']==str(run)]
        summaries.append(dict(run=str(run),questions=len(selected),
            reference={m:sum(any(r['scores'][m]['user']) for r in selected) for m in ('original','source_order')},
            user_turns={m:sum(sum(r['scores'][m]['user']) for r in selected) for m in ('original','source_order')},
            assistant_turns={m:sum(sum(r['scores'][m]['assistant']) for r in selected) for m in ('original','source_order')},
            gained=[r['qa_id'] for r in selected if any(r['scores']['source_order']['user']) and not any(r['scores']['original']['user'])],
            lost=[r['qa_id'] for r in selected if any(r['scores']['original']['user']) and not any(r['scores']['source_order']['user'])],
            turn_regressions=[r['qa_id'] for r in selected if any(a and not b for role in ('user','assistant')
                for a,b in zip(r['scores']['original'][role],r['scores']['source_order'][role]))],
            added_bytes={k:sum(r[k]['added_bytes'] for r in selected) for k in ('original','proposal')}))
    if args.mode=='extra_all_roles':
        for summary in summaries:
            selected=[r for r in rows if r['run']==summary['run']]
            summary['versus_user_lexical']=dict(
                before_reference=sum(any(r['scores']['user_lexical']['user']) for r in selected),
                after_reference=sum(any(r['scores']['source_order']['user']) for r in selected),
                gained=[r['qa_id'] for r in selected if any(r['scores']['source_order']['user']) and not any(r['scores']['user_lexical']['user'])],
                lost=[r['qa_id'] for r in selected if any(r['scores']['user_lexical']['user']) and not any(r['scores']['source_order']['user'])],
                turn_regressions=[r['qa_id'] for r in selected if any(a and not b for role in ('user','assistant')
                    for a,b in zip(r['scores']['user_lexical'][role],r['scores']['source_order'][role]))],
                user_lexical_bytes=sum(r['user_lexical']['added_bytes'] for r in selected))
    args.output.write_text(json.dumps(dict(mode=args.mode,pool=args.pool,summary=summaries,rows=rows,limitation=__doc__),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['second','extra_user','extra_user_novel','extra_all_roles'],default='second')
    p.add_argument('--pool',choices=['fine','coarse'],default='fine',help='Extra seat suppliers only; original two seats always use fine')
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
