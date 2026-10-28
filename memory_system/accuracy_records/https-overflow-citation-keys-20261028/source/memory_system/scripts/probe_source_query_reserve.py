"""Offline source-query champions before the existing context reserve.

One scored coarse candidate that missed fine can earn the first context seat
when it was the first source-text hit of an original question or literal option
topic. The accepted packet is immutable; the remaining original CE/lexical
reserve shares the SAME 5 KiB allowance. Source ranking is not fact validation.
"""
import argparse
import ast
import copy
import csv
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, config, evidence_packet, personal_evidence, retrieval_queries, search_pipeline
from app.fine_context_lexical import propose as lexical
from app.fine_context_lexical import source_scores
from app.fine_context_fill import propose as fill
from app.fine_context_reserve import propose as reserve
from evaluation_model_guard import check_versions
from probe_reserve_capacity import records
import audit_https_funnel as audit


def nominees(t, pool, ce_window=False, nearby_ce=False):
    routes = [r for r in t['routes'] if r['channel'] == 'source_text' and r['round'] == 1]
    if len(routes) != 1:
        raise ValueError('Require exactly one initial source route')
    specs = [s for s in t['query_specs'] if s['origin'] != 'grounded_followup']
    direct = {}
    for index, spec in enumerate(specs):
        if spec['origin'] == 'question' and spec['text'] == t['query']:
            direct[str(index)] = spec
        elif spec['origin'] == 'option_topic':
            oi = spec.get('option_index')
            if (type(oi) is not int or not 0 <= oi < len(t['options']) or
                    spec['text'] != retrieval_queries.topic_anchor(t['options'][oi])):
                raise ValueError('Topic is not the exact existing literal option view')
            direct[str(index)] = spec
    allowed = set(t['cascade']['coarse']['selected_ids']) - set(t['cascade']['fine']['selected_ids'])
    selection = t['cascade']['fine']['selection']
    max_rank = selection.get('soft_reservation_policy', {}).get('max_ce_rank')
    ranks = {d['id']: d.get('rank') for d in selection['decisions']}
    if (ce_window or nearby_ce) and (type(max_rank) is not int or max_rank < 1):
        raise ValueError('Require a recorded positive soft-reservation CE rank bound')
    if nearby_ce:
        proposals = [dict(candidate_id=mid, ce_rank=rank) for mid,rank in ranks.items()
                     if mid in allowed and mid in pool and mid not in t['selection']['selected_ids']
                     and type(rank) is int and 1 <= rank <= max_rank]
        proposals.sort(key=lambda p:p['ce_rank'])
        return [pool[p['candidate_id']] for p in proposals], dict(
            policy='nearby_ce_outsider', direct_queries={}, nominations=proposals,
            max_ce_rank=max_rank, rejected=[])
    proposals = []
    rejected = []
    for position, c in enumerate(routes[0]['candidates']):
        mid = c['id']
        if mid not in allowed or mid not in pool or mid in t['selection']['selected_ids']:
            continue
        hits = [key for key, rank in c.get('_query_ranks', {}).items() if rank == 1 and key in direct]
        if hits:
            entry = dict(candidate_id=mid, query_indices=sorted(hits, key=int), route_rank=position+1,
                         ce_rank=ranks.get(mid))
            if ce_window and (type(ranks.get(mid)) is not int or not 1 <= ranks[mid] <= max_rank):
                rejected.append(dict(entry, reason='outside_recorded_ce_window'))
            else:
                proposals.append(entry)
    proposals.sort(key=lambda p: (-len(p['query_indices']), p['route_rank']))
    return [pool[p['candidate_id']] for p in proposals], dict(direct_queries=direct, nominations=proposals,
        ce_window=ce_window, max_ce_rank=max_rank if ce_window else None, rejected=rejected)


def proposal(base, fine, champions, **kwargs):
    first, trace = reserve(base, champions, user_id=kwargs['user_id'], top_k=kwargs['top_k'],
                           byte_budget=kwargs['byte_budget'], reserve_bytes=kwargs['reserve_bytes'], limit=1)
    for item in first[len(base):]:
        item['score_kind'] = 'source_query_context'
    revised, rest = lexical(first, fine, **dict(kwargs, reserve_bytes=kwargs['reserve_bytes']-trace['added_bytes']))
    assert revised[:len(base)] == base
    return revised, dict(source_seat=trace, remainder=rest, added_bytes=trace['added_bytes']+rest['added_bytes'])


def sentence_scores(corpus, *, user_id, query, options):
    # Reuse eligibility checks, but score intact sentence views without clipping
    # the complete source unit delivered by reserve(). No generated text.
    eligible,_=source_scores(corpus,user_id=user_id,query=query,options=options)
    documents={};members={}
    for c in corpus:
        if c['id'] not in eligible:continue
        keys=[]
        for s in c['_packet_item']['sources']:
            if s.get('role')!='user':continue
            for match in personal_evidence._sentences(s['content']):
                key=(s['source_event_id'],s['request_id'],s['message_index'],match.start(),match.group())
                words=personal_evidence.terms(match.group())
                if words:documents[key]=words;keys.append(key)
        members[c['id']]=keys
    n=len(documents);avg=sum(map(len,documents.values()))/max(1,n)
    frequency=Counter(term for words in documents.values() for term in words)
    weights={term:math.log(1+(n-count+.5)/(count+.5)) for term,count in frequency.items()}
    wanted=personal_evidence.terms(' '.join([query,*options]))
    scores={};details={}
    for mid,keys in members.items():
        values=[]
        for key in keys:
            words=documents[key];hits=words&wanted
            score=sum(weights[t] for t in sorted(hits))*2.2/(1+1.2*(.25+.75*len(words)/max(1,avg)))
            values.append(dict(source_event_id=key[0],start=key[3],text=key[4],matched_terms=sorted(hits),score=score))
        scores[mid]=max((v['score'] for v in values),default=0)
        details[mid]=values
    return scores,details


def expanded_lexical(base, fine, champions, *, sentence=False, noun_concepts=None, **kwargs):
    """Keep both CE seats; let nearby coarse outsiders compete for the third."""
    limits={k:kwargs[k] for k in ('user_id','top_k','byte_budget','reserve_bytes')}
    first, prefix=reserve(base,fine,**limits)
    corpus=list(fine)+list(champions)
    scorer=sentence_scores if sentence else source_scores
    if noun_concepts is not None:
        from wordnet_context_diagnostic import scores as noun_scores
        scores,matches=noun_scores(corpus,user_id=kwargs['user_id'],query=kwargs['query'],options=kwargs['options'],concepts=noun_concepts)
    else:
        scores,matches=scorer(corpus,user_id=kwargs['user_id'],query=kwargs['query'],options=kwargs['options'])
    ordered=sorted([c for c in corpus if scores.get(c['id'],0)>0],key=lambda c:-scores[c['id']])
    result, extra=reserve(first,ordered,**dict(limits,reserve_bytes=limits['reserve_bytes']-prefix['added_bytes']),limit=1)
    assert result[:len(first)]==first
    return result,dict(source_seat=extra,remainder=prefix,added_bytes=prefix['added_bytes']+extra['added_bytes'],
                      scores=scores,matches=matches,ordered_ids=[c['id'] for c in ordered])


def lexical_residual(base, fine, champions, *, noun_concepts=None, **kwargs):
    """Keep all three seats, add at most one user-lexical source in spare bytes."""
    first,prefix=lexical(base,fine,**kwargs)
    if noun_concepts is not None:
        from wordnet_context_diagnostic import scores as noun_scores
        scores,matches=noun_scores(fine,user_id=kwargs['user_id'],query=kwargs['query'],options=kwargs['options'],
                                  concepts=noun_concepts,novel_from=first)
    else:
        scores,matches=source_scores(fine,user_id=kwargs['user_id'],query=kwargs['query'],options=kwargs['options'])
    ordered=sorted([c for c in fine if scores.get(c['id'],0)>0],key=lambda c:-scores[c['id']])
    limits={k:kwargs[k] for k in ('user_id','top_k','byte_budget','reserve_bytes')}
    result,extra=reserve(first,ordered,**dict(limits,reserve_bytes=limits['reserve_bytes']-prefix['added_bytes']),limit=1)
    assert result[:len(first)]==first
    return result,dict(source_seat=extra,remainder=prefix,added_bytes=prefix['added_bytes']+extra['added_bytes'],
                      scores=scores,matches=matches,ordered_ids=[c['id'] for c in ordered])


def main(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    csv.field_size_limit(32*1024*1024)
    with a.gold_csv.open() as f:
        gold = list(csv.DictReader(f))
    rows = []
    noun_concepts=None
    if a.wordnet:
        from wordnet_context_diagnostic import NounConcepts
        noun_concepts=NounConcepts(a.wordnet)
    for run in a.runs:
        if (run/'exit-code').read_text().strip() != '0':
            raise ValueError('Require completed successful arm')
        results = {r['search_id']: r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
        raw = {}
        for event in records(run/'server-memory-debug.jsonl'):
            raw.setdefault(event['user_id'], {}).update({audit.identity(s): s for s in event['source_messages']})
        for t in records(run/'server-search-debug.jsonl'):
            check_versions(t['versions'], 'source trace')
            assert evidence_packet.digest(t['returned']) == t['packet_hash']
            ctx=t['cascade']['conversation_context']
            base=[dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
            assert evidence_packet.digest(base)==ctx['base_packet_hash']
            with patch.multiple(config, **t['versions']['settings']), patch('socket.socket.connect', side_effect=AssertionError('Offline')):
                pool={c['id']:c for c in search_pipeline._foresight_filter(t['ranked'],copy.deepcopy(t['plan']))
                      if c.get('view_status')!='stale' and '_packet_item' in c}
                fine=[pool[mid] for mid in t['cascade']['fine']['selected_ids']
                      if mid in pool and mid not in t['selection']['selected_ids']]
                kw=dict(user_id=t['user_id'], top_k=t['top_k'], byte_budget=t['coverage_manifest']['token_budget'],
                        reserve_bytes=5120, query=t['query'], options=t['options'])
                baseline, old=lexical(base, fine, **kw)
                champions, nomination=nominees(t,pool,a.ce_window,a.nearby_ce or a.nearby_user_lexical or a.nearby_sentence)
                if noun_concepts is not None:
                    champions=[];nomination=dict(nominations=[],policy='wordnet_first_noun_fine_only')
                proposer=lexical_residual if a.lexical_residual or a.wordnet_residual_novel else expanded_lexical if a.nearby_user_lexical or a.nearby_sentence or noun_concepts is not None else proposal
                candidate, proposed=proposer(base,fine,champions,**kw,**({'sentence':True} if a.nearby_sentence else {}),
                    **({'noun_concepts':noun_concepts} if noun_concepts is not None else {}))
                if config.PACKET_FINE_CONTEXT_FILL:
                    actual,_=fill(base,fine,**kw)
                else:
                    actual=baseline
                assert [dict(p,packet_hash=evidence_packet.digest(actual)) for p in actual]==t['returned']
            qa=results[t['search_id']]['qa_id'];snippet=gold[int(qa)]['related_conversation_snippet']
            try:
                snippet=json.loads(snippet)
            except ValueError:
                snippet=ast.literal_eval(snippet)
            scores={};catalog_bytes={}
            for mode, packet in (('baseline',baseline),('candidate',candidate)):
                cards=[dict(s,text=s.get('content',''),id=f'{i}:{j}') for i,p in enumerate(packet) for j,s in enumerate(p['sources'])]
                scores[mode]={role:[v['full'] for v in audit.turn_coverage(
                    [audit.norm(s['content']) for s in snippet if s['role']==role],cards,raw[t['user_id']],role=role)]
                    for role in ('user','assistant')}
                digest=evidence_packet.digest(packet)
                catalogs=answer_choice.build_catalog([dict(p,packet_hash=digest) for p in packet])
                catalog_bytes[mode]=sum(len(c['text'].encode()) for cat in catalogs for c in cat.values())
            rows.append(dict(run=str(run),qa_id=qa,scores=scores,nomination=nomination,proposal=proposed,
                             baseline=old,catalog_bytes=catalog_bytes))
    summaries=[]
    for run in a.runs:
        rs=[r for r in rows if r['run']==str(run)]
        summaries.append(dict(run=str(run),questions=len(rs),actual_packets_verified=len(rs),
            nominated_questions=sum(bool(r['nomination']['nominations']) for r in rs),
            source_seat_questions=sum(bool(r['proposal']['source_seat']['appended_ids']) for r in rs),
            outsider_admitted_questions=[r['qa_id'] for r in rs if set(r['proposal']['source_seat']['appended_ids']) &
                {p['candidate_id'] for p in r['nomination']['nominations']}],
            references={m:sum(any(r['scores'][m]['user']) for r in rs) for m in ('baseline','candidate')},
            user_turns={m:sum(sum(r['scores'][m]['user']) for r in rs) for m in ('baseline','candidate')},
            catalog_bytes={m:sum(r['catalog_bytes'][m] for r in rs) for m in ('baseline','candidate')},
            gained=[r['qa_id'] for r in rs if any(r['scores']['candidate']['user']) and not any(r['scores']['baseline']['user'])],
            lost=[r['qa_id'] for r in rs if any(r['scores']['baseline']['user']) and not any(r['scores']['candidate']['user'])],
            turn_regressions=[r['qa_id'] for r in rs if any(x and not y for role in ('user','assistant')
                for x,y in zip(r['scores']['baseline'][role],r['scores']['candidate'][role]))]))
    a.output.write_text(json.dumps(dict(summary=summaries,rows=rows,limitation=__doc__,
        nomination_policy='wordnet_novel_noun_residual' if a.wordnet_residual_novel else
                          'wordnet_first_noun_fine_only' if a.wordnet else
                          'one_lexical_residual' if a.lexical_residual else
                          'nearby_user_sentence_third_seat' if a.nearby_sentence else
                          'nearby_user_lexical_third_seat' if a.nearby_user_lexical else
                          'nearby_ce_outsider' if a.nearby_ce else 'source_query_first_hit',
        lexical_resource_sha256=hashlib.sha256(a.wordnet.read_bytes()).hexdigest() if a.wordnet else None,
        lexical_helper_sha256=hashlib.sha256(Path(__file__).with_name('wordnet_context_diagnostic.py').read_bytes()).hexdigest() if a.wordnet else None,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ce-window',action='store_true',help='Reuse the recorded fine soft-reservation CE rank window')
    p.add_argument('--nearby-ce',action='store_true',help='Diagnostic: prioritize one coarse outsider by existing CE rank window alone')
    p.add_argument('--nearby-user-lexical',action='store_true',help='Diagnostic: add nearby coarse outsiders to the existing user-lexical third-seat competition')
    p.add_argument('--nearby-sentence',action='store_true',help='Diagnostic: score intact user sentences, deliver whole source units')
    p.add_argument('--lexical-residual',action='store_true',help='Diagnostic: preserve all three seats, add at most one extra user-lexical source')
    p.add_argument('--wordnet',type=Path,help='Diagnostic: normalize user-source nouns to WordNet first-sense concepts in third-seat scoring')
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    p.add_argument('--wordnet-residual-novel',action='store_true',help='Diagnostic: one extra source for noun concepts absent from the unchanged lexical packet')
    args=p.parse_args()
    if args.wordnet_residual_novel and not args.wordnet:p.error('--wordnet-residual-novel requires --wordnet')
    main(args)
