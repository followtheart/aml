"""Offline: one residual whole-source coarse outsider by existing query ranks.

All three original lexical seats remain intact; the original 5 KiB cap applies.
Compare all existing initial queries with literal question/topic queries. Query
rank ordering is a retrieval signal, never a personal-fact or preference label.
Gold is loaded only after all candidate choices are fixed, with network blocked.
"""
import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice,config,evidence_packet,retrieval_queries,search_pipeline
from app.fine_context_lexical import propose as lexical
from app.fine_context_reserve import propose as reserve
from evaluation_model_guard import check_runtime,check_versions
import audit_https_funnel as audit

MODES=('all_reciprocal_rank','literal_reciprocal_rank','literal_best_rank','question_first','literal_first')


def records(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def prepare(t):
    ctx=t['cascade']['conversation_context']
    base=[dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
    assert evidence_packet.digest(base)==ctx['base_packet_hash']
    pool={c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
        if c.get('view_status')!='stale' and '_packet_item' in c}
    fine=[pool[mid] for mid in t['cascade']['fine']['selected_ids'] if mid in pool and mid not in t['selection']['selected_ids']]
    kwargs=dict(user_id=t['user_id'],top_k=t['top_k'],byte_budget=t['coverage_manifest']['token_budget'])
    prefix,old=lexical(base,fine,**kwargs,reserve_bytes=5120,query=t['query'],options=t['options'])
    digest=evidence_packet.digest(prefix)
    assert [dict(p,packet_hash=digest) for p in prefix]==t['returned']
    remaining=5120-old['added_bytes']
    initial=[s for s in t['query_specs'] if s['origin']!='grounded_followup']
    literal=set();questions=set()
    for i,s in enumerate(initial):
        if s['origin']=='question' and s['text']==t['query']:
            literal.add(str(i));questions.add(str(i))
        elif s['origin']=='option_topic':
            oi=s.get('option_index')
            if type(oi) is not int or not 0<=oi<len(t['options']) or s['text']!=retrieval_queries.topic_anchor(t['options'][oi]):
                raise ValueError('Unexpected topic identity')
            literal.add(str(i))
    routes=[r for r in t['routes'] if r['channel']=='source_text' and r['round']==1]
    if len(routes)!=1:raise ValueError('Require one original source route')
    allowed=set(t['cascade']['coarse']['selected_ids'])-set(t['cascade']['fine']['selected_ids'])-set(t['selection']['selected_ids'])
    eligible=[];inventory=[]
    for pos,hit in enumerate(routes[0]['candidates']):
        mid=hit['id']
        if mid not in allowed or mid not in pool:continue
        packet,trial=reserve(prefix,[pool[mid]],**kwargs,reserve_bytes=remaining,limit=1)
        if not trial['appended_ids']:continue
        ranks=hit.get('_query_ranks',{})
        if any(not key.isdigit() or not 0<=int(key)<len(initial) or type(value) is not int or value<1 for key,value in ranks.items()):
            raise ValueError('Invalid source-query ranks')
        all_score=sum(1/r for r in ranks.values())
        literal_ranks=[r for key,r in ranks.items() if key in literal]
        scores=dict(all_reciprocal_rank=all_score,literal_reciprocal_rank=sum(1/r for r in literal_ranks),
            literal_best_rank=1/min(literal_ranks) if literal_ranks else 0,
            question_first=sum(ranks.get(key)==1 for key in questions),
            literal_first=sum(ranks.get(key)==1 for key in literal))
        eligible.append(pool[mid]);inventory.append(dict(candidate_id=mid,source_route_rank=pos+1,
            ce_score=pool[mid]['_ce_score'],query_ranks=ranks,scores=scores,cost=trial['added_bytes']))
    by_id={r['candidate_id']:r for r in inventory}
    packets={'baseline':prefix};proposals={}
    for mode in MODES:
        ordered=sorted([c for c in eligible if by_id[c['id']]['scores'][mode]>0],
            key=lambda c:-by_id[c['id']]['scores'][mode])
        packets[mode],proposals[mode]=reserve(prefix,ordered,**kwargs,reserve_bytes=remaining,limit=1)
        assert packets[mode][:len(prefix)]==prefix
    return dict(packets=packets,proposals=proposals,inventory=inventory,remaining_bytes=remaining)


def main(args, *, preparer=prepare, modes=MODES, limitation=__doc__, script_path=Path(__file__)):
    check_runtime()
    if args.output.exists():raise FileExistsError(args.output)
    jobs=[];inputs=[]
    for run in args.runs:
        if (run/'exit-code').read_text().strip()!='0':raise ValueError('Incomplete run')
        results={r['search_id']:r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
        inputs.append(dict(run=str(run),trace_sha256=hashlib.sha256((run/'server-search-debug.jsonl').read_bytes()).hexdigest()))
        for t in records(run/'server-search-debug.jsonl'):
            check_versions(t['versions'],'service trace')
            with patch.multiple(config,**t['versions']['settings']):
                if config.PACKET_FINE_CONTEXT_FILL or config.PACKET_SOURCE_QUERY_CONTEXT or config.PACKET_QUESTION_CONTEXT_CE:
                    raise ValueError('Require original lexical control')
                jobs.append(dict(run=str(run),qa_id=results[t['search_id']]['qa_id'],user_id=t['user_id'],**preparer(t)))
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:gold=list(csv.DictReader(f))
    rows=[]
    for run in args.runs:
        raw={}
        for event in records(run/'server-memory-debug.jsonl'):
            raw.setdefault(event['user_id'],{}).update({audit.identity(s):s for s in event['source_messages']})
        for j in jobs:
            if j['run']!=str(run):continue
            snippet=gold[int(j['qa_id'])]['related_conversation_snippet']
            try:snippet=json.loads(snippet)
            except ValueError:snippet=ast.literal_eval(snippet)
            coverage={};catalog_bytes={}
            for mode,packet in j['packets'].items():
                cards=[dict(s,text=s.get('content',''),id=f'{i}:{k}') for i,p in enumerate(packet) for k,s in enumerate(p['sources'])]
                coverage[mode]={role:[v['full'] for v in audit.turn_coverage(
                    [audit.norm(s['content']) for s in snippet if s['role']==role],cards,raw[j['user_id']],role=role)]
                    for role in ('user','assistant')}
                catalogs=answer_choice.build_catalog([dict(p,packet_hash=evidence_packet.digest(packet)) for p in packet])
                catalog_bytes[mode]=sum(len(c['text'].encode()) for cat in catalogs for c in cat.values())
            rows.append(dict(run=str(run),qa_id=j['qa_id'],inventory=j['inventory'],remaining_bytes=j['remaining_bytes'],
                coverage=coverage,catalog_bytes=catalog_bytes,proposals=j['proposals']))
    summary=[]
    for run in args.runs:
        rs=[r for r in rows if r['run']==str(run)]
        for mode in modes:
            summary.append(dict(run=str(run),mode=mode,questions=len(rs),actual_packets_verified=len(rs),
                appended_questions=sum(bool(r['proposals'][mode]['appended_ids']) for r in rs),
                reference_questions={m:sum(any(r['coverage'][m]['user']) for r in rs) for m in ('baseline',mode)},
                user_turns={m:sum(sum(r['coverage'][m]['user']) for r in rs) for m in ('baseline',mode)},
                catalog_bytes={m:sum(r['catalog_bytes'][m] for r in rs) for m in ('baseline',mode)},
                gained=[r['qa_id'] for r in rs if any(r['coverage'][mode]['user']) and not any(r['coverage']['baseline']['user'])],
                lost=[r['qa_id'] for r in rs if any(r['coverage']['baseline']['user']) and not any(r['coverage'][mode]['user'])],
                turn_regressions=[r['qa_id'] for r in rs if any(x and not y for role in ('user','assistant')
                    for x,y in zip(r['coverage']['baseline'][role],r['coverage'][mode][role]))]))
    args.output.write_text(json.dumps(dict(summary=summary,rows=rows,inputs=inputs,limitation=limitation,
        script_sha256=hashlib.sha256(script_path.read_bytes()).hexdigest(),
        common_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    with patch('socket.socket.connect',side_effect=AssertionError('Offline diagnostic')):
        main(p.parse_args())
