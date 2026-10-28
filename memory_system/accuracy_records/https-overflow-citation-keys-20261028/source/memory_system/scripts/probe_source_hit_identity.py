"""Reconstruct actual lexical hits before source-route diversity allocation.

The current merge groups all linked user messages, even when different messages
caused the hit. Compare grouping only source messages at the recorded per-query
BM25 score. Preserve every per-query candidate, score, cap and eligibility join.
This is an initial-route diagnostic, not live fusion/fine/packet/answer recall.
Gold carrier metadata is read only after all 56 allocations finish.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import config,personal_evidence,schemas,search_pipeline as search,store
from evaluation_model_guard import check_runtime,check_versions


def records(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def matching_sources(st,t,query,lane):
    tokens=sorted(personal_evidence.terms(query))[:40]
    if not tokens:
        if lane:raise ValueError('Nonempty source route for empty query terms')
        return {}
    rank_args=[]
    if re.search(r'[\u3400-\u9fff]{2}',query):
        match='('+' OR '.join('source_fts.content LIKE ?' for _ in tokens)+')'
        match_args=[f'%{token}%' for token in tokens]
        rank='-('+' + '.join('(source_fts.content LIKE ?)' for _ in tokens)+')'
        rank_args=match_args
    else:
        match='source_fts MATCH ?'
        match_args=[' OR '.join('"'+token.replace('"','""')+'"' for token in tokens)]
        rank='bm25(source_fts)'
    scores={c['id']:c['_score'] for c in lane}
    result={mid:[] for mid in scores}
    rows=st.conn.execute(f'''SELECT a.id,s.request_id,s.message_index,s.source_event_id,s.role,s.content,{rank} AS rank
        FROM source_fts
        JOIN source_messages s ON s.request_id=source_fts.request_id AND s.message_index=source_fts.message_index
        JOIN amu_sources l ON l.request_id=s.request_id AND l.message_index=s.message_index
        JOIN amu a ON a.id=l.amu_id
        WHERE {match} AND source_fts.user_id=? AND a.user_id=? AND s.user_id=?
          AND (? OR a.valid_to IS NULL)
          AND (? OR (a.sensitivity!='sensitive' AND s.sensitivity!='sensitive'))
          AND a.sensitivity!='suppressed' AND s.sensitivity!='suppressed'
          AND a.view_status='ready' AND a.resolution_status!='retracted'
        ORDER BY rank,a.id''',(*rank_args,*match_args,t['user_id'],t['user_id'],t['user_id'],
            t['include_history'],t['include_sensitive']))
    # include_cold=True is the actual initial lexical-search policy.
    for r in rows:
        if r['id'] in scores and -float(r['rank'])==scores[r['id']]:
            result[r['id']].append(dict(request_id=r['request_id'],message_index=r['message_index'],
                source_event_id=r['source_event_id'],role=r['role'],content=r['content'],score=-float(r['rank'])))
    if any(not value for value in result.values()):
        raise ValueError('Recorded candidate score lacks eligible original source')
    return result


def main(args):
    check_runtime()
    root=args.run.resolve()
    if args.output.exists():raise FileExistsError(args.output)
    plan=json.loads((root/'plan.json').read_text())
    if (root/'exit-code').read_text().strip()!='0' or sha(root/'frozen-memory.db')!=plan['database_sha256']:
        raise ValueError('Require complete frozen cohort')
    traces=records(root/'search/server-search-debug.jsonl')
    st=store.Store(':memory:')
    with sqlite3.connect((root/'frozen-memory.db').as_uri()+'?mode=ro',uri=True) as src:src.backup(st.conn)
    st.conn.execute('PRAGMA query_only=ON')
    jobs=[]
    try:
        with patch('socket.socket.connect',side_effect=AssertionError('Offline source-hit diagnostic')):
            for t in traces:
                check_versions(t['versions'],'archive Search')
                st.read_user_id=t['user_id']
                req=schemas.SearchRequest(user_id=t['user_id'],query=t['query'],options=t['options'],
                    include_sensitive=t['requested_include_sensitive'])
                query_plan=dict(copy.deepcopy(t['plan']),_include_history=t['include_history'],_include_sensitive=t['include_sensitive'])
                specs=[s for s in t['query_specs'] if s['origin']!='grounded_followup']
                observed=next(r for r in t['routes'] if r['channel']=='source_text' and r['round']==1)
                observed_fts=next(r for r in t['routes'] if r['channel']=='full_text' and r['round']==1)
                capture={}
                original_merge=search._merge_source_query_results
                def capture_merge(routes,limit,source_keys,**kwargs):
                    capture.update(routes=routes,limit=limit,source_keys=source_keys)
                    return original_merge(routes,limit,source_keys,**kwargs)
                with patch.multiple(config,**t['versions']['settings']),patch.object(search,'_merge_source_query_results',capture_merge):
                    if not config.SOURCE_RECALL_DIVERSITY or config.SOURCE_RECALL_NOVELTY:
                        raise ValueError('Require baseline distinct-source-set policy')
                    fts,baseline=search._lexical_recall(st,req,query_plan,specs)
                    if [c['id'] for c in baseline]!=[c['id'] for c in observed['candidates']]:
                        raise ValueError('Historical source route differs')
                    if [c['id'] for c in fts]!=[c['id'] for c in observed_fts['candidates']]:
                        raise ValueError('Historical FTS route differs')
                    lanes=capture['routes'];limit=capture['limit'];linked=capture['source_keys']
                    keys={mid:set() for mid in linked};hits=[]
                    for index,(spec,lane) in enumerate(zip(specs,lanes,strict=True)):
                        matched=matching_sources(st,t,spec['text'],lane)
                        for mid,sources in matched.items():
                            keys[mid].update((s['request_id'],s['message_index']) for s in sources)
                        hits.append(dict(query_index=index,query=spec['text'],candidate_ids=[c['id'] for c in lane],
                            sources=matched))
                    variants=dict(baseline=baseline,
                        actual_hit_sets=original_merge(lanes,limit,keys),
                        actual_hit_novelty=original_merge(lanes,limit,keys,novel_first=True))
                allowed={c['id'] for lane in lanes for c in lane}
                for name,cs in variants.items():
                    ids=[c['id'] for c in cs]
                    if len(ids)>limit or len(ids)!=len(set(ids)) or not set(ids)<=allowed:
                        raise ValueError('Invalid source route proposal')
                prepared={c['id']:c for c in t['ranked']}
                visibility=[]
                for mid in [c['id'] for c in baseline]:
                    actual={}
                    for lane in hits:
                        for s in lane['sources'].get(mid,[]):actual[s['request_id'],s['message_index']]=s
                    for key,s in actual.items():
                        visible=prepared.get(mid,{}).get('_packet_item',{}).get('sources',[])
                        texts=[v.get('content','') for v in visible if (v.get('request_id'),v.get('message_index'))==key]
                        status=('not_prepared' if mid not in prepared else 'full' if s['content'] in texts
                            else 'partial' if any(texts) else 'hidden')
                        visibility.append(dict(candidate_id=mid,source_key=list(key),role=s['role'],status=status))
                jobs.append(dict(search_id=t['search_id'],user_id=t['user_id'],limit=limit,hits=hits,
                    source_sets={mid:dict(linked=sorted(linked[mid]),matched=sorted(keys[mid])) for mid in linked},
                    variants={name:[c['id'] for c in cs] for name,cs in variants.items()},
                    other_route_ids=sorted({c['id'] for route in t['routes'] if route is not observed for c in route['candidates']}),
                    baseline_prepared_hit_visibility=visibility))
    finally:st.conn.close()
    if len(jobs)!=56 or len({j['search_id'] for j in jobs})!=56:raise ValueError('Incomplete cohort')
    # All choices now fixed; reference carrier IDs only enter the scorer below.
    results={r['search_id']:r for r in records(root/'control/completed/data/results/personamem-v2-32k.jsonl')}
    stage=json.loads((root/'control/funnel/stage-transport.json').read_text())
    if stage['summary']['progress_snapshot']:raise ValueError('Incomplete reference audit')
    refs={(r['conversation_id'],r['qa_id']):r for r in stage['rows']}
    for j in jobs:
        r=results[j['search_id']];j['qa_id']=r['qa_id']
        target=set(refs[r['conversation_id'],r['qa_id']]['carrier_ids']['memory'])
        j['reference_carriers']=sorted(target)
        j['reference_hit']={mode:dict(source=bool(set(ids)&target),
            union_other=bool((set(ids)|set(j['other_route_ids']))&target)) for mode,ids in j['variants'].items()}
    summary={}
    for mode in jobs[0]['variants']:
        summary[mode]=dict(changed_routes=sum(j['variants'][mode]!=j['variants']['baseline'] for j in jobs),
            selected_candidates=sum(len(j['variants'][mode]) for j in jobs),
            independent_matched_sources=sum(len({tuple(s) for mid in j['variants'][mode] for s in j['source_sets'][mid]['matched']}) for j in jobs))
        for metric in ('source','union_other'):
            summary[mode][metric]=dict(reference_questions=sum(j['reference_hit'][mode][metric] for j in jobs),
                gained=[j['qa_id'] for j in jobs if j['reference_hit'][mode][metric] and not j['reference_hit']['baseline'][metric]],
                lost=[j['qa_id'] for j in jobs if not j['reference_hit'][mode][metric] and j['reference_hit']['baseline'][metric]])
    result=dict(summary=summary,questions=len(jobs),initial_source_routes_reproduced=len(jobs),fts_routes_reproduced=len(jobs),
        linked_set_differs_from_matching_set=sum(set(map(tuple,s['linked']))!=set(map(tuple,s['matched'])) for j in jobs for s in j['source_sets'].values()),
        source_sets_examined=sum(len(j['source_sets']) for j in jobs),
        baseline_prepared_hit_visibility=dict(Counter(v['status'] for j in jobs for v in j['baseline_prepared_hit_visibility'])),
        rows=jobs,database_sha256=sha(root/'frozen-memory.db'),trace_sha256=sha(root/'search/server-search-debug.jsonl'),
        script_sha256=sha(Path(__file__)),provider_calls=0,limitation=__doc__,
        reference_limitation='Carrier IDs are a linkage proxy; no claim that prepared text, final packets or answers improve. Other routes, including expanded graph routes, remain historical.')
    with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
