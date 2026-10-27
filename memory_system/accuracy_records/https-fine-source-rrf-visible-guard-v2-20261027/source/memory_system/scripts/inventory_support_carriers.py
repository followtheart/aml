"""Observe inherited-source matches and whole-request cost, without selection changes.

Primary source ownership comes from the frozen database, not summary keywords.
Lexical matches are diagnostic, not semantic relevance or factual entailment.
Other-candidate source containment is not permission to remove a primary claim.
Marginal request sizes are measurements, not a counterfactual admission result.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

from audit_retained_porter_https import require, same, selection_code
from probe_assistant_admission_order import replay
from inventory_visible_source_redundancy import records


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(4*1024*1024):digest.update(block)
    return digest.hexdigest()


def batches(cursor):
    while batch:=cursor.fetchmany(256):yield from batch


def identity(source):
    return source.get('request_id'),source.get('message_index')


def primary_keys(candidate, links, sessions):
    if candidate.get('type')=='session_summary':
        return sessions.get((candidate['user_id'],candidate.get('session_id')),set())
    return set().union(*(links.get(mid,set()) for mid in
        [candidate['id'],*candidate.get('_equivalent_ids',[])]))


def observation(candidate, main, keys, facets, fine, context, render):
    unit=candidate['_packet_item'];body=unit.get('_evidence_body')
    visible=[s for s in unit.get('sources',[]) if s.get('content')]
    original=candidate.get('_rank_text',candidate['content'])
    authenticated=isinstance(body,str) and context.with_evidence(body,unit['sources'])==original
    primary=[s for s in visible if identity(s) in keys]
    inherited=[s for s in visible if identity(s) not in keys]
    def matches(sources):
        subset=dict(candidate,_packet_item=dict(unit,sources=sources))
        return sorted(fine.relevance(subset,facets))
    primary_matches=matches(primary);inherited_matches=matches(inherited)
    others=[c for c in main if c['id']!=candidate['id']]
    coverage=[]
    for source in primary:
        owners=[]
        for c in others:
            if any(identity(s)==identity(source) and s.get('role')==source.get('role')
                   and s.get('timestamp')==source.get('timestamp')
                   and source['content'] in s.get('content','') for s in c['_packet_item']['sources']):
                owners.append(c['id'])
        coverage.append(dict(source_hash=hashlib.sha256(json.dumps(identity(source)).encode()).hexdigest(),
                             content_bytes=len(source['content'].encode()),owners=owners))
    return dict(candidate_id=candidate['id'],epistemic_status=candidate.get('epistemic_status'),
        ce_score=candidate.get('_ce_score'),authenticated_full_view=authenticated,
        bridge_ids=candidate.get('_bridge_ids',[]),relation_path_ids=candidate.get('_relation_path_ids',[]),
        protected=bool(candidate.get('_user_rule') or unit.get('is_constraint') or candidate.get('type')=='rule'),
        primary_ownership_known=bool(keys),primary_source_count=len(primary),inherited_source_count=len(inherited),
        primary_visible_bytes=sum(len(context.with_evidence('',[s]).encode()) for s in primary),
        inherited_visible_bytes=sum(len(context.with_evidence('',[s]).encode()) for s in inherited),
        body_bytes=len((body or '').encode()),primary_matches=primary_matches,inherited_matches=inherited_matches,
        matches_only_inherited=bool(keys) and bool(inherited_matches) and not primary_matches,
        complete_primary_visible_in_other_main=bool(primary) and all(c['owners'] for c in coverage),
        primary_source_coverage=coverage,
        actual_prompt_bytes_removed=render(main)-render(others))


def main(args):
    run=args.run.resolve();root=run.parent
    plan=json.loads((root/'plan.json').read_text())
    audited=json.loads((root/'retained-selection-audit.json').read_text())
    same(audited['status'],'passed','Terminal selection audit')
    same(sha(root/'plan.json'),audited['plan_sha256'],'Frozen plan')
    for name,value in plan['source_files'].items():same(sha(root/'source'/name),value,'Frozen source')
    paths=[run/name for name in ('server-search-debug.jsonl','funnel/stage-transport.json',
        'isolated-manifest.json','completed/data/results/personamem-v2-32k.jsonl')]
    for path in paths:same(sha(path),audited['input_sha256'][str(path)],'Audited input')
    manifest=json.loads(paths[2].read_text())
    require(manifest['client_exit_code']==manifest['audit_exit_code']==0 and manifest['server_exit_code'] in (0,-15),'Terminal cohort')
    sys.path.insert(0,str(root/'source/memory_system'))
    from app import answer_context,cascade_rerank,config,fine_evidence,schemas
    modules=cascade_rerank,config,fine_evidence,schemas
    code,code_sha=selection_code(cascade_rerank)
    same(code_sha,audited['selection_ast_sha256'],'Exact frozen selector')
    links={};sessions={};raw={}
    with sqlite3.connect((run/'memory.db').as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        for row in batches(db.execute('SELECT amu_id,request_id,message_index FROM amu_sources')):
            links.setdefault(row['amu_id'],set()).add((row['request_id'],row['message_index']))
        for row in batches(db.execute('SELECT user_id,session_id,request_id,message_index,role,content,timestamp FROM source_messages')):
            source=dict(row);key=identity(source);raw[key]=source
            sessions.setdefault((source['user_id'],source['session_id']),set()).add(key)
    results={r['search_id']:{k:r[k] for k in ('qa_id','conversation_id','user_id','packet_hash')} for r in records(paths[3])}
    same(len(results),plan['questions'],'Full cohort')
    rows=[];seen=set()
    with patch('socket.socket.connect',side_effect=AssertionError('Offline inventory')):
        for t in records(paths[0]):
            sid=t['search_id'];require(sid not in seen and sid in results,'Trace identity');seen.add(sid)
            result=results[sid]
            same((t['user_id'],t['packet_hash']),(result['user_id'],result['packet_hash']),'Packet identity')
            with patch.multiple(config,**t['versions']['settings']):
                baseline,_,size=replay(t,modules,code)
                same(baseline['trace']['fine'],t['cascade']['fine'],'Full fine trace')
                same(baseline['trace'].get('adaptive_admission'),t['cascade'].get('adaptive_admission'),'Adaptive decisions')
                main_rows=baseline['submitted']
                same([c['id'] for c in main_rows],t['cascade']['listwise']['candidate_ids'],'Main order')
                same(size,t['cascade']['listwise']['input_bytes'],'Main bytes')
                for candidate in baseline['fine']:
                    for source in candidate['_packet_item']['sources']:
                        if not source.get('content'):continue
                        original=raw[identity(source)]
                        same(original['user_id'],t['user_id'],'Source isolation')
                        same(original['role'],source['role'],'Source role')
                        require(source['content'] in original['content'],'Visible source is verbatim')
                facets=baseline['requirements']+baseline['fine_selection']['adaptive'].get('context_facets',[])
                def render(items):
                    return len(cascade_rerank.listwise_request(baseline['req'],baseline['plan'],items)['prompt'].encode())
                observed=[observation(c,main_rows,primary_keys(c,links,sessions),facets,
                          fine_evidence,answer_context,render) for c in main_rows]
                main_ids={c['id'] for c in main_rows}
                missing=[]
                for c in baseline['fine']:
                    if c['id'] in main_ids:continue
                    trial=cascade_rerank.listwise_request(baseline['req'],baseline['plan'],main_rows+[c])
                    headroom=t['cascade']['admission_budget']['token_headroom']
                    missing.append(dict(candidate_id=c['id'],ce_score=c.get('_ce_score'),
                        added_prompt_bytes=len(trial['prompt'].encode())-size,
                        exceeds_prompt_bytes=max(0,len(trial['prompt'].encode())-config.RERANK_MAX_PROMPT_BYTES),
                        exceeds_token_headroom=max(0,trial['reservation']-headroom) if headroom is not None else 0))
                rows.append(dict(**result,main_bytes=size,main=observed,missing_fine=missing))
            print(json.dumps(dict(qa_id=result['qa_id'],baseline_exact=True,
                main_with_inherited_sources=sum(bool(c['inherited_source_count']) for c in observed))),flush=True)
    same(seen,set(results),'Complete trace cohort')
    stages={(r['conversation_id'],r['qa_id']):r for r in json.loads(paths[1].read_text())['rows']}
    for row in rows:
        stage=stages[row['conversation_id'],row['qa_id']]
        row['missing_fine_to_main_turns']=[i for i,(a,b) in enumerate(zip(stage['visible']['fine']['turn_coverage'],stage['visible']['listwise_input']['turn_coverage'],strict=True)) if a['full'] and not b['full']]
        reference=set(stage['carrier_ids']['fine'])
        row['reference_linked_missing_ids']=[c['candidate_id'] for c in row['missing_fine'] if c['candidate_id'] in reference]
    all_main=[c for r in rows for c in r['main']]
    selected=[c for c in all_main if c['matches_only_inherited']]
    summary=dict(questions=len(rows),baseline_exact_questions=len(rows),main_candidates=len(all_main),
        with_inherited_sources=sum(bool(c['inherited_source_count']) for c in all_main),
        inherited_only_lexical_matches=len(selected),
        inherited_only_and_primary_visible_elsewhere=sum(c['complete_primary_visible_in_other_main'] for c in selected),
        unknown_primary_ownership=sum(not c['primary_ownership_known'] for c in all_main),
        unauthenticated_views=sum(not c['authenticated_full_view'] for c in all_main))
    inputs=[Path(__file__),Path(__file__).with_name('inventory_visible_source_redundancy.py'),
        Path(__file__).with_name('probe_assistant_admission_order.py'),Path(__file__).with_name('audit_retained_porter_https.py'),run/'memory.db',root/'plan.json',*paths]
    with args.output.open('x') as stream:
        json.dump(dict(status='complete',summary=summary,rows=rows,provider_calls=0,production_changed=False,
            input_sha256={str(p):sha(p) for p in inputs},limitation=__doc__),stream,indent=2);stream.write('\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
