"""Independently verify source-anchor proposal selection, budgets and provenance."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_context


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def records(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main(args):
    root=args.run.resolve();project=Path(__file__).resolve().parents[1]
    read=lambda name:json.loads((root/name).read_text())
    traces={t['search_id']:t for t in records(root/'search/server-search-debug.jsonl')}
    inputs={j['qa']['id']:j['payload']['search_id'] for j in records(root/'http-reader-inputs.jsonl')}
    measured=read('source-hit-identity-proposal.json')
    if (measured['questions']!=56 or measured['trace_sha256']!=sha(root/'search/server-search-debug.jsonl') or
            measured['database_sha256']!=sha(root/'frozen-memory.db')):
        raise ValueError('Source measurement inputs changed')
    with sqlite3.connect((root/'frozen-memory.db').as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        sources={(r['request_id'],r['message_index']):dict(r) for r in db.execute('SELECT * FROM source_messages')}
        amus={r['id']:dict(r) for r in db.execute('SELECT * FROM amu')}
        links={(r['amu_id'],r['request_id'],r['message_index']) for r in db.execute('SELECT * FROM amu_sources')}
    source_occurrences=0;tie_lanes=0;total_lanes=0
    for job in measured['rows']:
        t=traces[job['search_id']]
        observed=next(r for r in t['routes'] if r['channel']=='source_text' and r['round']==1)
        if job['variants']['baseline']!=[c['id'] for c in observed['candidates']]:
            raise ValueError('Baseline source allocation differs')
        initial=[s for s in t['query_specs'] if s['origin']!='grounded_followup']
        if [q['query'] for q in job['hits']]!=[s['text'] for s in initial]:
            raise ValueError('Query sequence changed')
        match_union={mid:set() for mid in job['source_sets']}
        for lane in job['hits']:
            if set(lane['sources'])!=set(lane['candidate_ids']):raise ValueError('Candidate/hit correspondence changed')
            for mid,ss in lane['sources'].items():
                total_lanes+=1;tie_lanes+=len(ss)>1
                a=amus[mid]
                if (a['user_id']!=t['user_id'] or a['sensitivity']=='suppressed' or a['view_status']!='ready'
                        or a['resolution_status']=='retracted' or not t['include_history'] and a['valid_to'] is not None):
                    raise ValueError('Ineligible source carrier')
                if not ss or len({s['score'] for s in ss})!=1:raise ValueError('Score-compatible source set invalid')
                for s in ss:
                    k=s['request_id'],s['message_index'];raw=sources[k]
                    if (raw['user_id']!=t['user_id'] or raw['sensitivity']=='suppressed' or
                            not t['include_sensitive'] and (raw['sensitivity']=='sensitive' or a['sensitivity']=='sensitive') or
                            (mid,*k) not in links or any(s[key]!=raw[key] for key in ('content','role','source_event_id'))):
                        raise ValueError('Source text, ownership, sensitivity or linkage changed')
                    match_union[mid].add(k);source_occurrences+=1
        for mid,value in match_union.items():
            if value!=set(map(tuple,job['source_sets'][mid]['matched'])):raise ValueError('Matched-source merge differs')
    selections_verified=0;gains={};proposal_summaries={}
    baseline_rows={r['qa_id']:r for r in read('semantic-source-context-proposal.json')['rows']}
    variants={'historical8_remainder':(8,False,False),'novel8_remainder':(8,True,False),
        'novel40_remainder':(40,True,False),'any40_extra4k':(40,False,True),'novel40_extra4k':(40,True,True),
        'all8_remainder':(8,False,False),'missing8_remainder':(8,True,False),
        'missing40_remainder':(40,True,False),'missing40_extra4k':(40,True,True)}
    for filename in ('semantic-novel-anchor-proposal.json','lexical-missing-anchor-proposal.json'):
        data=read(filename)
        if len(data['rows'])!=56 or len({r['qa_id'] for r in data['rows']})!=56:raise ValueError('Incomplete proposals')
        for row in data['rows']:
            t=traces[inputs[row['qa_id']]]
            if row['coverage']['baseline']!=baseline_rows[row['qa_id']]['coverage']['baseline']:
                raise ValueError('Reference baseline changed')
            for mode,prop in row['proposals'].items():
                view,variant=mode.split('_',1) if filename.startswith('semantic-') else (None,mode)
                depth,novel,extra=variants[variant]
                allowance=4096 if extra else row['remaining_bytes']
                inventory=[d for d in row['inventory'] if view is None or d['view']==view]
                ordered=[];costs={}
                for d in inventory:
                    if d['source_rank']>depth or novel and d['anchor_already_delivered']:continue
                    for mid,cost in zip(d['candidate_ids'],d['costs'],strict=True):
                        if mid in costs and costs[mid]!=cost:raise ValueError('Carrier cost differs across anchors')
                        if mid not in costs:ordered.append(mid)
                        costs[mid]=cost
                used=sum(len(p['content'].encode()) for p in t['returned'])+max(0,len(t['returned'])-1)
                added=[];bytes_added=0
                expected=[]
                pool={c['id']:c for c in t['ranked']}
                def source_identity(s):
                    return json.dumps({k:s.get(k) for k in ('request_id','message_index','source_event_id',
                        'user_id','role','content_span','content')},sort_keys=True,ensure_ascii=False)
                def complete_user(s):
                    span=s.get('content_span')
                    return s.get('role')=='user' and (not span or
                        span['start']==0 and span['end']==span['original_length']==len(s['content']))
                known={source_identity(s) for p in t['returned'] for s in p['sources'] if s.get('content')}
                for mid in ordered:
                    ss=[s for s in pool[mid]['_packet_item']['sources'] if s.get('content')]
                    # A preceding addition may already supply every complete
                    # user message of this carrier; it then has no proposal row.
                    if not any(complete_user(s) and source_identity(s) not in known for s in ss):continue
                    cost=costs[mid]
                    if cost!=len(answer_context.with_evidence('Conversation excerpt:',ss).encode())+bool(t['returned']):
                        raise ValueError('Recorded carrier byte cost differs from original source text')
                    reason=('context_limit' if added else 'top_k' if len(t['returned'])>=t['top_k'] else
                        'reserve_bytes' if bytes_added+cost>allowance else
                        'token_budget' if used+cost>t['coverage_manifest']['token_budget'] else 'admitted')
                    expected.append((mid,cost,reason))
                    if reason=='admitted':
                        added.append(mid);bytes_added+=cost;used+=cost
                        known.update(source_identity(s) for s in ss)
                actual=[(p['candidate_id'],p['added_bytes'],p['reason']) for p in prop['proposed']]
                if (actual!=expected or prop['appended_ids']!=added or prop['added_bytes']!=bytes_added or
                        prop['total_bytes']!=used or prop['reserve_bytes']!=allowance):
                    raise ValueError('Independent selection or budget check failed '+json.dumps(dict(
                        qa_id=row['qa_id'],mode=mode,actual=actual,expected=expected,
                        totals=[prop['total_bytes'],used],added=[prop['added_bytes'],bytes_added],
                        ids=[prop['appended_ids'],added],reserve=[prop['reserve_bytes'],allowance])))
                if any(a and not b for role in ('user','assistant') for a,b in zip(row['coverage']['baseline'][role],row['coverage'][mode][role],strict=True)):
                    raise ValueError('Reference regression')
                selections_verified+=1
        for s in data['summary']:
            mode=s['mode']
            for scope in ('baseline',mode):
                if (s['reference_questions'][scope]!=sum(any(r['coverage'][scope]['user']) for r in data['rows']) or
                    s['user_turns'][scope]!=sum(sum(r['coverage'][scope]['user']) for r in data['rows']) or
                    s['complete_sets'][scope]!=sum(bool(r['coverage'][scope]['user']) and all(r['coverage'][scope]['user']) for r in data['rows']) or
                    s['catalog_bytes'][scope]!=sum(r['catalog_bytes'][scope] for r in data['rows'])):
                    raise ValueError('Proposal summary mismatch')
            gains[mode]=[dict(qa_id=r['qa_id'],before=r['coverage']['baseline']['user'],after=r['coverage'][mode]['user'],
                added_ids=r['proposals'][mode]['appended_ids']) for r in data['rows'] if r['coverage']['baseline']['user']!=r['coverage'][mode]['user']]
        proposal_summaries[filename]=data['summary']
    if selections_verified!=1064:raise ValueError('Not all 19 variants verified')
    from app import config
    if config.LLM_MODEL!='gpt-4o-mini' or config.CE_MODEL!='qwen3-rerank' or config.EMBED_MODEL!='openai/text-embedding-v4' or config.EMBED_DIM!=256:
        raise ValueError('Runtime model changed')
    cfg=project/'app/config.py';module=ast.parse(cfg.read_text())
    module.body=[n for n in module.body if not(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='_load_dotenv')]
    defaults={'__file__':str(cfg)}
    with patch.dict(os.environ,{},clear=True):exec(compile(module,str(cfg),'exec'),defaults)
    old=records(project/'runs/https-fine-context-fill-20261023/control/server-search-debug.jsonl')[0]['versions']['settings']
    bools={k:v for k,v in old.items() if type(v) is bool}
    if len(bools)!=65 or any(defaults[k]!=v for k,v in bools.items()):raise ValueError('Historical boolean default changed')
    files=read('plan.json')['source_files'];app_files={k:v for k,v in files.items() if k.startswith('memory_system/app/')}
    for name,expected in app_files.items():
        if sha(project/name.removeprefix('memory_system/'))!=expected:raise ValueError('Runtime code changed')
    archive=root/'source-anchor-study-scripts';archive.mkdir(exist_ok=False)
    names=('probe_semantic_novel_anchor.py','probe_source_hit_identity.py','probe_lexical_missing_anchor.py',
        'probe_source_rank_residual.py','probe_semantic_source_context.py','audit_https_funnel.py',
        'evaluation_model_guard.py','finalize_source_anchor_study.py')
    for name in names:shutil.copy2(project/'scripts'/name,archive/name)
    artifacts={str(p.relative_to(root)):sha(p) for p in archive.iterdir()}
    for stem in ('semantic-novel-anchor-proposal','source-hit-identity-proposal','lexical-missing-anchor-proposal'):
        for suffix in ('.json','.log'):artifacts[stem+suffix]=sha(root/(stem+suffix))
    result=dict(variants_verified=selections_verified,questions=56,source_routes_reproduced=56,fts_routes_reproduced=56,
        source_occurrences_checked=source_occurrences,query_candidate_lanes=total_lanes,ambiguous_equal_score_source_lanes=tie_lanes,
        reference_gains=gains,source_hit_summary=measured['summary'],proposal_summaries=proposal_summaries,
        historical_boolean_defaults_verified=len(bools),unchanged_app_files=len(app_files),provider_calls=0,new_https_run=False,
        runtime_promoted=False,artifacts=artifacts,
        decision='No new reference question in any append variant; source-hit reallocation loses source-route carriers. Do not promote.',
        limitations=['BM25 equal-score ties identify all compatible sources, not a uniquely recoverable historical row.',
            'Full reference sets measure transport, not semantic support or answer accuracy.',
            'The original packet and other historical retrieval routes are frozen; this is not a live Search.'])
    with (root/'source-anchor-allocation-study.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('variants_verified','source_occurrences_checked','query_candidate_lanes',
        'ambiguous_equal_score_source_lanes','historical_boolean_defaults_verified','unchanged_app_files','provider_calls','decision')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    main(p.parse_args())
