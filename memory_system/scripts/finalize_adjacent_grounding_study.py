"""Audit saved adjacent-context and reader diagnostics, freeze sources and ledger."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main(args):
    root=args.run.resolve()
    project=Path(__file__).resolve().parents[1]
    read=lambda name:json.loads((root/name).read_text())
    vector_plan=read('source-adjacent-separate-vectors/plan.json')
    vector_audit=read('source-adjacent-separate-vectors/verification.json')
    if (vector_plan['pairing_policy']!='eligible_adjacent' or
            vector_audit['database_sha256']!=sha(root/'frozen-memory.db') or
            vector_audit['rankings_sha256']!=sha(root/'source-adjacent-separate-vectors/rankings.json') or
            vector_audit['max_score_drift']!=0):
        raise ValueError('Source measurement not verified')
    pairs=[m for m in vector_plan['exchange_sources'].values() if len(m)==2]
    separate=[m for m in pairs if not set(m[0]['amu_ids']) & set(m[1]['amu_ids'])]
    if len(pairs)!=193 or len(separate)!=11 or vector_audit['reused_vector_rows_exact']!=1250:
        raise ValueError('Unexpected pair inventory')
    for u,a in pairs:
        if (u['role']!='user' or a['role']!='assistant' or
                any(u[k]!=a[k] for k in ('user_id','session_id','request_id')) or
                u['message_index']+1!=a['message_index']):
            raise ValueError('Invalid source adjacency')
    proposals=read('separate-carrier-context-proposal.json')
    if (len(proposals['rows'])!=56 or len({r['qa_id'] for r in proposals['rows']})!=56 or
            proposals['measured_plan_sha256']!=sha(root/'source-adjacent-separate-vectors/plan.json') or
            proposals['measured_rankings_sha256']!=vector_audit['rankings_sha256']):
        raise ValueError('Incomplete or stale proposals')
    for row in proposals['rows']:
        inventory={d['source_index']:d for d in row['inventory']}
        for mode in ('ranked_top8','retained_top40'):
            prop=row['proposals'][mode]
            admitted=[p for p in prop['proposed'] if p['reason']=='admitted']
            if len(admitted)>1 or len(admitted)!=len(prop['appended_ids']):
                raise ValueError('Proposal item count differs')
            if prop['added_bytes']>row['remaining_bytes']:
                raise ValueError('Original context remainder exceeded')
            for p in admitted:
                d=inventory[p['source_index']]
                if (not d['eligible'] or d['full_visible_members']!=2 or any(s is None for s in d['suppliers']) or
                        not 1<=d['source_rank']<=(8 if mode=='ranked_top8' else 40) or
                        (mode=='retained_top40' and not d['retained_members']) or p['added_bytes']!=d['added_bytes']):
                    raise ValueError('Unqualified context admitted')
            for role in ('user','assistant'):
                if row['coverage'][mode][role]!=row['coverage']['baseline'][role]:
                    raise ValueError('Unexpected reference change')
    factorial=read('core-gap-scope-factorial/summary.json')
    factorial_audit=read('core-gap-scope-factorial/audit.json')
    tail_plan=read('identity-grounding-tail/plan.json')
    tail_summary=read('identity-grounding-tail/summary.json')
    parent_plan=read('core-gap-scope-factorial/plan.json')
    parent_rows=records(root/'core-gap-scope-factorial/results.jsonl')
    tails=records(root/'identity-grounding-tail/results.jsonl')
    for directory,summary in (('core-gap-scope-factorial',factorial),('identity-grounding-tail',tail_summary)):
        if (root/directory/'exit-code').read_text().strip()!='0':
            raise ValueError('Diagnostic lacks successful terminal state')
        for filename,key in (('plan.json','plan_sha256'),('results.jsonl','results_sha256')):
            if sha(root/directory/filename)!=summary[key]:
                raise ValueError('Diagnostic inputs changed')
    for key in ('plan_sha256','results_sha256'):
        if factorial_audit[key]!=factorial[key] or tail_plan['parent_'+key]!=factorial[key]:
            raise ValueError('Parent audit or tail provenance stale')
    if (factorial_audit['readers_replayed']!=24 or factorial_audit['calls_replayed']!=146 or
            len(tails)!=24 or tail_summary['verified_tails']!=24):
        raise ValueError('Incomplete reader verification')
    strata={(r['qa_id'],r['repetition']):r for r in parent_rows if r['packet_mode']=='augmented' and r['cited_scope']}
    expected={(qid,rep,repeat,enabled) for qid,rep in strata for repeat in (1,2) for enabled in (False,True)}
    actual={(r['qa_id'],r['archive_repetition'],r['repetition'],r['enabled']) for r in tails}
    if expected!=actual or len(actual)!=len(tails):
        raise ValueError('Unbalanced or missing fixed-prefix comparison')
    verdicts=[]
    for r in tails:
        original=strata[r['qa_id'],r['archive_repetition']]
        boundary=next(i for i,c in enumerate(original['diagnostics']['answer_calls']) if c['stage']=='eval.choice_entailment')
        if r['diagnostics']['answer_calls'][:boundary]!=original['diagnostics']['answer_calls'][:boundary]:
            raise ValueError('Archived prefix changed')
        first=r['requests'][0]
        if first['prompt']!=original['requests'][boundary]['prompt']+(tail_plan['note'] if r['enabled'] else ''):
            raise ValueError('Uncontrolled prompt intervention')
        for key in ('schema','system','max_tokens'):
            if first.get(key)!=original['requests'][boundary].get(key):
                raise ValueError('Provider settings changed')
        for req,call in zip(r['requests'],r['diagnostics']['answer_calls'][boundary:],strict=True):
            if req['stage']!=call['stage'] or hashlib.sha256(req['prompt'].encode()).hexdigest()!=call['prompt_sha256']:
                raise ValueError('Saved prompt differs from call')
        if r['live_calls']!=len(r['requests']) or len(r['fixed_calls'])!=boundary:
            raise ValueError('Tail accounting differs')
        verdicts.append(dict(qa_id=r['qa_id'],archive_repetition=r['archive_repetition'],repetition=r['repetition'],
            enabled=r['enabled'],prediction=r['prediction'],score=r['score'],
            checks=next(c['response']['checks'] for c in r['diagnostics']['answer_calls'] if c['stage']=='eval.choice_entailment')))
    groups={str(enabled):dict(readers=sum(r['enabled']==enabled for r in tails),
        score=sum(r['score'] for r in tails if r['enabled']==enabled),
        predictions={qid:[r['prediction'] for r in tails if r['enabled']==enabled and r['qa_id']==qid]
            for qid in {c['qa']['id'] for c in parent_plan['cases']}}) for enabled in (False,True)}
    if groups!=tail_summary['groups']:
        raise ValueError('Tail summary differs')
    events=[]
    ledger={}
    for directory in ('source-adjacent-separate-vectors','core-gap-scope-factorial','identity-grounding-tail'):
        es=[dict(p.split('=',1) for p in line.split() if '=' in p)
            for line in (root/directory/'provider.log').read_text().splitlines() if 'provider_call ' in line]
        if any(e['model']!=('openai/text-embedding-v4' if directory.startswith('source-') else 'gpt-4o-mini') for e in es):
            raise ValueError('Model dispatch changed')
        ledger[directory]=dict(calls=len(es),recorded_tokens=sum(int(e['total_tokens']) for e in es),
            statuses=dict(Counter(e['status'] for e in es)))
        events.extend(es)
    if len(events)!=180 or sum(int(e['total_tokens']) for e in events)!=813935:
        raise ValueError('Study ledger differs')
    cfg=project/'app/config.py'
    module=ast.parse(cfg.read_text())
    module.body=[n for n in module.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='_load_dotenv')]
    defaults={'__file__':str(cfg)}
    with patch.dict(os.environ,{},clear=True):
        exec(compile(module,str(cfg),'exec'),defaults)
    old=next(iter(records(project/'runs/https-fine-context-fill-20261023/control/server-search-debug.jsonl')))['versions']['settings']
    booleans={k:v for k,v in old.items() if type(v) is bool}
    if len(booleans)!=65 or any(defaults[k]!=v for k,v in booleans.items()):
        raise ValueError('Historical defaults changed')
    from app import config
    if (config.LLM_MODEL!='gpt-4o-mini' or config.CE_MODEL!='qwen3-rerank' or
            config.EMBED_MODEL!='openai/text-embedding-v4' or vector_plan['dimension']!=256):
        raise ValueError('Runtime models changed')
    frozen=read('plan.json')['source_files']
    app_files={k:v for k,v in frozen.items() if k.startswith('memory_system/app/')}
    for name,expected_hash in app_files.items():
        if sha(project/name.removeprefix('memory_system/'))!=expected_hash:
            raise ValueError('Application changed')
    archive=root/'adjacent-grounding-study-scripts'
    archive.mkdir(exist_ok=False)
    names=('probe_source_exchange_vectors.py','audit_current_source_vectors.py',
        'verify_source_semantic_measurements.py','probe_separate_carrier_context.py',
        'probe_semantic_source_context.py','probe_source_rank_residual.py','probe_source_ce.py',
        'probe_core_gap_scope_factorial.py','audit_core_gap_scope_factorial.py','probe_identity_grounding_tail.py',
        'replay_query_context.py','evaluation_model_guard.py','evaluate_shared_search_https.py',
        'audit_core_gap_context.py','finalize_adjacent_grounding_study.py')
    for name in names:
        shutil.copy2(project/'scripts'/name,archive/name)
    artifacts={}
    for directory in ('source-adjacent-separate-vectors','core-gap-scope-factorial','identity-grounding-tail','adjacent-grounding-study-scripts'):
        for p in sorted((root/directory).rglob('*')):
            if p.is_file():artifacts[str(p.relative_to(root))]=sha(p)
    for name in ('separate-carrier-context-proposal.json','separate-carrier-context-proposal.log',
                 'core-gap-scope-factorial.log','identity-grounding-tail.log'):
        artifacts[name]=sha(root/name)
    result=dict(ledger=ledger,provider_calls=len(events),recorded_tokens=813935,
        provider_statuses=dict(Counter(e['status'] for e in events)),
        paired_sources=len(pairs),new_separate_pairs=len(separate),vector_verification=vector_audit,
        context_proposals=proposals['summary'],factorial=factorial['summary'],tail=tail_summary,
        entailment_verdicts=verdicts,historical_defaults_verified=len(booleans),unchanged_app_files=len(app_files),
        script_snapshots=len(names),artifacts=artifacts,new_https_run=False,runtime_promoted=False,
        decision='Neither separate-carrier context nor the grounding-tail proposal justified promotion. Latest full HTTPS remains 31/56 vs 34/56.',
        limitations=['Three selected questions and repeated tails are not a cohort accuracy estimate.',
            'The parent factorial contains a provider HTTP503; recorded zero usage is not a billing assertion.',
            tail_plan['limitation']])
    with (root/'adjacent-grounding-study.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps({k:result[k] for k in ('ledger','provider_calls','recorded_tokens',
        'historical_defaults_verified','unchanged_app_files','script_snapshots','decision')},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    main(p.parse_args())
