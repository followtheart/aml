"""Verify frozen upstream candidates, CE provenance, defaults and accounting."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from probe_semantic_upstream import records


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def key(c):
    return c['id'] + ':' + hashlib.sha256(c['_rank_text'].encode()).hexdigest()


def main(args):
    root = args.run.resolve()
    upstream, ce = root/'source-upstream-diagnostic', root/'source-upstream-ce-diagnostic'
    for directory in (upstream, ce):
        if (directory/'exit-code').read_text().strip() != '0':
            raise ValueError('Require complete diagnostics')
    trace_path = root/'search/server-search-debug.jsonl'
    traces = {t['search_id']:t for t in records(trace_path)}
    upstream_summary = json.loads((upstream/'summary.json').read_text())
    ce_summary, ce_plan = [json.loads((ce/name).read_text()) for name in ('summary.json','plan.json')]
    if sha(upstream/'selections.jsonl') != upstream_summary['selections_sha256'] or sha(upstream/'selections.jsonl') != ce_plan['upstream_sha256']:
        raise ValueError('Upstream selection hash changed')
    if sha(trace_path) != ce_plan['source_trace_sha256'] or sha(ce/'plan.json') != ce_summary['plan_sha256']:
        raise ValueError('CE inputs changed')
    if sha(ce/'selections.jsonl') != ce_summary['selections_sha256']:
        raise ValueError('CE selections changed')
    source_plan = json.loads((root/'source-vector-diagnostic/plan.json').read_text())
    if sha(root/'frozen-memory.db') != source_plan['database_sha256']:
        raise ValueError('Database changed')
    provenance = json.loads((upstream/'selection-provenance.json').read_text())
    original = upstream/'original-probe_semantic_upstream.py'
    if sha(original) != provenance['selection_script_sha256'] or provenance['selections_sha256'] != upstream_summary['selections_sha256']:
        raise ValueError('Original selection producer changed')
    current = Path(__file__).parent/'probe_semantic_upstream.py'
    def functions(path):
        return {n.name:ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    old_funcs, new_funcs = functions(original), functions(current)
    for name in ('nominate','stages'):
        if old_funcs[name] != new_funcs[name]:
            raise ValueError('Recovery changed selection code')
    scored = {r['qa_id']:r for r in records(ce/'selections.jsonl')}
    live = records(ce/'live.jsonl')
    novel_scores = {}
    for row in live:
        if 'error' in row or len(row['keys']) != len(row['scores']):
            raise ValueError('Incomplete CE response')
        for k, value in zip(row['keys'], row['scores']):
            identity = row['qa_id'], k
            if identity in novel_scores:
                raise ValueError('Repeated fresh CE pair')
            novel_scores[identity] = value
    expected_pairs = {(j['qa_id'], v['key']) for j in ce_plan['jobs'] for v in j['novel']}
    if set(novel_scores) != expected_pairs:
        raise ValueError('Missing or extra fresh CE pairs')
    checks, fresh_assignments, reused_assignments = [], 0, 0
    for j in (json.loads(line) for line in (upstream/'selections.jsonl').open()):
        t, final = traces[j['search_id']], scored[j['qa_id']]
        base = j['variants']['baseline']
        if ([c['id'] for c in base['fused']] != [c['id'] for c in t['fused']]
                or any(c['_fusion'] != old['_fusion'] or c['_fused'] != old['_fused']
                       for c,old in zip(base['fused'], t['fused']))):
            raise ValueError('Saved baseline fusion differs')
        original_candidates = {c['id']:c for c in t['ranked']}
        for c in base['prepared']:
            if any(c[k] != original_candidates[c['id']][k] for k in ('_rank_text','_packet_item')):
                raise ValueError('Saved baseline source text differs')
        if [c['id'] for c in base['coarse']] != t['cascade']['coarse']['selected_ids']:
            raise ValueError('Saved coarse differs')
        historical = {key(c):c['_ce_score'] for c in t['ranked'] if c.get('_ce_score') is not None}
        if ([c['id'] for c in final['variants']['baseline']['fine']] != t['cascade']['fine']['selected_ids']
                or [c['id'] for c in final['variants']['baseline']['main']] != t['cascade']['listwise']['candidate_ids']
                or final['variants']['baseline']['main_bytes'] != t['cascade']['listwise']['input_bytes']):
            raise ValueError('Saved baseline fine/main differs')
        for mode, variant in j['variants'].items():
            if len(variant['coarse']) > t['cascade']['coarse']['limit']:
                raise ValueError('Coarse budget exceeded')
            actual = final['variants'][mode]
            if actual['main_bytes'] > t['versions']['settings']['RERANK_MAX_PROMPT_BYTES']:
                raise ValueError('Main byte budget exceeded')
            for c in variant['coarse']:
                k = key(c)
                if k in historical:
                    value = historical[k]
                    reused_assignments += 1
                else:
                    value = novel_scores[j['qa_id'],k]
                    fresh_assignments += 1
                if actual['coarse_scores'][c['id']] != value:
                    raise ValueError('Coarse score does not match provider provenance')
            for stage in ('fine','main'):
                for c in actual[stage]:
                    if c['_ce_score'] != actual['coarse_scores'][c['id']]:
                        raise ValueError('Fine/main score changed')
        checks.append(j['qa_id'])
    if len(checks) != 56 or len(set(checks)) != 56 or len(scored) != 56:
        raise ValueError('Incomplete cohort')
    events = [dict(p.split('=',1) for p in line.split() if '=' in p)
              for line in (ce/'provider.log').read_text().splitlines() if 'provider_call ' in line]
    if (len(events) != ce_plan['expected_calls'] or len(events) != ce_summary['calls']
            or sum(int(e['total_tokens']) for e in events) != ce_summary['tokens']
            or any(e['model'] != 'qwen3-rerank' or e['kind'] != 'rerank' or e['status'] != 'ok' for e in events)):
        raise ValueError('CE provider accounting differs')
    project = Path(__file__).resolve().parents[1]
    cfg = project/'app/config.py'
    module = ast.parse(cfg.read_text())
    module.body = [n for n in module.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id == '_load_dotenv')]
    defaults = {'__file__':str(cfg)}
    with patch.dict(os.environ, {}, clear=True):
        exec(compile(module,str(cfg),'exec'),defaults)
    old_settings = records(project/'runs/https-fine-context-fill-20261023/control/server-search-debug.jsonl')[0]['versions']['settings']
    booleans = {k:v for k,v in old_settings.items() if type(v) is bool}
    if len(booleans) != 65 or any(defaults[k] != v for k,v in booleans.items()):
        raise ValueError('Historical default changed')
    if defaults['LLM_MODEL'] != 'gpt-4o-mini' or ce_plan['llm_model'] != 'gpt-4o-mini':
        raise ValueError('LLM changed')
    frozen = json.loads((root/'plan.json').read_text())['source_files']
    app_files = {k:v for k,v in frozen.items() if k.startswith('memory_system/app/')}
    for name, expected in app_files.items():
        if sha(project/name.removeprefix('memory_system/')) != expected:
            raise ValueError('Production app changed')
    result = dict(questions_verified=len(checks), variants_verified=224,
        historical_stages_verified=['fusion','prepared','coarse','fine','main_input'],
        distinct_new_ce_pairs=len(novel_scores), fresh_score_assignments=fresh_assignments,
        historical_score_assignments=reused_assignments, provider_calls=len(events), tokens=ce_summary['tokens'],
        historical_boolean_defaults_verified=len(booleans), unchanged_app_files=len(app_files),
        verifier_sha256=sha(Path(__file__)), limitation='No live HTTPS, downstream selection, final-packet or answer-rate validation.')
    with args.output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
