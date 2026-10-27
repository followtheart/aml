"""Offline audit of retained-coarse Porter HTTPS experiments.

Streams one search trace at a time and uses the experiment's frozen production
code. Replays lexical nomination, original coarse selection, actual CE-driven
fine selection and main admission. Does not reconstruct combined fusion for
rejected supplemental candidates: their query vectors/semantic scores were not
recorded. Source transport is not semantic entailment or answer correctness.
Run serially after both cohorts' services and automatic stage audits exit.
"""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    # Reservation rows come from set iteration. Normalize only that metadata
    # list; preserve duplicates and every value, and all selection list order.
    if isinstance(value, dict):
        return {k: sorted((canonical(r) for r in v), key=lambda r: json.dumps(r, sort_keys=True))
                if k == 'reservations' and isinstance(v, list) else canonical(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value


def same(actual, expected, name):
    require(canonical(actual) == canonical(expected), name + ' differs')


def selection_code(cascade):
    body = ast.parse(inspect.getsource(cascade.rank)).body[0].body

    def assignment(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets)

    starts = [i for i, node in enumerate(body) if assignment(node, 'failed')]
    stops = [i for i, node in enumerate(body) if assignment(node, 'overflow_deadline')]
    require(len(starts) == len(stops) == 1 and starts[0] < stops[0], 'Production block changed')
    fragment = ast.Module(body=body[starts[0]:stops[0]], type_ignores=[])
    require(not any(isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith))
                    for n in ast.walk(fragment)), 'Selection fragment is not offline')
    return (compile(ast.fix_missing_locations(fragment), '<frozen-fine-main>', 'exec'),
            hashlib.sha256(ast.dump(fragment).encode()).hexdigest())


def audit_trace(t, expected, result, arm, manifest, modules, code, snapshot):
    cascade, config, porter, schemas, search = modules
    same(t['plan'], expected['plan'], 'Frozen planner')
    same(t['anchor_time'], expected['anchor_time'], 'Frozen anchor')
    for key, value in expected['request'].items():
        same(t.get('requested_include_sensitive') if key == 'include_sensitive' else t[key],
             value, 'Request ' + key)
    require(result['fake'] is False and t['fake'] is False and result['transport'] == 'https',
            'Expected real HTTPS')
    same(result['user_id'], t['user_id'], 'Result user')
    same(result['packet_hash'], t['packet_hash'], 'Result packet hash')
    for name, value in t['versions']['file_hashes'].items():
        same(value, manifest['source_files'][name], 'Frozen source ' + name)
    settings = t['versions']['settings']
    same(settings['RECALL_PORTER_SUPPLEMENT'], arm == 'treatment', 'Arm toggle')
    expected_settings = dict(manifest['reader_versions']['settings'], RECALL_PORTER_SUPPLEMENT=arm == 'treatment')
    same(settings, expected_settings, 'All runtime settings')
    require(settings['COARSE_EVIDENCE_EXTEND'] and settings['FINE_COARSE_CORE_QUOTA']
            and settings['FINE_EVIDENCE_ADAPTIVE'], 'Unexpected selection policy')
    same(settings['RERANK_MAX_PROMPT_BYTES'], 24000, 'Main byte limit')
    req = schemas.SearchRequest(**expected['request'])
    requirements = [{k: r[k] for k in ('id', 'text', 'origin') if k in r}
                    for r in t['coverage_manifest']['coverage']['requirements']]
    plan = dict(copy.deepcopy(t['plan']), _anchor=t['anchor_time'],
                _include_history=t['include_history'], _include_sensitive=t['include_sensitive'],
                _query_specs=copy.deepcopy(t['query_specs']), _coverage_requirements=requirements,
                _fusion_edges=copy.deepcopy(t['fusion_edges']),
                _fusion_triples=copy.deepcopy(t['fusion_triples']))
    c = t['cascade']
    pool = {v['id']: v for v in t['ranked']}
    require(len(pool) == len(t['ranked']), 'Duplicate ranked identity')
    info = t['porter_recall']
    nominated, admitted = info.get('nominated_ids', []), info.get('admitted_ids', [])
    ordinary_routes = [r for r in t['routes'] if r['channel'] != 'lexical_porter']
    known = {v['id'] for r in ordinary_routes for v in r['candidates']}
    require(len(nominated) == len(set(nominated)) <= porter.LIMIT, 'Nomination cap/duplicates')
    require(not set(nominated) & known, 'Supplement already exists in original routes')
    require(len(admitted) == len(set(admitted)) and set(admitted) <= set(nominated), 'Admitted identities')
    same(admitted, c['coarse'].get('supplemental_ids', []), 'Supplemental CE pool')
    if arm == 'control':
        same(info, {'status': 'disabled'}, 'Disabled recall')
    elif info['status'] == 'ok':
        same(info['limit'], porter.LIMIT, 'Nomination limit')
        require(info['original_coarse_preserved'], 'Preservation declaration')
        specs = [q for q in plan['_query_specs'] if q['origin'] != 'grounded_followup']
        snapshot.read_user_id = t['user_id']
        with porter.lexical_view(snapshot, t['user_id']) as lexical:
            full_text, source_text = search._lexical_recall(lexical, req, copy.deepcopy(plan), specs)
        same(info['lexical_ids'], dict(full_text=[v['id'] for v in full_text],
                                      source_text=[v['id'] for v in source_text]), 'Porter lexical routes')
        replay_nominations = search._merge_query_results(
            [[v for v in group if v['id'] not in known] for group in (full_text, source_text)], porter.LIMIT)
        replay_nominations = [v for v in replay_nominations if search._profile_not_expired(
            v, plan['_anchor'], plan['_include_history'])]
        same(nominated, [v['id'] for v in replay_nominations], 'Porter nomination/eligibility')
        supplemental_routes = [r for r in t['routes'] if r['channel'] == 'lexical_porter']
        require(len(supplemental_routes) == 1, 'Expected one supplemental route')
        same(nominated, [v['id'] for v in supplemental_routes[0]['candidates']], 'Recorded nomination route')
    else:
        require(info['status'] in ('error', 'incompatible_settings') and not nominated and not admitted,
                'Unexpected optional fallback')

    # Recover pre-CE candidates in original fused order; CE scores were added
    # later and must not contaminate coarse trace decision metadata.
    original_order = [v['id'] for v in t['fused'] if v['id'] not in set(admitted)]
    # The appended prepared pool follows supplemental coarse order, which can
    # differ from the supplemental fused order captured by the trace.
    ordinary = [copy.deepcopy(pool[mid]) for mid in original_order + admitted
                if mid in pool and not cascade.protected(pool[mid])]
    for v in ordinary:
        v.pop('_ce_score', None)
        v.pop('_final', None)
    plan['_porter_supplement_ids'] = admitted
    replay_coarse, coarse_meta = cascade.recall_shortlist(ordinary, requirements, plan)
    same(coarse_meta, c['coarse'], 'Original coarse preservation and supplemental append')
    coarse = [copy.deepcopy(pool[v['id']]) for v in replay_coarse]
    scores = {}
    for v in coarse:
        score = v.get('_ce_score')
        require(score is None or type(score) in (int, float) and math.isfinite(score), 'Invalid CE score')
        if score is None:
            v.pop('_final', None)
        else:
            v['_final'] = scores[v['id']] = score
    ce = c['cross_encoder']
    succeeded = {mid for batch in ce['batches'] if batch['status'] == 'ok' for mid in batch['candidate_ids']}
    same(set(scores), succeeded, 'Successful CE batch identities')
    same(ce['input_count'], len(coarse), 'CE input count')
    same(ce['scored_count'], len(scores), 'CE scored count')
    same(ce['status'], 'ok' if len(scores) == len(coarse) else 'partial' if scores else 'fallback', 'CE status')
    b = c['admission_budget']
    limits = None if b['token_headroom'] is None else SimpleNamespace(
        max_tokens=b['token_headroom'], tokens=0, reserved_tokens=0,
        max_calls=b['call_headroom'], calls=0, reserved_calls=0,
        deadline=time.monotonic() + b['deadline_seconds'])
    replay_trace = dict(coarse=copy.deepcopy(c['coarse']), omitted=[], reservation_missing=[])
    env = dict(vars(cascade), coarse=coarse, coarse_selection=coarse_meta['selection'],
               scores=scores, trace=replay_trace, req=req, plan=plan, requirements=requirements, limits=limits)
    exec(code, env)
    same(replay_trace['fine'], c['fine'], 'Fine selection and reservations')
    same(replay_trace.get('adaptive_admission'), c.get('adaptive_admission'), 'Adaptive admission')
    same(env['admission_selection'], c['listwise']['selection'], 'Main selection')
    same(sorted(env['baseline_ids']), c['listwise']['baseline_candidate_ids'], 'Main baseline')
    same(sorted(env['carried_ids']), c['listwise']['carried_ids'], 'Main carried IDs')
    main_ids = [v['id'] for v in env['submitted']]
    same(main_ids, c['listwise']['candidate_ids'], 'Main candidate order')
    request = cascade.listwise_request(req, plan, env['submitted']) if main_ids else None
    main_bytes = len(request['prompt'].encode()) if request else 0
    same(main_bytes, c['listwise']['input_bytes'], 'Main prompt bytes')
    require(main_bytes <= 24000, 'Main prompt cap')
    require(request is None or b['token_headroom'] is None or request['reservation'] <= b['token_headroom'],
            'Main token headroom')
    return dict(qa_id=result['qa_id'], search_id=t['search_id'], status='passed',
                porter_status=info['status'], nominated_ids=nominated, admitted_ids=admitted,
                coarse_count=len(coarse), ce_status=ce['status'], ce_errors=ce['errors'],
                fine_ids=[v['id'] for v in env['fine']], main_ids=main_ids, main_bytes=main_bytes,
                new_main_ids=[mid for mid in main_ids if mid in admitted],
                reader_error=result.get('error_stage'), score=result['score'])


def main(args):
    root = args.run.resolve()
    require(not args.output.exists(), 'Refusing to overwrite audit')
    require((root / 'exit-code').exists(), 'Experiment is still active')
    manifest = json.loads((root / 'plan.json').read_text())
    fixture = json.loads((root / 'fixture.json').read_text())
    same(manifest['toggle'], 'RECALL_PORTER_SUPPLEMENT', 'Experiment toggle')
    same(manifest['setting_values'], dict(control=False, treatment=True), 'Arm settings')
    same(sha(root / 'fixture.json'), manifest['fixture_sha256'], 'Fixture hash')
    same(sha(root / 'frozen-memory.db'), manifest['database_sha256'], 'Frozen DB hash')
    for name, digest in manifest['source_files'].items():
        same(sha(root / 'source' / name), digest, 'Frozen source ' + name)
    project = root / 'source/memory_system'
    sys.path.insert(0, str(project))
    from app import cascade_rerank, config, porter_recall, schemas, search_pipeline, store
    modules = cascade_rerank, config, porter_recall, schemas, search_pipeline
    code, code_sha = selection_code(cascade_rerank)
    expected = {(r['conversation_id'], r['qa_id']): r for r in fixture['requests']}
    require(len(expected) == len(fixture['requests']) == manifest['questions'], 'Fixture question count')
    inputs, arms = {}, {}
    snapshot = store.Store(':memory:')
    with sqlite3.connect((root / 'frozen-memory.db').as_uri() + '?mode=ro', uri=True) as db:
        db.backup(snapshot.conn)
    snapshot.conn.execute('PRAGMA query_only=ON')
    try:
        with patch('socket.socket.connect', side_effect=AssertionError('Offline audit forbids network')):
            for arm in ('control', 'treatment'):
                run = root / arm
                terminal = json.loads((run / 'isolated-manifest.json').read_text())
                require('client_exit_code' in terminal and terminal.get('server_exit_code') in (0, -15),
                        'Arm is not terminal')
                same(terminal.get('audit_exit_code'), 0, 'Automatic source transport audit')
                same(sha(Path(terminal['data'])), manifest['data_sha256'], 'Prepared dataset hash')
                same(terminal['initial_database_sha256'], manifest['database_sha256'], 'Initial arm DB')
                result_path = run / 'completed/data/results/personamem-v2-32k.jsonl'
                rows = list(records(result_path))
                same({(r['conversation_id'], r['qa_id']) for r in rows}, set(expected), 'Question set')
                require(len(rows) == len(expected), 'Duplicate result identity')
                results = {r['search_id']: r for r in rows if r.get('search_id')}
                require(len(results) == len(rows), 'Missing or duplicate search results')
                audited, seen = [], set()
                trace_path = run / 'server-search-debug.jsonl'
                for t in records(trace_path):
                    sid = t['search_id']
                    require(sid not in seen and sid in results, 'Duplicate/unexpected trace')
                    seen.add(sid)
                    r = results[sid]
                    try:
                        with patch.multiple(config, **t['versions']['settings']):
                            row = audit_trace(t, expected[r['conversation_id'], r['qa_id']], r,
                                              arm, manifest, modules, code, snapshot)
                    except Exception as exc:
                        row = dict(qa_id=r['qa_id'], search_id=sid, status='failed',
                                   error=type(exc).__name__, detail=str(exc))
                    audited.append(row)
                    print(json.dumps(dict(arm=arm, qa_id=r['qa_id'], status=row['status'],
                                          detail=row.get('detail'))), flush=True)
                same(seen, set(results), 'Trace completeness')
                arms[arm] = dict(questions=len(audited), failed=sum(r['status'] != 'passed' for r in audited),
                                 terminal_exit_code=terminal['client_exit_code'], rows=audited)
                for path in (trace_path, result_path, run / 'funnel/stage-transport.json', run / 'isolated-manifest.json'):
                    inputs[str(path)] = sha(path)
    finally:
        snapshot.conn.close()
    report = dict(status='failed' if any(a['failed'] for a in arms.values()) else 'passed',
                  arms=arms, provider_calls=0, limitation=__doc__, input_sha256=inputs,
                  auditor_sha256=sha(Path(__file__)), selection_ast_sha256=code_sha,
                  plan_sha256=sha(root / 'plan.json'), fixture_sha256=manifest['fixture_sha256'])
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
