"""Freeze affordable source nominees before scoring single-addition recall bounds.

This is a preflight, not a learned selector, provider call or answer evaluation.
All complete cohorts are fixed without references before any bound is scored.
"""
import argparse
import ast
from collections import Counter
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

from batched_records import json_records, csv_records, file_digest
from audit_retained_porter_https import require, same, selection_code
from probe_assistant_admission_order import replay, cards
import audit_https_funnel as transport
import local_candidate_ids
from screened_source_views import novel_views, schedule_bounds


def read(path):
    data = bytearray()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            data.extend(block)
    return json.loads(data)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def view_builder(config, cross_encoder, answer_context):
    from app.conversation_context import source_key, usable
    from app.fine_context_reserve import full_user
    path = Path(__file__).with_name('probe_source_ce.py')
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_views']
    require(len(nodes) == 1, 'Unique pure source view builder')
    env = dict(hashlib=hashlib, copy=copy, config=config, cross_encoder=cross_encoder,
               answer_context=answer_context, source_key=source_key, usable=usable, full_user=full_user)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), env)
    return env['build_views']


def prepare(args):
    run = args.run.resolve()
    root = run.parent
    plan, manifest = read(root/'plan.json'), read(run/'isolated-manifest.json')
    require(run.name == 'control' and manifest['server_exit_code'] in (0, -15)
            and manifest['client_exit_code'] == manifest['audit_exit_code'] == 0, 'Terminal control')
    for name, digest in plan['source_files'].items():
        same(file_digest(root/'source'/name), digest, 'Frozen runtime')
    same(file_digest(root/'fixture.json'), plan['fixture_sha256'], 'Frozen fixture')
    sys.path.insert(0, str(root/'source/memory_system'))
    from app import cascade_rerank, config, fine_evidence, schemas, cross_encoder, answer_context
    modules = cascade_rerank, config, fine_evidence, schemas
    code, code_hash = selection_code(cascade_rerank)
    build_views = view_builder(config, cross_encoder, answer_context)
    fixture = {(r['conversation_id'], str(r['qa_id'])): r for r in read(root/'fixture.json')['requests']}
    results = {}
    for row in json_records(run/'completed/data/results/personamem-v2-32k.jsonl'):
        require(row['search_id'] not in results, 'Unique result')
        results[row['search_id']] = {k: row[k] for k in ('conversation_id','qa_id','user_id','packet_hash')}
    same(len(results), plan['questions'], 'Question count')
    same({(r['conversation_id'], str(r['qa_id'])) for r in results.values()}, set(fixture), 'Fixture population')
    seen, totals = set(), Counter()
    with args.output.open('x') as output, patch('socket.socket.connect', side_effect=AssertionError('Offline preflight')):
        for trace in json_records(run/'server-search-debug.jsonl'):
            sid = trace['search_id']
            require(sid in results and sid not in seen, 'Unique expected trace')
            seen.add(sid)
            result = results[sid]
            fixed = fixture[result['conversation_id'], str(result['qa_id'])]
            same(trace['packet_hash'], result['packet_hash'], 'Packet')
            same(trace['plan'], fixed['plan'], 'Planner')
            for key, value in fixed['request'].items():
                same(trace['requested_include_sensitive'] if key == 'include_sensitive' else trace[key], value, key)
            with patch.multiple(config, **trace['versions']['settings']):
                baseline, _, size = replay(trace, modules, code)
                recorded = trace['cascade']
                same(baseline['trace']['fine'], recorded['fine'], 'Baseline fine metadata')
                same(baseline['trace'].get('adaptive_admission'), recorded.get('adaptive_admission'), 'Baseline admission')
                same(baseline['admission_selection'], recorded['listwise']['selection'], 'Baseline shortlist')
                same(sorted(baseline['baseline_ids']), recorded['listwise']['baseline_candidate_ids'], 'Baseline core')
                same(sorted(baseline['carried_ids']), recorded['listwise']['carried_ids'], 'Baseline carried IDs')
                same([c['id'] for c in baseline['submitted']], recorded['listwise']['candidate_ids'], 'Baseline main')
                same(size, recorded['listwise']['input_bytes'], 'Baseline bytes')
                original = baseline['submitted']
                req, selection_plan = baseline['req'], baseline['plan']
                original_request = cascade_rerank.listwise_request(req, selection_plan, original)
                encoded, original_audit = local_candidate_ids.request(req, selection_plan, original, cascade_rerank)
                headroom = recorded['admission_budget']
                visible = {c['id']: fine_evidence.selection.fragments(c) for c in baseline['coarse']}
                excluded = {c['id'] for c in baseline['fine'] + original}
                ordered = sorted(baseline['coarse'], key=fine_evidence.key, reverse=True)
                feasible, audits, decisions = [], {}, []
                for rank, candidate in enumerate(ordered, 1):
                    mid = candidate['id']
                    if mid in excluded:
                        continue
                    score = candidate.get('_ce_score')
                    reason = None
                    details = {}
                    if type(score) not in (int, float) or not math.isfinite(score): reason = 'no_finite_ce'
                    elif not visible[mid]: reason = 'no_visible_source'
                    elif fine_evidence.dependencies(candidate): reason = 'not_singleton'
                    elif fine_evidence.selection.represented_by(candidate, original, visible): reason = 'represented_source'
                    elif len(original) + 1 > config.CASCADE_LLM_LIMIT + config.FINE_EXTRA_LIMIT: reason = 'count_budget'
                    else:
                        request, audit = local_candidate_ids.request(req, selection_plan, original+[candidate], cascade_rerank)
                        details = dict(trial_bytes=len(request['prompt'].encode()), reservation=request['reservation'])
                        if details['trial_bytes'] > config.RERANK_MAX_PROMPT_BYTES: reason = 'prompt_budget'
                        elif headroom['token_headroom'] is not None and request['reservation'] > headroom['token_headroom']:
                            reason = 'token_budget'
                        elif local_candidate_ids.missing_graph(original_audit['decoded_graph'], audit['decoded_graph']):
                            reason = 'original_graph_context_loss'
                        else:
                            feasible.append(candidate)
                            audits[mid] = dict(**details, ce_rank=rank, display_audit=audit)
                    decisions.append(dict(id=mid, ce_rank=rank, reason=reason or 'affordable', **details))
                query = recorded['cross_encoder']['query']
                views, members, _, omitted = build_views(feasible, trace['user_id'], query, max_sources=None)
                novel = novel_views(views, members, original, query, request_size=cross_encoder.request_size,
                    batch_limit=config.CE_BATCH_SIZE, byte_limit=config.CE_MAX_REQUEST_BYTES)
                nominees = [c for c in feasible if novel['members'].get(c['id'])]
                reservations = [original_request['reservation']] + [audits[c['id']]['reservation'] for c in nominees]
                bounds = [cross_encoder.request_size(query, [novel['views'][key]['text'] for key in batch])
                          for batch in novel['batches']]
                scheduling = schedule_bounds(bounds, main_reservation=max(reservations),
                    token_headroom=headroom['token_headroom'], call_headroom=headroom['call_headroom'],
                    deadline_seconds=headroom['deadline_seconds'], main_seconds=config.RERANK_DEADLINE_SECONDS+2,
                    ce_seconds=config.CE_DEADLINE_SECONDS)
                require(len(query.encode()) <= config.CE_MAX_DOCUMENT_BYTES, 'CE query bound')
                nomination_rows = [dict(id=c['id'], **audits[c['id']], cards=cards([c]),
                    candidate=copy.deepcopy(c), source_view_ids=novel['members'][c['id']]) for c in nominees]
                row = dict(**result, search_id=sid, query=trace['query'], options=trace['options'], ce_query=query,
                    original_main_ids=[c['id'] for c in original], original_bytes=size,
                    encoded_original_bytes=len(encoded['prompt'].encode()), original_display_audit=original_audit,
                    fine_cards=cards(baseline['fine']), original_cards=cards(original), decisions=decisions,
                    nominees=nomination_rows, views=novel['views'], batches=novel['batches'], batch_bounds=bounds,
                    omitted_source_views=omitted, represented_view_ids=novel['represented_view_ids'],
                    scheduling=scheduling, original_headroom=headroom, ce_model=recorded['cross_encoder']['model'],
                    ce_api_format=config.CE_API_FORMAT, no_op='Preserve original unencoded request when skipped')
                output.write(json.dumps(row, ensure_ascii=False)+'\n')
                output.flush()
                totals.update(questions=1, affordable_candidates=len(feasible), nominees=len(nominees),
                    qualified_questions=bool(nominees), eligible_questions=scheduling['eligible'],
                    eligible_nominees=len(nominees) if scheduling['eligible'] else 0,
                    eligible_calls=len(bounds) if scheduling['eligible'] else 0,
                    eligible_ce_input_bound=sum(bounds) if scheduling['eligible'] else 0)
                print(json.dumps(dict(qa_id=result['qa_id'], affordable=len(feasible), nominees=len(nominees),
                                      scheduling=scheduling)), flush=True)
    same(seen, set(results), 'All traces')
    write(args.output.with_suffix('.manifest.json'), dict(status='complete_fixed_inventory', summary=dict(totals),
          inventory_sha256=file_digest(args.output), selection_ast_sha256=code_hash, provider_calls=0))


def score(args):
    # No labels may be loaded until both complete inventories have been frozen.
    for cohort in ('regression','validation'):
        path = args.output.parent/(cohort+'-inventory.jsonl')
        manifest = read(path.with_suffix('.manifest.json'))
        same(manifest['status'], 'complete_fixed_inventory', 'Both cohorts frozen')
        same(file_digest(path), manifest['inventory_sha256'], 'Inventory digest')
    run = args.run.resolve()
    jobs = list(json_records(args.inventory))
    manifest = read(run/'isolated-manifest.json')
    conversations = {c['conversation_id']: dict(source_history=c['source_history'], qa=c['qa'])
                     for c in json_records(Path(manifest['data']))}
    needed = {int(j['qa_id']) for j in jobs}
    csv.field_size_limit(32*1024*1024)
    references = {}
    for index, row in enumerate(csv_records(args.gold_csv)):
        if index in needed: references[str(index)] = row
        if index >= max(needed): break
    raw = {}
    with sqlite3.connect((run/'memory.db').as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        cursor = db.execute('SELECT * FROM source_messages')
        while batch := cursor.fetchmany(256):
            for source in batch:
                source = dict(source)
                raw.setdefault(source['user_id'], {})[transport.identity(source)] = source
    stages = {(r['conversation_id'], str(r['qa_id'])):r for r in read(run/'funnel/stage-transport.json')['rows']}
    rows = []
    for job in jobs:
        conversation = conversations[job['conversation_id']]
        qa = next(q for q in conversation['qa'] if str(q['id']) == str(job['qa_id']))
        metadata = references[str(job['qa_id'])]
        same(qa['question'], job['query'], 'Prepared question')
        same(qa['options'], job['options'], 'Prepared options')
        same(transport.norm(ast.literal_eval(metadata['user_query'])['content']), transport.norm(job['query']), 'Gold question')
        same(metadata['correct_answer'], qa['answer'], 'Gold answer identity')
        require(conversation['source_history'] in (metadata['chat_history_32k_link'], metadata['chat_history_128k_link']), 'Gold history')
        try: snippet = json.loads(metadata['related_conversation_snippet'])
        except ValueError: snippet = ast.literal_eval(metadata['related_conversation_snippet'])
        turns = [transport.norm(s['content']) for s in snippet if s['role']=='user']
        def coverage(cards_):
            return [r['full'] for r in transport.turn_coverage(turns, cards_, raw[job['user_id']])]
        baseline = coverage(job['original_cards'])
        stage = stages[job['conversation_id'], str(job['qa_id'])]
        same(baseline, [r['full'] for r in stage['visible']['listwise_input']['turn_coverage']], 'Baseline main recall')
        same(coverage(job['fine_cards']), [r['full'] for r in stage['visible']['fine']['turn_coverage']], 'Baseline fine recall')
        trials = []
        for nominee in job['nominees']:
            extra = [dict(c, id='addition:'+c['id']) for c in nominee['cards']]
            after = coverage(job['original_cards']+extra)
            losses = [i for i,(a,b) in enumerate(zip(baseline, after, strict=True)) if a and not b]
            require(not losses, 'Append cannot lose an original reference')
            gains = [i for i,(a,b) in enumerate(zip(baseline, after, strict=True)) if b and not a]
            trials.append(dict(id=nominee['id'], gains=gains, ce_rank=nominee['ce_rank']))
        maximum = max((len(t['gains']) for t in trials), default=0)
        eligible = job['scheduling']['eligible']
        rows.append(dict(qa_id=job['qa_id'], conversation_id=job['conversation_id'], baseline=baseline,
            scheduling=job['scheduling'], trials=trials, max_single_addition_gain=maximum,
            eligible_max_single_addition_gain=maximum if eligible else 0))
    summary = dict(questions=len(rows), baseline_questions=sum(any(r['baseline']) for r in rows),
        baseline_turns=sum(sum(r['baseline']) for r in rows),
        potential_questions_before_scheduling=sum(r['max_single_addition_gain']>0 for r in rows),
        potential_questions=sum(r['eligible_max_single_addition_gain']>0 for r in rows),
        max_single_addition_turns=sum(r['eligible_max_single_addition_gain'] for r in rows),
        qualified_nominees=sum(len(r['trials']) for r in rows),
        skipped_reasons=dict(Counter(r['scheduling']['reason'] for r in rows if not r['scheduling']['eligible'])))
    write(args.output, dict(status='complete_single_addition_upper_bound', summary=summary, rows=rows,
        provider_calls=0, runtime_changed=False, limitation='Oracle single-addition bound, not measured selector gain. '
        'Different candidates cannot share the one seat. Same cohorts were previously inspected.',
        inventory_sha256=file_digest(args.inventory)))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare','score'), required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--gold-csv', type=Path)
    args = parser.parse_args()
    (prepare if args.phase == 'prepare' else score)(args)
