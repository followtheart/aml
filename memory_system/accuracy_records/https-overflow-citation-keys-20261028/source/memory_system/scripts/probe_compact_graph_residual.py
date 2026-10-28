"""Replay compact graph metadata plus one coarse outsider in the original main budget.

Freeze both complete cohorts before scoring. Preserve complete original fine
and main input; no providers, deletion, new calls, truncation or budget growth.
"""
import argparse
import ast
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

from batched_records import json_records, csv_records, file_digest
from audit_retained_porter_https import require, same, selection_code
from probe_assistant_admission_order import replay, cards
import audit_https_funnel as transport
import main_residual_admission as residual
import compact_graph_metadata


def read(path):
    data = bytearray()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            data.extend(block)
    return json.loads(data)


def constraint_missing(before, after):
    def identity(card):
        return tuple(card.get(key) for key in ('request_id', 'message_index', 'source_event_id', 'role'))
    return [card for card in before if not any(identity(card) == identity(other)
            and card['text'] in other['text'] for other in after)]


def review_unit(candidate):
    return {key: candidate.get(key) for key in ('id', 'type', 'content', '_rank_text', '_packet_item',
        '_ce_score', 'evidence', 'polarity', 'epistemic_status', 'resolution_status', 'valid_from', 'valid_to', 'state')}


def run_replay(args):
    run = args.run.resolve()
    root = run.parent
    plan, manifest = read(root / 'plan.json'), read(run / 'isolated-manifest.json')
    require(run.name == 'control', 'Use complete current control')
    require(manifest['server_exit_code'] in (0, -15) and
            manifest['client_exit_code'] == manifest['audit_exit_code'] == 0, 'Arm not terminal')
    joint = read(root.parent / 'final-decision.json')
    require(joint['decision'] == 'reject_answer_or_reference_or_error_gate', 'Require completed prior experiment')
    input_paths = [run / name for name in ('isolated-manifest.json', 'funnel/stage-transport.json',
                  'completed/data/results/personamem-v2-32k.jsonl')]
    for path in input_paths:
        same(file_digest(path), joint['input_sha256'][str(path)], 'Prior finalizer input')
    for name, digest in plan['source_files'].items():
        same(file_digest(root / 'source' / name), digest, 'Frozen runtime source')
    sys.path.insert(0, str(root / 'source/memory_system'))
    from app import cascade_rerank, config, fine_evidence, schemas, answer_choice
    modules = cascade_rerank, config, fine_evidence, schemas
    code, code_hash = selection_code(cascade_rerank)
    fixture = {(row['conversation_id'], str(row['qa_id'])): row for row in read(root / 'fixture.json')['requests']}
    results = {row['search_id']: {key: row[key] for key in ('conversation_id', 'qa_id', 'user_id', 'packet_hash')}
               for row in json_records(input_paths[-1])}
    same(len(results), len(fixture), 'Complete result count')
    same({(r['conversation_id'], str(r['qa_id'])) for r in results.values()}, set(fixture), 'Complete identities')
    seen, additions = set(), 0
    with args.output.open('x') as output, patch('socket.socket.connect', side_effect=AssertionError('Offline replay')):
        for trace in json_records(run / 'server-search-debug.jsonl'):
            sid = trace['search_id']
            require(sid in results and sid not in seen, 'Unexpected trace')
            seen.add(sid)
            result = results[sid]
            request = fixture[result['conversation_id'], str(result['qa_id'])]
            same(trace['packet_hash'], result['packet_hash'], 'Packet identity')
            same(trace['plan'], request['plan'], 'Planner')
            for key, value in request['request'].items():
                same(trace['requested_include_sensitive'] if key == 'include_sensitive' else trace[key], value, key)
            cfg = trace['versions']['settings']
            require(not cfg['RECALL_PORTER_SUPPLEMENT'] and not cfg['PACKET_ENDPOINT_FINE_COMPLETION'],
                    'Expected original upstream and completion disabled')
            with patch.multiple(config, **cfg):
                baseline, _, original_bytes = replay(trace, modules, code)
                recorded = trace['cascade']
                same(baseline['trace']['fine'], recorded['fine'], 'Original fine and metadata')
                same(baseline['trace'].get('adaptive_admission'), recorded.get('adaptive_admission'), 'Original admission')
                same(baseline['admission_selection'], recorded['listwise']['selection'], 'Original shortlist')
                same(sorted(baseline['baseline_ids']), recorded['listwise']['baseline_candidate_ids'], 'Original main core')
                same(sorted(baseline['carried_ids']), recorded['listwise']['carried_ids'], 'Original carried IDs')
                same([c['id'] for c in baseline['submitted']], recorded['listwise']['candidate_ids'], 'Original main order')
                same(original_bytes, recorded['listwise']['input_bytes'], 'Original main bytes')
                compact_render = compact_graph_metadata.renderer(cascade_rerank.prompts.render)
                def request_for(rows):
                    original_request = cascade_rerank.listwise_request(baseline['req'], baseline['plan'], rows)
                    with patch.object(cascade_rerank.prompts, 'render', compact_render):
                        compact_request = cascade_rerank.listwise_request(baseline['req'], baseline['plan'], rows)
                    same({k:v for k,v in compact_request.items() if k not in ('prompt','reservation')},
                         {k:v for k,v in original_request.items() if k not in ('prompt','reservation')},
                         'Same system/schema/output protocol for identical candidate input')
                    saved = len(original_request['prompt'].encode()) - len(compact_request['prompt'].encode())
                    require(saved >= 0, 'Nonincreasing exact prompt bytes')
                    same(original_request['reservation'] - compact_request['reservation'], saved,
                         'Only prompt bytes change the token reservation')
                    return compact_request
                compact_original_bytes = len(request_for(baseline['submitted'])['prompt'].encode()) if baseline['submitted'] else 0
                submitted, proposal = residual.propose(baseline['coarse'], baseline['fine'],
                    baseline['submitted'], helpers=fine_evidence, request_for=request_for,
                    count_limit=cfg['CASCADE_LLM_LIMIT'] + cfg['FINE_EXTRA_LIMIT'],
                    byte_limit=cfg['RERANK_MAX_PROMPT_BYTES'],
                    token_headroom=recorded['admission_budget']['token_headroom'])
                changed = dict(baseline, submitted=submitted)
                admission = proposal['decisions']
                proposed_bytes = len(request_for(submitted)['prompt'].encode()) if submitted else 0
                same(submitted[:len(baseline['submitted'])], baseline['submitted'], 'Complete main prefix')
                require(len(submitted) <= cfg['CASCADE_LLM_LIMIT'] + cfg['FINE_EXTRA_LIMIT'], 'Main count')
                require(proposed_bytes <= cfg['RERANK_MAX_PROMPT_BYTES'], 'Main bytes')
                headroom = recorded['admission_budget']['token_headroom']
                require(not submitted or headroom is None or request_for(submitted)['reservation'] <= headroom,
                        'Main tokens')
                same(changed['fine_selection']['adaptive']['core_ids'],
                     baseline['fine_selection']['adaptive']['core_ids'], 'Protected core order')
                same(Counter(c['type'] for c in changed['fine']), Counter(c['type'] for c in baseline['fine']),
                     'Type distribution')
                coarse = {c['id']: c for c in baseline['coarse']}
                for candidate in changed['fine'] + changed['submitted']:
                    for field in ('content', 'type', '_rank_text', '_packet_item', '_ce_score', 'evidence',
                                  'polarity', 'epistemic_status', 'resolution_status', 'valid_from', 'valid_to', 'state'):
                        same(candidate.get(field), coarse[candidate['id']].get(field), 'Unchanged candidate ' + field)
                before, after = ([c['id'] for c in env['submitted']] for env in (baseline, changed))
                if proposal['addition'] is None:
                    same(after, before, 'No-op main order')
                    same(proposed_bytes, compact_original_bytes, 'No-addition compact main bytes')
                else:
                    additions += 1
                constraints = {}
                for stage in ('fine', 'submitted'):
                    before_rules = list(answer_choice.build_catalog([c['_packet_item'] for c in baseline[stage]])[1].values())
                    after_rules = list(answer_choice.build_catalog([c['_packet_item'] for c in changed[stage]])[1].values())
                    constraints[stage] = dict(original=before_rules, proposed=after_rules,
                                             missing=constraint_missing(before_rules, after_rules))
                addition = proposal['addition']
                row = dict(**result, query=trace['query'], options=trace['options'], proposal=proposal,
                    original_main_ids=before, proposed_main_ids=after,
                    original_bytes=original_bytes, compact_original_bytes=compact_original_bytes,
                    graph_bytes_saved=original_bytes-compact_original_bytes, proposed_bytes=proposed_bytes, admission=admission,
                    fine_cards=cards(baseline['fine']), proposed_fine_cards=cards(changed['fine']),
                    original_cards=cards(baseline['submitted']), proposed_cards=cards(changed['submitted']),
                    constraints=constraints,
                    changed_units=(dict(added=review_unit(coarse[addition['id']])) if addition else {}),
                    removed_main=[review_unit(c) for c in baseline['submitted'] if c['id'] not in after],
                    added_main=[review_unit(c) for c in changed['submitted'] if c['id'] not in before])
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
            print(json.dumps(dict(qa_id=result['qa_id'], baseline_exact=True, addition=bool(proposal['addition']))), flush=True)
    same(seen, set(results), 'Complete traces')
    report = dict(status='complete_fixed_selections', questions=len(seen), additions=additions, provider_calls=0,
        selection_ast_sha256=code_hash, selection_file_sha256=file_digest(args.output))
    args.output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2) + '\n')


def run_score(args):
    for cohort in ('regression', 'validation'):
        path = args.output.parent / (cohort + '-selections.jsonl')
        manifest = read(path.with_suffix('.manifest.json'))
        require(manifest['status'] == 'complete_fixed_selections', 'Both selections must finish before scoring')
        same(file_digest(path), manifest['selection_file_sha256'], 'Frozen selections')
    run = args.run.resolve()
    manifest = read(run / 'isolated-manifest.json')
    jobs = list(json_records(args.selections))
    conversations = {c['conversation_id']: dict(source_history=c['source_history'], qa=c['qa'])
                     for c in json_records(Path(manifest['data']))}
    needed = {int(job['qa_id']) for job in jobs}
    csv.field_size_limit(32 * 1024 * 1024)
    references = {}
    for index, row in enumerate(csv_records(args.gold_csv)):
        if index in needed:
            references[str(index)] = row
        if index >= max(needed):
            break
    raw = {}
    with sqlite3.connect((run / 'memory.db').as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        cursor = db.execute('SELECT * FROM source_messages')
        while batch := cursor.fetchmany(256):
            for source in batch:
                source = dict(source)
                raw.setdefault(source['user_id'], {})[transport.identity(source)] = source
    stages = {(r['conversation_id'], str(r['qa_id'])): r for r in read(run / 'funnel/stage-transport.json')['rows']}
    for job in jobs:
        conversation = conversations[job['conversation_id']]
        qa = next(q for q in conversation['qa'] if str(q['id']) == str(job['qa_id']))
        metadata = references[str(job['qa_id'])]
        same(qa['question'], job['query'], 'Prepared question')
        same(qa['options'], job['options'], 'Prepared options')
        same(transport.norm(ast.literal_eval(metadata['user_query'])['content']), transport.norm(job['query']), 'Gold query')
        same(metadata['correct_answer'], qa['answer'], 'Gold answer identity')
        require(conversation['source_history'] in (metadata['chat_history_32k_link'], metadata['chat_history_128k_link']),
                'Gold history identity')
        try:
            snippet = json.loads(metadata['related_conversation_snippet'])
        except ValueError:
            snippet = ast.literal_eval(metadata['related_conversation_snippet'])
        turns = [transport.norm(s['content']) for s in snippet if s['role'] == 'user']
        coverage = {name: [r['full'] for r in transport.turn_coverage(turns, job.pop(name + '_cards'), raw[job['user_id']])]
                    for name in ('fine', 'proposed_fine', 'original', 'proposed')}
        stage = stages[job['conversation_id'], str(job['qa_id'])]
        for name, recorded in (('fine', 'fine'), ('original', 'listwise_input')):
            same(coverage[name], [r['full'] for r in stage['visible'][recorded]['turn_coverage']], 'Original coverage')
        job['reference_turns'] = coverage
        for name, before, after in (('fine', 'fine', 'proposed_fine'), ('main', 'original', 'proposed')):
            job[name + '_turn_gains'] = [i for i, (a, b) in enumerate(zip(coverage[before], coverage[after], strict=True)) if b and not a]
            job[name + '_turn_losses'] = [i for i, (a, b) in enumerate(zip(coverage[before], coverage[after], strict=True)) if a and not b]
    summary = dict(questions=len(jobs), appended=sum(bool(j['proposal']['addition']) for j in jobs),
        coverage={name: dict(questions=sum(any(j['reference_turns'][name]) for j in jobs),
                            turns=sum(sum(j['reference_turns'][name]) for j in jobs))
                  for name in ('fine', 'proposed_fine', 'original', 'proposed')},
        changes={name: [dict(qa_id=j['qa_id'], turns=j[name]) for j in jobs if j[name]]
                 for name in ('fine_turn_gains', 'fine_turn_losses', 'main_turn_gains', 'main_turn_losses')},
        missing_constraints={stage: [dict(qa_id=j['qa_id'], cards=j['constraints'][stage]['missing'])
                                    for j in jobs if j['constraints'][stage]['missing']]
                             for stage in ('fine', 'submitted')})
    report = dict(status='complete_offline_compact_graph_residual_admission', summary=summary, rows=jobs, provider_calls=0,
                  production_changed=False, limitation=__doc__)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('replay', 'score'), required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selections', type=Path)
    parser.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1] / 'data/raw/personamem-v2.csv')
    args = parser.parse_args()
    (run_replay if args.phase == 'replay' else run_score)(args)
