"""Replay stable deferral of weak assistant lexical extras at main admission.

Freeze the complete original fine pool, its core, source views, CE scores and
all request limits. Reorder only existing singleton lexical additions whose
full-pool CE rank is below the existing fine cutoff and whose lexical matches
come exclusively from assistant text. Every addition remains eligible later.
References are loaded only after both selections are fixed. No provider calls,
final-packet counterfactual, answer estimate or production setting change.
"""
import argparse
import ast
import copy
import csv
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from audit_retained_porter_https import records, require, same, sha, selection_code
import audit_https_funnel as transport


def deferred_ids(coarse, requirements, adaptive, cutoff, fine_evidence):
    """Use visible-source matches and existing ranks, without reference labels."""
    visible = {c['id']: fine_evidence.selection.fragments(c) for c in coarse}
    facets = requirements + adaptive.get('context_facets', [])
    sparse = fine_evidence.sparse_matches(coarse, facets, visible)
    ordered = sorted([c for c in coarse if c.get('_ce_score') is not None],
                     key=fine_evidence.key, reverse=True)
    ranks = {c['id']: i + 1 for i, c in enumerate(ordered)}
    pool = {c['id']: c for c in coarse}
    eligible = []
    for addition in adaptive.get('additions', []):
        mid = addition['candidate_id']
        candidate = pool[mid]
        matches = fine_evidence.relevance(candidate, facets, visible[mid])
        for rid, value in sparse[mid].items():
            matches[rid] = max(matches.get(rid, (False, False, 0)), value)
        assistant_only = (bool(matches) and not any(v[0] for v in matches.values())
                          and all(key[1] in ('user', 'assistant') for key, _ in visible[mid]))
        if (addition['reason'] in ('novel_visible_evidence', 'sparse_topic_evidence')
                and assistant_only and ranks.get(mid, 0) > cutoff
                and len(addition['ids']) == len(addition['group_ids']) == 1
                and not fine_evidence.dependencies(candidate)
                and mid not in adaptive.get('core_ids', [])):
            eligible.append(mid)
    return eligible


def replay(trace, modules, code, defer=()):
    cascade, config, fine_evidence, schemas = modules
    c = trace['cascade']
    pool = {v['id']: v for v in trace['ranked']}
    coarse = [copy.deepcopy(pool[mid]) for mid in c['coarse']['selected_ids']]
    scores = {}
    for candidate in coarse:
        score = candidate.get('_ce_score')
        require(score is None or type(score) in (int, float) and math.isfinite(score), 'Invalid CE')
        if score is None:
            candidate.pop('_final', None)
        else:
            candidate['_final'] = scores[candidate['id']] = score
    require(set(scores) == {mid for batch in c['cross_encoder']['batches']
                           if batch['status'] == 'ok' for mid in batch['candidate_ids']}, 'CE batch mismatch')
    requirements = [{k: r[k] for k in ('id', 'text', 'origin') if k in r}
                    for r in trace['coverage_manifest']['coverage']['requirements']]
    plan = dict(copy.deepcopy(trace['plan']), _anchor=trace['anchor_time'],
                _include_history=trace['include_history'], _include_sensitive=trace['include_sensitive'],
                _query_specs=copy.deepcopy(trace['query_specs']), _coverage_requirements=requirements,
                _fusion_edges=copy.deepcopy(trace['fusion_edges']),
                _fusion_triples=copy.deepcopy(trace['fusion_triples']), _porter_supplement_ids=[])
    request = schemas.SearchRequest(user_id=trace['user_id'], query=trace['query'], options=trace['options'],
        top_k=trace['top_k'], include_sensitive=trace['requested_include_sensitive'],
        include_history=trace['include_history'], reference_time=trace['reference_time'],
        evidence_token_budget=trace['evidence_token_budget'])
    b = c['admission_budget']
    limits = None if b['token_headroom'] is None else SimpleNamespace(
        max_tokens=b['token_headroom'], tokens=0, reserved_tokens=0,
        max_calls=b['call_headroom'], calls=0, reserved_calls=0,
        deadline=time.monotonic() + b['deadline_seconds'])
    replay_trace = dict(coarse=copy.deepcopy(c['coarse']), omitted=[], reservation_missing=[])
    env = dict(vars(cascade), coarse=coarse, coarse_selection=copy.deepcopy(c['coarse']['selection']),
               scores=scores, trace=replay_trace, req=request, plan=plan, requirements=requirements, limits=limits)
    original = fine_evidence.admit_additions
    ordering = {}

    def admit(submitted, candidates, additions, **kwargs):
        # Previously admitted candidates and dependency groups retain priority.
        delayed = set(defer) - {v['id'] for v in submitted}
        ordered = ([a for a in additions if a['candidate_id'] not in delayed]
                   + [a for a in additions if a['candidate_id'] in delayed])
        ordering.update(before=[a['candidate_id'] for a in additions],
                        after=[a['candidate_id'] for a in ordered], delayed_ids=sorted(delayed),
                        initial_submitted_ids=[v['id'] for v in submitted])
        return original(submitted, candidates, ordered, **kwargs)

    with patch.object(fine_evidence, 'admit_additions', side_effect=admit):
        exec(code, env)
    rendered = cascade.listwise_request(request, plan, env['submitted']) if env['submitted'] else None
    size = len(rendered['prompt'].encode()) if rendered else 0
    require(size <= config.RERANK_MAX_PROMPT_BYTES, 'Main byte cap')
    require(not rendered or b['token_headroom'] is None or rendered['reservation'] <= b['token_headroom'],
            'Main token headroom')
    require(len(env['submitted']) <= config.CASCADE_LLM_LIMIT + config.FINE_EXTRA_LIMIT, 'Main count cap')
    return env, ordering, size


def cards(candidates):
    return [dict(s, id=f'{i}:{j}', text=s.get('content', ''))
            for i, c in enumerate(candidates) for j, s in enumerate(c['_packet_item'].get('sources', []))]


def main(args):
    run = args.run.resolve()
    require(run.name == 'control', 'Initial diagnostic uses complete current controls')
    root = run.parent
    require(not args.output.exists(), 'Refusing to overwrite replay')
    plan = json.loads((root / 'plan.json').read_text())
    manifest = json.loads((run / 'isolated-manifest.json').read_text())
    require(manifest['client_exit_code'] == manifest['audit_exit_code'] == 0
            and manifest['server_exit_code'] in (0, -15), 'Require terminal stage-audited control')
    process = Path('/proc') / str(manifest['server_pid']) / 'cmdline'
    require(not process.exists() or str(root).encode() not in process.read_bytes(), 'Service still live')
    audited = json.loads((root / 'retained-selection-audit.json').read_text())
    same(audited['status'], 'passed', 'Original selection audit')
    same(sha(root / 'plan.json'), audited['plan_sha256'], 'Plan identity')
    for name, digest in plan['source_files'].items():
        same(sha(root / 'source' / name), digest, 'Frozen source ' + name)
    paths = [run / name for name in ('server-search-debug.jsonl', 'funnel/stage-transport.json',
             'isolated-manifest.json', 'completed/data/results/personamem-v2-32k.jsonl')]
    for path in paths:
        same(sha(path), audited['input_sha256'][str(path)], 'Audited input ' + str(path))
    same(sha(Path(manifest['data'])), plan['data_sha256'], 'Prepared data')
    sys.path.insert(0, str(root / 'source/memory_system'))
    from app import cascade_rerank, config, fine_evidence, schemas
    modules = cascade_rerank, config, fine_evidence, schemas
    code, code_sha = selection_code(cascade_rerank)
    same(code_sha, audited['selection_ast_sha256'], 'Frozen selection block')
    results = {}
    for r in records(paths[-1]):
        require(r['search_id'] not in results, 'Duplicate result')
        results[r['search_id']] = {k: r[k] for k in ('qa_id', 'conversation_id', 'user_id', 'packet_hash')}
    fixture = json.loads((root / 'fixture.json').read_text())
    same(sha(root / 'fixture.json'), plan['fixture_sha256'], 'Fixture')
    expected = {(r['conversation_id'], r['qa_id']): r for r in fixture['requests']}
    same({(r['conversation_id'], r['qa_id']) for r in results.values()}, set(expected), 'Question identities')
    same(len(results), plan['questions'], 'Complete question count')
    inventory = json.loads(args.inventory.read_text())
    same(inventory['input_sha256'][str(paths[0])], sha(paths[0]), 'Inventory trace')
    inventory_rows = {r['search_id']: r for r in inventory['rows']}
    jobs, seen = [], set()
    with patch('socket.socket.connect', side_effect=AssertionError('Offline admission replay')):
        for t in records(paths[0]):
            sid = t['search_id']
            require(sid in results and sid not in seen, 'Unexpected trace identity')
            seen.add(sid)
            result = results[sid]
            fixture_row = expected[result['conversation_id'], result['qa_id']]
            same(t['packet_hash'], result['packet_hash'], 'Packet')
            same(t['plan'], fixture_row['plan'], 'Frozen planner')
            for key, value in fixture_row['request'].items():
                same(t.get('requested_include_sensitive') if key == 'include_sensitive' else t[key],
                     value, 'Request ' + key)
            cfg = t['versions']['settings']
            require(not cfg['RECALL_PORTER_SUPPLEMENT'], 'Expected original recall')
            with patch.multiple(config, **cfg):
                baseline, original_order, original_bytes = replay(t, modules, code)
                c = t['cascade']
                same(baseline['trace']['fine'], c['fine'], 'Baseline fine')
                same(baseline['trace'].get('adaptive_admission'), c.get('adaptive_admission'), 'Baseline admission')
                same(baseline['admission_selection'], c['listwise']['selection'], 'Baseline shortlist')
                same(sorted(baseline['baseline_ids']), c['listwise']['baseline_candidate_ids'], 'Baseline core IDs')
                same(sorted(baseline['carried_ids']), c['listwise']['carried_ids'], 'Baseline carried IDs')
                same([x['id'] for x in baseline['submitted']], c['listwise']['candidate_ids'], 'Baseline main order')
                same(original_bytes, c['listwise']['input_bytes'], 'Baseline prompt bytes')
                eligible = deferred_ids(baseline['coarse'], baseline['requirements'],
                    baseline['fine_selection']['adaptive'], cfg['CASCADE_FINE_LIMIT'], fine_evidence)
                same(set(eligible), {a['candidate_id'] for a in inventory_rows[sid]['additions']
                                    if a['diagnostic_population']}, 'Independent inventory eligibility')
                changed, ordering, changed_bytes = replay(t, modules, code, eligible)
                same(changed['trace']['fine'], baseline['trace']['fine'], 'Preserved complete fine pool')
                same(ordering['initial_submitted_ids'], original_order['initial_submitted_ids'], 'Preserved initial main')
                require(set(ordering['initial_submitted_ids']) <= {x['id'] for x in changed['submitted']},
                        'Original admitted core lost')
                before, after = ([x['id'] for x in env['submitted']] for env in (baseline, changed))
                if not eligible:
                    same(after, before, 'Untriggered main order')
                    same(changed_bytes, original_bytes, 'Untriggered bytes')
                jobs.append(dict(**result, query=t['query'], options=t['options'], eligible_ids=eligible,
                    ordering=ordering, original_main_ids=before, proposed_main_ids=after,
                    original_bytes=original_bytes, proposed_bytes=changed_bytes,
                    fine_cards=cards(baseline['fine']), original_cards=cards(baseline['submitted']),
                    proposed_cards=cards(changed['submitted']),
                    new_main=[dict(candidate_id=x['id'], rank_text=x.get('_rank_text', x.get('content', '')))
                              for x in changed['submitted'] if x['id'] not in before]))
            print(json.dumps(dict(qa_id=result['qa_id'],baseline_exact=True,eligible=len(eligible),
                                  main_changed=before != after)), flush=True)
    same(seen, set(results), 'Full trace coverage')
    # Only the already-fixed selections below can see reference information.
    conversations = {r['conversation_id']: r for r in records(Path(manifest['data']))}
    wanted = {int(j['qa_id']) for j in jobs}
    references = {}
    csv.field_size_limit(32 * 1024 * 1024)
    with args.gold_csv.open() as stream:
        for index, metadata in enumerate(csv.DictReader(stream)):
            if index in wanted:
                references[str(index)] = metadata
            if index >= max(wanted):
                break
    raw = {}
    for event in records(run / 'server-memory-debug.jsonl'):
        raw.setdefault(event['user_id'], {}).update({transport.identity(s): s for s in event['source_messages']})
    stages = {(r['conversation_id'], r['qa_id']): r
              for r in json.loads(paths[1].read_text())['rows']}
    rows = []
    for job in jobs:
        conversation = conversations[job['conversation_id']]
        qa = next(q for q in conversation['qa'] if str(q['id']) == job['qa_id'])
        metadata = references[job['qa_id']]
        same(qa['question'], job['query'], 'Prepared query')
        same(qa['options'], job['options'], 'Prepared options')
        same(transport.norm(ast.literal_eval(metadata['user_query'])['content']), transport.norm(job['query']), 'Gold query')
        same(metadata['correct_answer'], qa['answer'], 'Gold identity')
        require(conversation['source_history'] in (metadata['chat_history_32k_link'], metadata['chat_history_128k_link']),
                'Gold history identity')
        try:
            snippet = json.loads(metadata['related_conversation_snippet'])
        except ValueError:
            snippet = ast.literal_eval(metadata['related_conversation_snippet'])
        turns = [transport.norm(s['content']) for s in snippet if s['role'] == 'user']
        coverage = {name: [v['full'] for v in transport.turn_coverage(turns, job.pop(name + '_cards'), raw[job['user_id']])]
                    for name in ('fine', 'original', 'proposed')}
        stage = stages[job['conversation_id'], job['qa_id']]
        for name, source in (('fine', 'fine'), ('original', 'listwise_input')):
            same(coverage[name], [v['full'] for v in stage['visible'][source]['turn_coverage']], 'Original transport ' + name)
        job['reference_turns'] = coverage
        job['turn_losses'] = [i for i, (a, b) in enumerate(zip(coverage['original'], coverage['proposed'], strict=True)) if a and not b]
        job['turn_gains'] = [i for i, (a, b) in enumerate(zip(coverage['original'], coverage['proposed'], strict=True)) if b and not a]
        rows.append(job)
    summary = dict(questions=len(rows), baseline_exact_questions=len(rows),
        eligible_questions=sum(bool(r['eligible_ids']) for r in rows), eligible_instances=sum(len(r['eligible_ids']) for r in rows),
        main_changed_questions=sum(r['original_main_ids'] != r['proposed_main_ids'] for r in rows),
        original_reference_questions=sum(any(r['reference_turns']['original']) for r in rows),
        proposed_reference_questions=sum(any(r['reference_turns']['proposed']) for r in rows),
        original_reference_turns=sum(sum(r['reference_turns']['original']) for r in rows),
        proposed_reference_turns=sum(sum(r['reference_turns']['proposed']) for r in rows),
        turn_losses=[dict(qa_id=r['qa_id'],turn_indices=r['turn_losses']) for r in rows if r['turn_losses']],
        turn_gains=[dict(qa_id=r['qa_id'],turn_indices=r['turn_gains']) for r in rows if r['turn_gains']])
    inputs = [Path(__file__), Path(__file__).with_name('audit_retained_porter_https.py'),
              root / 'plan.json', root / 'fixture.json', root / 'retained-selection-audit.json',
              args.inventory, args.gold_csv, Path(manifest['data']), run / 'server-memory-debug.jsonl', *paths]
    report = dict(status='complete_offline_admission_replay', summary=summary, rows=rows,
                  provider_calls=0, production_changed=False, selection_ast_sha256=code_sha,
                  input_sha256={str(p): sha(p) for p in inputs}, limitation=__doc__)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1] / 'data/raw/personamem-v2.csv')
    main(parser.parse_args())
