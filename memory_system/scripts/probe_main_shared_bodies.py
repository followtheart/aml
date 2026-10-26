"""Replay exact repeated-body sharing at the original main-input budget.

Reconstruct all original fine/main choices before changing only the display of
repeated literal candidate-body strings. Keep original coarse/CE/fine, all source
quotes/roles/times, body labels, support IDs and request limits. Run actual frozen
main admission, allowing real membership changes; gold is loaded afterward.
No provider calls or final-packet/answer prediction. Decode validation does not
establish model comprehension of the altered display; HTTPS would be required.
"""
import argparse
import ast
import csv
import json
from pathlib import Path
import sys
from unittest.mock import patch

from audit_retained_porter_https import records, require, same, sha, selection_code
from probe_assistant_admission_order import replay, cards
import audit_https_funnel as transport
import listwise_shared_bodies as encoding


def compact_replay(trace, modules, code, baseline):
    from app import listwise_sources
    cascade, config, _, _ = modules
    original = listwise_sources.shared_candidates
    with patch.object(listwise_sources, 'shared_candidates', side_effect=encoding.renderer(original)):
        fixed = cascade.listwise_request(baseline['req'], baseline['plan'], baseline['submitted'])
        changed, admission, size = replay(trace, modules, code)
    original_bytes = trace['cascade']['listwise']['input_bytes']
    fixed_bytes = len(fixed['prompt'].encode())
    require(fixed_bytes <= original_bytes, 'Fixed-input bytes increased')
    return changed, dict(fixed_savings=original_bytes-fixed_bytes, fixed_input_bytes=fixed_bytes,
                         admission=admission), size


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
                changed, ordering, changed_bytes = compact_replay(t, modules, code, baseline)
                same(changed['trace']['fine'], baseline['trace']['fine'], 'Preserved complete fine pool')
                before, after = ([x['id'] for x in env['submitted']] for env in (baseline, changed))
                jobs.append(dict(**result, query=t['query'], options=t['options'],
                    ordering=ordering, original_main_ids=before, proposed_main_ids=after,
                    original_bytes=original_bytes, proposed_bytes=changed_bytes,
                    fine_cards=cards(baseline['fine']), original_cards=cards(baseline['submitted']),
                    proposed_cards=cards(changed['submitted']),
                    new_main=[dict(candidate_id=x['id'], rank_text=x.get('_rank_text', x.get('content', '')))
                              for x in changed['submitted'] if x['id'] not in before]))
            print(json.dumps(dict(qa_id=result['qa_id'],baseline_exact=True,fixed_savings=ordering['fixed_savings'],
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
        fixed_input_saved_bytes=sum(r['ordering']['fixed_savings'] for r in rows),
        fixed_input_changed_questions=sum(r['ordering']['fixed_savings'] > 0 for r in rows),
        original_main_candidates=sum(len(r['original_main_ids']) for r in rows),
        proposed_main_candidates=sum(len(r['proposed_main_ids']) for r in rows),
        main_changed_questions=sum(r['original_main_ids'] != r['proposed_main_ids'] for r in rows),
        original_reference_questions=sum(any(r['reference_turns']['original']) for r in rows),
        proposed_reference_questions=sum(any(r['reference_turns']['proposed']) for r in rows),
        original_reference_turns=sum(sum(r['reference_turns']['original']) for r in rows),
        proposed_reference_turns=sum(sum(r['reference_turns']['proposed']) for r in rows),
        turn_losses=[dict(qa_id=r['qa_id'],turn_indices=r['turn_losses']) for r in rows if r['turn_losses']],
        turn_gains=[dict(qa_id=r['qa_id'],turn_indices=r['turn_gains']) for r in rows if r['turn_gains']])
    inputs = [Path(__file__), Path(encoding.__file__), Path(__file__).with_name('probe_assistant_admission_order.py'),
              Path(__file__).with_name('audit_retained_porter_https.py'), Path(__file__).with_name('audit_https_funnel.py'),
              root / 'plan.json', root / 'fixture.json', root / 'retained-selection-audit.json',
              args.gold_csv, Path(manifest['data']), run / 'server-memory-debug.jsonl', *paths]
    report = dict(status='complete_offline_shared_body_replay', summary=summary, rows=rows,
                  provider_calls=0, production_changed=False, selection_ast_sha256=code_sha,
                  input_sha256={str(p): sha(p) for p in inputs}, limitation=__doc__)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1] / 'data/raw/personamem-v2.csv')
    main(parser.parse_args())
