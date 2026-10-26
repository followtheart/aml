"""Offline single-seat fine swap from a weak assistant-only lexical addition.

Use original complete fine selection, core, actual CE scores and all budgets.
Replace at most one singleton assistant-only optional addition below the core
CE cutoff with a missing source-novel singleton within the effective fine CE
cutoff and strictly higher CE score. Keep all other original fine candidates.
No gold-based eligibility, provider calls, final-packet or answer claims.
"""
import argparse
import ast
import copy
import csv
import json
from pathlib import Path
import sys
from unittest.mock import patch

from audit_retained_porter_https import records, require, same, sha, selection_code
from probe_assistant_admission_order import replay, deferred_ids, cards
import audit_https_funnel as transport


def swap_replay(trace, modules, code, baseline, eligible):
    cascade, config, fine_evidence, schemas = modules
    original = fine_evidence.propose
    selection = baseline['fine_selection']
    adaptive = selection['adaptive']
    fine = baseline['fine']
    fine_ids = {c['id'] for c in fine}
    coarse = baseline['coarse']
    visible = {c['id']: fine_evidence.selection.fragments(c) for c in coarse}
    protected = set(adaptive['core_ids'])
    protected.update(mid for c in fine for mid in fine_evidence.dependencies(c))
    protected.update(r['candidate_id'] for r in selection['reservations'] if r['selected'])
    donors = [c for c in fine if c['id'] in eligible and c['id'] not in protected]
    donor = min(donors, key=fine_evidence.key) if donors else None
    ordered = sorted([c for c in coarse if c.get('_ce_score') is not None],
                     key=fine_evidence.key, reverse=True)
    nominees = [c for c in ordered[:adaptive['capacity']]
                if donor is not None and c['id'] not in fine_ids
                and visible[c['id']] and not fine_evidence.dependencies(c)
                and c['_ce_score'] > donor['_ce_score']
                and not fine_evidence.selection.represented_by(c, fine, visible)]
    nominee = nominees[0] if nominees else None
    info = dict(swap=None, eligible_donors=[c['id'] for c in donors],
                eligible_nominees=[c['id'] for c in nominees],
                effective_ce_cutoff=adaptive['capacity'])
    if nominee is not None:
        info['swap'] = dict(removed_id=donor['id'], added_id=nominee['id'],
            removed_ce=donor['_ce_score'], added_ce=nominee['_ce_score'],
            removed_rank=next(i+1 for i,c in enumerate(ordered) if c['id']==donor['id']),
            added_rank=next(i+1 for i,c in enumerate(ordered) if c['id']==nominee['id']))

    def propose(rows, *args, **kwargs):
        chosen, metadata = original(rows, *args, **kwargs)
        same([c['id'] for c in chosen], [c['id'] for c in fine], 'Original fine before swap')
        if nominee is None:
            return chosen, metadata
        old, new = donor['id'], nominee['id']
        by_id = {c['id']: c for c in rows}
        chosen = [by_id[new] if c['id'] == old else c for c in chosen]
        additions = metadata['adaptive']['additions']
        index = next(i for i, a in enumerate(additions) if a['candidate_id'] == old)
        require(additions[index]['ids'] == additions[index]['group_ids'] == [old], 'Singleton donor')
        additions[index] = dict(candidate_id=new, ids=[new], group_ids=[new],
            requirement_ids=[], reason='higher_ce_boundary_swap', unscored=False)
        metadata['adaptive']['proposed_ids'] = [c['id'] for c in chosen]
        metadata['adaptive']['boundary_swap'] = copy.deepcopy(info['swap'])
        metadata['adaptive']['rejected'] = [r for r in metadata['adaptive']['rejected'] if r['id'] != new]
        for decision in metadata['decisions']:
            if decision['id'] == old:
                decision.update(selected=False, reason='lower_ce_assistant_only_swap', represented_by=[])
            elif decision['id'] == new:
                decision.update(selected=True, reason='higher_ce_boundary_swap', represented_by=[])
        require({c['id'] for c in chosen} == (fine_ids - {old}) | {new}, 'Single swap membership')
        require(protected <= {c['id'] for c in chosen}, 'Protected fine member lost')
        return chosen, metadata

    with patch.object(fine_evidence, 'propose', side_effect=propose):
        changed, admission, size = replay(trace, modules, code)
    info['admission'] = admission
    return changed, info, size


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
    with patch('socket.socket.connect', side_effect=AssertionError('Offline fine-boundary swap replay')):
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
                changed, ordering, changed_bytes = swap_replay(t, modules, code, baseline, eligible)
                same(changed['trace']['fine']['selection']['adaptive']['core_ids'],
                     baseline['trace']['fine']['selection']['adaptive']['core_ids'], 'Preserved fine core')
                same(len(changed['fine']), len(baseline['fine']), 'Preserved fine count')
                before, after = ([x['id'] for x in env['submitted']] for env in (baseline, changed))
                if not ordering['swap']:
                    same(after, before, 'Untriggered main order')
                    same(changed_bytes, original_bytes, 'Untriggered bytes')
                jobs.append(dict(**result, query=t['query'], options=t['options'], eligible_ids=eligible,
                    ordering=ordering, original_main_ids=before, proposed_main_ids=after,
                    original_bytes=original_bytes, proposed_bytes=changed_bytes,
                    fine_cards=cards(baseline['fine']), proposed_fine_cards=cards(changed['fine']),
                    original_cards=cards(baseline['submitted']),
                    proposed_cards=cards(changed['submitted']),
                    new_fine=[dict(candidate_id=x['id'], rank_text=x.get('_rank_text', x.get('content', '')))
                              for x in changed['fine'] if x['id'] not in {c['id'] for c in baseline['fine']}],
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
                    for name in ('fine', 'proposed_fine', 'original', 'proposed')}
        stage = stages[job['conversation_id'], job['qa_id']]
        for name, source in (('fine', 'fine'), ('original', 'listwise_input')):
            same(coverage[name], [v['full'] for v in stage['visible'][source]['turn_coverage']], 'Original transport ' + name)
        job['reference_turns'] = coverage
        job['fine_turn_losses'] = [i for i, (a, b) in enumerate(zip(coverage['fine'], coverage['proposed_fine'], strict=True)) if a and not b]
        job['fine_turn_gains'] = [i for i, (a, b) in enumerate(zip(coverage['fine'], coverage['proposed_fine'], strict=True)) if b and not a]
        job['turn_losses'] = [i for i, (a, b) in enumerate(zip(coverage['original'], coverage['proposed'], strict=True)) if a and not b]
        job['turn_gains'] = [i for i, (a, b) in enumerate(zip(coverage['original'], coverage['proposed'], strict=True)) if b and not a]
        rows.append(job)
    summary = dict(questions=len(rows), baseline_exact_questions=len(rows),
        swapped_questions=sum(bool(r['ordering']['swap']) for r in rows),
        original_fine_reference_questions=sum(any(r['reference_turns']['fine']) for r in rows),
        proposed_fine_reference_questions=sum(any(r['reference_turns']['proposed_fine']) for r in rows),
        original_fine_reference_turns=sum(sum(r['reference_turns']['fine']) for r in rows),
        proposed_fine_reference_turns=sum(sum(r['reference_turns']['proposed_fine']) for r in rows),
        fine_turn_losses=[dict(qa_id=r['qa_id'], turn_indices=r['fine_turn_losses']) for r in rows if r['fine_turn_losses']],
        fine_turn_gains=[dict(qa_id=r['qa_id'], turn_indices=r['fine_turn_gains']) for r in rows if r['fine_turn_gains']],
        eligible_questions=sum(bool(r['eligible_ids']) for r in rows), eligible_instances=sum(len(r['eligible_ids']) for r in rows),
        main_changed_questions=sum(r['original_main_ids'] != r['proposed_main_ids'] for r in rows),
        original_reference_questions=sum(any(r['reference_turns']['original']) for r in rows),
        proposed_reference_questions=sum(any(r['reference_turns']['proposed']) for r in rows),
        original_reference_turns=sum(sum(r['reference_turns']['original']) for r in rows),
        proposed_reference_turns=sum(sum(r['reference_turns']['proposed']) for r in rows),
        turn_losses=[dict(qa_id=r['qa_id'],turn_indices=r['turn_losses']) for r in rows if r['turn_losses']],
        turn_gains=[dict(qa_id=r['qa_id'],turn_indices=r['turn_gains']) for r in rows if r['turn_gains']])
    inputs = [Path(__file__), Path(__file__).with_name('probe_assistant_admission_order.py'),
              Path(__file__).with_name('audit_retained_porter_https.py'), Path(__file__).with_name('audit_https_funnel.py'),
              root / 'plan.json', root / 'fixture.json', root / 'retained-selection-audit.json',
              args.inventory, args.gold_csv, Path(manifest['data']), run / 'server-memory-debug.jsonl', *paths]
    report = dict(status='complete_offline_fine_boundary_swap_replay', summary=summary, rows=rows,
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
