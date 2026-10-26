"""Offline one-seat main admission swap within all original request limits.

Original main is the baseline. Protect reservations and dependency groups. Only
replace an unprotected assistant-only lexical match with a strictly higher-CE,
source-novel missing fine candidate with nonempty visible source evidence.
Fine eligibility already passed; no extra lexical-match gate is added at main.
Both are intact singleton units; no source clipping, count increase or score
change. Choose nominees by CE descending and donors by CE ascending, first
whole-request fit. At most one swap. Gold is loaded only after selections.
No provider calls or final-packet/answer prediction.
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


def swap_replay(trace, modules, code, baseline):
    cascade, config, fine_evidence, _ = modules
    coarse, fine, submitted = baseline['coarse'], baseline['fine'], baseline['submitted']
    adaptive = baseline['fine_selection']['adaptive']
    facets = baseline['requirements'] + adaptive.get('context_facets', [])
    visible = {c['id']: fine_evidence.selection.fragments(c) for c in coarse}
    sparse = fine_evidence.sparse_matches(coarse, facets, visible)
    matches = {}
    for candidate in coarse:
        mid = candidate['id']
        combined = fine_evidence.relevance(candidate, facets, visible[mid])
        for rid, value in sparse[mid].items():
            combined[rid] = max(combined.get(rid, (False, False, 0)), value)
        matches[mid] = combined
    protected = set(baseline['carried_ids'])
    protected.update(mid for c in fine for mid in fine_evidence.dependencies(c))
    protected.update(mid for group in adaptive.get('path_groups', []) for mid in group)
    donors = []
    for candidate in submitted:
        mid = candidate['id']
        assistant_only = (bool(matches[mid]) and not any(v[0] for v in matches[mid].values())
                          and all(source[1] in ('user', 'assistant') for source, _ in visible[mid]))
        if (mid not in protected and candidate.get('_ce_score') is not None
                and not candidate.get('_user_rule') and not candidate.get('_packet_item', {}).get('is_constraint')
                and not fine_evidence.dependencies(candidate) and assistant_only):
            donors.append(candidate)
    donors.sort(key=fine_evidence.key)
    original_ids = {c['id'] for c in submitted}
    nominees = sorted([c for c in fine if c['id'] not in original_ids
                       and c.get('_ce_score') is not None
                       and not fine_evidence.dependencies(c)
                       and bool(visible[c['id']])],
                      key=fine_evidence.key, reverse=True)
    positions = {c['id']: i for i, c in enumerate(fine)}
    info = dict(swap=None, protected_ids=sorted(protected),
        eligible_donors=[c['id'] for c in donors], eligible_nominees=[c['id'] for c in nominees], attempts=[])
    for nominee in nominees:
        for donor in donors:
            if nominee['_ce_score'] <= donor['_ce_score']:
                continue
            kept = [c for c in submitted if c['id'] != donor['id']]
            if fine_evidence.selection.represented_by(nominee, kept, visible):
                continue
            trial = sorted(kept + [nominee], key=lambda c: positions[c['id']])
            request = cascade.listwise_request(baseline['req'], baseline['plan'], trial)
            size = len(request['prompt'].encode())
            headroom = trace['cascade']['admission_budget']['token_headroom']
            reason = ('prompt_budget' if size > config.RERANK_MAX_PROMPT_BYTES else
                      'token_budget' if headroom is not None and request['reservation'] > headroom else 'admitted')
            info['attempts'].append(dict(removed_id=donor['id'], added_id=nominee['id'],
                                         main_bytes=size, reservation=request['reservation'], reason=reason))
            if reason != 'admitted':
                continue
            new_ids = {c['id'] for c in trial}
            same(len(trial), len(submitted), 'Main count unchanged')
            require(len(trial) <= config.CASCADE_LLM_LIMIT + config.FINE_EXTRA_LIMIT, 'Count bound')
            same(new_ids, (original_ids - {donor['id']}) | {nominee['id']}, 'Exactly one swap')
            require(original_ids & protected <= new_ids, 'Protected original main lost')
            info['swap'] = dict(removed_id=donor['id'], added_id=nominee['id'],
                removed_ce=donor['_ce_score'], added_ce=nominee['_ce_score'],
                donor_matches=matches[donor['id']], nominee_matches=matches[nominee['id']])
            changed = dict(baseline, submitted=trial)
            return changed, info, size
    return baseline, info, trace['cascade']['listwise']['input_bytes']


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
                changed, ordering, changed_bytes = swap_replay(t, modules, code, baseline)
                same(changed['trace']['fine'], baseline['trace']['fine'], 'Preserved complete fine pool')
                before, after = ([x['id'] for x in env['submitted']] for env in (baseline, changed))
                jobs.append(dict(**result, query=t['query'], options=t['options'],
                    ordering=ordering, original_main_ids=before, proposed_main_ids=after,
                    original_bytes=original_bytes, proposed_bytes=changed_bytes,
                    fine_cards=cards(baseline['fine']), original_cards=cards(baseline['submitted']),
                    proposed_cards=cards(changed['submitted']),
                    new_main=[dict(candidate_id=x['id'], rank_text=x.get('_rank_text', x.get('content', '')))
                              for x in changed['submitted'] if x['id'] not in before]))
            print(json.dumps(dict(qa_id=result['qa_id'],baseline_exact=True,swapped=bool(ordering['swap']),
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
        swapped_questions=sum(bool(r['ordering']['swap']) for r in rows),
        eligible_donor_questions=sum(bool(r['ordering']['eligible_donors']) for r in rows),
        eligible_nominee_questions=sum(bool(r['ordering']['eligible_nominees']) for r in rows),
        original_main_candidates=sum(len(r['original_main_ids']) for r in rows),
        proposed_main_candidates=sum(len(r['proposed_main_ids']) for r in rows),
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
              args.gold_csv, Path(manifest['data']), run / 'server-memory-debug.jsonl', *paths]
    report = dict(status='complete_offline_main_ce_swap_replay', summary=summary, rows=rows,
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
