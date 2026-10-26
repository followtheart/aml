"""Joint decision evidence for two terminal retained-Porter HTTPS cohorts.

All original questions/errors remain in the denominator. Reference preservation
compares individual user-turn positions, not aggregate counts. This report does
not certify semantic relevance or enable an application setting.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from audit_retained_porter_https import records, require, same, sha


def identity(row):
    return row['conversation_id'], row['qa_id']


def turn_changes(before, after, stage):
    left = before['visible'][stage]['turn_coverage']
    right = after['visible'][stage]['turn_coverage']
    same([v['normalized_chars'] for v in left], [v['normalized_chars'] for v in right],
         'Reference turn inventory')
    return dict(lost=[i for i, (a, b) in enumerate(zip(left, right, strict=True)) if a['full'] and not b['full']],
                gained=[i for i, (a, b) in enumerate(zip(left, right, strict=True)) if b['full'] and not a['full']])


def summarize(root):
    inputs = {}

    def read(relative):
        path = root / relative
        inputs[str(path)] = sha(path)
        return json.loads(path.read_text())

    plan = read('plan.json')
    fixture = read('fixture.json')
    same(plan['toggle'], 'RECALL_PORTER_SUPPLEMENT', 'Experiment toggle')
    same(inputs[str(root / 'fixture.json')], plan['fixture_sha256'], 'Fixture hash')
    same(plan['setting_values'], dict(control=False, treatment=True), 'Arm settings')
    for name, digest in plan['source_files'].items():
        same(sha(root / 'source' / name), digest, 'Frozen source ' + name)
    require((root / 'exit-code').read_text().strip() in ('0', '1'), 'Nonterminal experiment')
    for name in ('funnel', 'runtime', 'retained-selection'):
        path = root / f'final-{name}.exit-code'
        same(path.read_text().strip(), '0', 'Terminal audit ' + name)
        inputs[str(path)] = sha(path)
    funnel = read('funnel-comparison.json')
    runtime = read('runtime-accounting-audit.json')
    selection = read('retained-selection-audit.json')
    same(funnel['settings_differences'], {'RECALL_PORTER_SUPPLEMENT': [False, True]}, 'Single intervention')
    same(selection['status'], 'passed', 'Selection audit')
    same(selection['plan_sha256'], inputs[str(root / 'plan.json')], 'Selection plan hash')
    same(selection['provider_calls'], 0, 'Offline selection replay')
    require(not runtime['recovered_questions'], 'Unexpected successful-answer/recovery rerun')
    for name, digest in selection['input_sha256'].items():
        same(sha(Path(name)), digest, 'Selection audit input ' + name)
    expected = {identity(r) for r in fixture['requests']}
    same(len(expected), plan['questions'], 'Question count')
    rows, answers, arms = {}, {}, {}
    for arm in ('control', 'treatment'):
        manifest = read(arm + '/isolated-manifest.json')
        require(manifest.get('server_exit_code') in (0, -15), 'Nonterminal service')
        process = Path('/proc') / str(manifest['server_pid']) / 'cmdline'
        require(not process.exists() or str(root).encode() not in process.read_bytes(), 'Service still running')
        same(manifest['audit_exit_code'], 0, 'Automatic stage audit')
        stage = read(arm + '/funnel/stage-transport.json')
        path = root / arm / 'completed/data/results/personamem-v2-32k.jsonl'
        results = list(records(path))
        inputs[str(path)] = sha(path)
        same({identity(r) for r in stage['rows']}, expected, 'Stage question set')
        same({identity(r) for r in results}, expected, 'Result question set')
        same(len(stage['rows']), plan['questions'], 'Stage duplicates')
        same(len(results), plan['questions'], 'Result duplicates')
        rows[arm] = {identity(r): r for r in stage['rows']}
        answers[arm] = {identity(r): r for r in results}
        for r in results:
            require(r['fake'] is False and r['transport'] == 'https', 'Expected real HTTPS')
            require(not r.get('error_stage') or r['error_stage'] == 'answer' and r['score'] == 0,
                    'Missing Search outcome needs explicit investigation')
        score = sum(r['score'] for r in results)
        same(score, funnel['summaries'][arm]['score'], 'Answer score')
        same(manifest['client_exit_code'], int(any(r.get('error_stage') for r in results)), 'Client exit')
        summary = stage['summary']
        cost = runtime['arms'][arm]
        same(cost['search_budget_calls'], funnel['summaries'][arm]['calls'], 'Search calls')
        same(cost['search_budget_tokens'], funnel['summaries'][arm]['tokens'], 'Search tokens')
        arms[arm] = dict(questions=len(results), score=score, accuracy=score / len(results),
            stage_reference_questions=summary['visible_full_turn_question_counts'],
            stage_reference_turns=summary['delivered_available_turn_counts'],
            available_complete_sets=summary['available_set_complete_question_counts'],
            reference_complete_sets=summary['strict_reference_set_complete_question_counts'],
            terminal_errors=[r['qa_id'] for r in results if r.get('error_stage')],
            porter_recall=funnel['summaries'][arm]['porter_recall'], runtime=cost)

    stages = list(next(iter(rows['control'].values()))['visible'])
    deltas = {stage: dict(gained_questions=[], lost_questions=[], turn_gains=[], turn_losses=[]) for stage in stages}
    wins, losses, residual = [], [], Counter()
    for key in sorted(expected):
        a, b = rows['control'][key], rows['treatment'][key]
        same(a['raw_turn_coverage'], b['raw_turn_coverage'], 'Frozen raw reference coverage')
        for stage in stages:
            change = turn_changes(a, b, stage)
            if change['lost']:
                deltas[stage]['turn_losses'].append(dict(qa_id=key[1], turn_indices=change['lost']))
            if change['gained']:
                deltas[stage]['turn_gains'].append(dict(qa_id=key[1], turn_indices=change['gained']))
            if a['visible'][stage]['full_turns'] and not b['visible'][stage]['full_turns']:
                deltas[stage]['lost_questions'].append(key[1])
            if b['visible'][stage]['full_turns'] and not a['visible'][stage]['full_turns']:
                deltas[stage]['gained_questions'].append(key[1])
        aa, ab = answers['control'][key], answers['treatment'][key]
        if ab['score'] > aa['score']:
            wins.append(key[1])
        elif ab['score'] < aa['score']:
            losses.append(dict(qa_id=key[1], final_reference_change=turn_changes(a, b, 'catalog'),
                               control_error=aa.get('error_stage'), treatment_error=ab.get('error_stage')))
        if not b['visible']['catalog']['full_turns']:
            residual['raw_unavailable' if not b['raw_full_turns'] else
                     next((stage for stage in stages if not b['visible'][stage]['full_turns']), 'unknown')] += 1
    before, after = arms['control'], arms['treatment']
    evidence = dict(reference_preserved=not deltas['catalog']['turn_losses'],
                    answer_nonregression=after['score'] >= before['score'],
                    final_reference_gain=(after['stage_reference_questions']['catalog'] > before['stage_reference_questions']['catalog']
                        or after['stage_reference_turns']['catalog'] > before['stage_reference_turns']['catalog']))
    return dict(arms=arms, deltas=deltas, wins=wins, losses=losses,
                residual_first_missing_stage=dict(residual), gate_evidence=evidence,
                accounting_fully_reconciled=all(v['runtime']['calls_match'] and v['runtime']['tokens_match'] for v in arms.values()),
                sensitivity=runtime['sensitivity'], input_sha256=inputs,
                original_pool_preservation=dict(prepared_questions=funnel['same_original_precoarse_questions'],
                                                 coarse_questions=funnel['same_original_coarse_questions']))


def main(args):
    require(not args.output.exists(), 'Refusing to overwrite joint decision')
    require(args.regression.resolve() != args.validation.resolve(), 'Cohorts must be separate experiments')
    plans = [json.loads((path / 'plan.json').read_text()) for path in (args.regression, args.validation)]
    same([p['questions'] for p in plans], [56, 61], 'Preregistered cohort sizes')
    same(plans[0]['source_files'], plans[1]['source_files'], 'Shared frozen implementation')
    cohorts = {name: summarize(path.resolve()) for name, path in
               (('regression', args.regression), ('validation', args.validation))}
    gates = [v['gate_evidence'] for v in cohorts.values()]
    passed = all(v['reference_preserved'] and v['answer_nonregression'] for v in gates) and any(
        v['final_reference_gain'] for v in gates)
    report = dict(status='terminal_audited_evidence', cohorts=cohorts,
                  preregistered_transport_answer_gate_passed=passed,
                  semantic_review_status='pending', default_changed=False,
                  decision='requires_semantic_review' if passed else 'reject_by_preregistered_gate',
                  auditor_sha256=sha(Path(__file__)), limitation=__doc__)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(decision=report['decision'], gate_evidence={k: v['gate_evidence'] for k, v in cohorts.items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regression', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
