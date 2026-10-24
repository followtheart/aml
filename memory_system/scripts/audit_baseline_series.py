"""Describe selected historical control runs; do not infer independent trials.

The same 56 questions are repeated. Frozen-input hashes and recorded settings
are checked, but application revisions, provider state and execution protocols
can differ. Pooled accuracy is descriptive, not an estimate on new questions.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from audit_baseline_repeat import load
from evaluation_model_guard import check_versions


def main(args):
    cohorts, runs = [], []
    baseline, base_settings, base_fixture = None, None, None
    for root in args.runs:
        plan = json.loads((root / 'plan.json').read_text())
        fixture = json.loads((root / 'fixture.json').read_text())
        requests = {r['qa_id']: r for r in fixture['requests']}
        cohort = load(root / 'control')
        settings = next(iter(cohort.values()))[1]['versions']['settings']
        if baseline is None:
            baseline = {q: {k: t[k] for k in ('query', 'options', 'plan')}
                        for q, (_, t, _) in cohort.items()}
            base_settings, base_fixture = settings, requests
            input_hashes = {k: plan[k] for k in ('database_sha256', 'data_sha256')}
        if (cohort.keys() != baseline.keys() or requests != base_fixture
                or any(plan[k] != v for k, v in input_hashes.items())
                or any(settings.get(k) != v for k, v in base_settings.items())
                or any(settings[k] is not False for k in settings.keys() - base_settings.keys())):
            raise ValueError('Frozen inputs or baseline settings differ')
        for q, (_, trace, _) in cohort.items():
            check_versions(trace['versions'], str(root))
            if (trace['versions']['settings'] != settings
                    or any(trace[k] != baseline[q][k] for k in ('query', 'options', 'plan'))
                    or trace['cascade']['cross_encoder']['model'] != 'qwen3-rerank'):
                raise ValueError('Question/plan/settings/model changed within series')
        scores = [r[0]['score'] for r in cohort.values()]
        runs.append(dict(run=str(root), protocol=plan['protocol'], questions=len(cohort),
            correct=sum(scores), accuracy=sum(scores)/len(scores),
            reference_questions=sum(s['visible']['packet']['full_turns'] > 0 for _, _, s in cohort.values()),
            additional_disabled_settings=sorted(settings.keys() - base_settings.keys()),
            plan_sha256=hashlib.sha256((root / 'plan.json').read_bytes()).hexdigest(),
            results_sha256=hashlib.sha256((root / 'control/completed/data/results/personamem-v2-32k.jsonl').read_bytes()).hexdigest()))
        cohorts.append({q: (r['score'], s) for q, (r, _, s) in cohort.items()})
        del cohort
    questions = []
    for q in sorted(baseline, key=int):
        stages = [c[q][1] for c in cohorts]
        present = [s['visible']['packet']['full_turns'] > 0 for s in stages]
        raw = [any(t.get('full') for t in s['raw_turn_coverage']) for s in stages]
        if len(set(raw)) != 1:
            raise ValueError('Raw reference availability differs')
        scores = [c[q][0] for c in cohorts]
        questions.append(dict(qa_id=q, scores=scores, correct_runs=sum(scores),
            raw_reference_available=raw[0], packet_reference_present=present,
            packet_reference_runs=sum(present),
            stage_reference_runs={stage:sum(s['visible'][stage]['full_turns'] > 0 for s in stages)
                for stage in stages[0]['visible']}))
    summary = dict(runs=len(runs), unique_questions=len(questions),
        protocol_counts=dict(Counter(r['protocol'] for r in runs)),
        scores=[r['correct'] for r in runs],
        descriptive_mean_accuracy=statistics.mean(r['accuracy'] for r in runs),
        accuracy_range=[min(r['accuracy'] for r in runs), max(r['accuracy'] for r in runs)],
        always_missing_reference=[q['qa_id'] for q in questions if q['raw_reference_available'] and not q['packet_reference_runs']],
        variable_reference=[q['qa_id'] for q in questions if 0 < q['packet_reference_runs'] < len(runs)],
        variable_score=[q['qa_id'] for q in questions if 0 < q['correct_runs'] < len(runs)],
        always_wrong=[q['qa_id'] for q in questions if not q['correct_runs']],
        recorded_base_settings=len(base_settings), frozen_input_hashes=input_hashes)
    report = dict(summary=summary, runs=runs, questions=questions, limitation=__doc__,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
