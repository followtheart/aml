"""Numerical gate for paired bridge-summary CE diagnostics, never adoption."""
import argparse
import hashlib
import json
from pathlib import Path


def main(root):
    paths, cohorts = [], {}
    for cohort in ('regression', 'validation'):
        directory = root / cohort
        required = [directory / name for name in ('plan.json', 'result.json', 'audit.json')]
        plan, result, audit = [json.loads(path.read_bytes()) for path in required]
        assert plan['policy'] == 'retrieval_bridge_summary_ce_ablation_complete_original_sources'
        assert audit['status'] == 'passed'
        assert audit['provider_calls'] == result['summary']['calls']
        assert audit['reported_tokens'] == result['summary']['reported_tokens']
        assert len(result['rows']) == plan['preflight']['questions'] == audit['questions']
        rows = result['rows']
        comparisons = {}
        for before, after in (('control', 'candidate'), ('control', 'candidate_cost_adjusted'),
                              ('archived', 'candidate'), ('archived', 'candidate_cost_adjusted')):
            stages = {}
            for stage in ('fine', 'main'):
                gains, losses, unavailable = [], [], []
                for row in rows:
                    a, b = row['variants'][before], row['variants'][after]
                    if a is None or b is None:
                        unavailable.append(row['qa_id'])
                        continue
                    for index, (old, new) in enumerate(zip(a[stage]['user'], b[stage]['user'], strict=True)):
                        if new and not old:
                            gains.append(dict(qa_id=row['qa_id'], turn=index))
                        if old and not new:
                            losses.append(dict(qa_id=row['qa_id'], turn=index))
                stages[stage] = dict(gains=gains, losses=losses, unavailable=unavailable)
            comparisons[before + '_to_' + after] = stages
        cohorts[cohort] = dict(summary=result['summary'], comparisons=comparisons,
            calls_by_arm=audit['arms'], historical_control_matches=audit['historical_control_matches'],
            main_membership_changed_questions=[row['qa_id'] for row in rows
                if row['variants']['control'] is not None and row['variants']['candidate'] is not None
                and set(row['variants']['control']['main']['ids']) != set(row['variants']['candidate']['main']['ids'])])
        paths.extend(required)
    incomplete = any(c['summary']['failed_arms'] for c in cohorts.values())
    loss = any(stage['losses'] for c in cohorts.values() for comparison in c['comparisons'].values()
               for stage in comparison.values())
    gain = all(any(c['comparisons'][comparison]['main']['gains'] for c in cohorts.values())
               for comparison in ('control_to_candidate', 'control_to_candidate_cost_adjusted'))
    decision = ('reject_incomplete_scores' if incomplete else 'reject_reference_losses' if loss
                else 'reject_no_main_gain' if not gain else 'pending_full_semantic_review')
    output = dict(status='complete_numerical_gate', decision=decision, cohorts=cohorts,
        provider_calls=sum(c['summary']['calls'] for c in cohorts.values()),
        reported_tokens=sum(c['summary']['reported_tokens'] for c in cohorts.values()),
        production_changed=False, new_https_evaluation=False,
        limitation='Sequential diagnostic on previously seen cohorts. Passing numerical checks requires full semantic review before integration and HTTPS; it is not evidence of answer accuracy.',
        input_sha256={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [Path(__file__), *paths]})
    with (root / 'numerical-decision.json').open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: output[k] for k in ('status', 'decision', 'provider_calls', 'reported_tokens')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
