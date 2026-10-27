"""Full-cohort frozen-coarse to CE-fine/main funnel for the bridge diagnostic.

This is stage transport, not causal attribution to one score/budget decision,
and not a simulation of downstream listwise selection, packets or answers.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def summarize(masks):
    return dict(reference_questions=sum(any(mask) for mask in masks),
                reference_turns=sum(sum(mask) for mask in masks))


def main(root):
    state = json.loads((root / 'live-state.json').read_bytes())
    assert state['status'] == 'complete'
    decision = json.loads((root / 'numerical-decision.json').read_bytes())
    for name, digest in decision['input_sha256'].items():
        assert sha(Path(name)) == digest, name
    cohorts, all_rows, paths = {}, [], [Path(__file__), root / 'live-state.json', root / 'numerical-decision.json']
    for cohort in ('regression', 'validation'):
        directory = root / cohort
        plan = json.loads((directory / 'plan.json').read_bytes())
        report = json.loads((directory / 'result.json').read_bytes())
        stage_path = Path(plan['run']) / 'funnel/stage-transport.json'
        assert sha(stage_path) == plan['stage_sha256']
        stages = {(row['conversation_id'], row['qa_id']): row
                  for row in json.loads(stage_path.read_bytes())['rows']}
        assert set(stages) == {(row['conversation_id'], row['qa_id']) for row in report['rows']}
        rows = []
        for row in report['rows']:
            stage = stages[row['conversation_id'], row['qa_id']]
            raw = [entry['full'] for entry in stage['raw_turn_coverage']]
            coarse = [entry['full'] for entry in stage['visible']['coarse']['turn_coverage']]
            observed = dict(cohort=cohort, qa_id=row['qa_id'], conversation_id=row['conversation_id'],
                            raw=raw, coarse=coarse, variants={})
            for arm, variant in row['variants'].items():
                if variant is None:
                    observed['variants'][arm] = None
                    continue
                fine, main = variant['fine']['user'], variant['main']['user']
                assert len(raw) == len(coarse) == len(fine) == len(main)
                if arm == 'archived':
                    assert fine == [entry['full'] for entry in stage['visible']['fine']['turn_coverage']]
                    assert main == [entry['full'] for entry in stage['visible']['listwise_input']['turn_coverage']]
                observed['variants'][arm] = dict(fine=fine, main=main, main_bytes=variant['main_bytes'],
                    coarse_to_fine_missing=[i for i, (a, b) in enumerate(zip(coarse, fine)) if a and not b],
                    fine_to_main_missing=[i for i, (a, b) in enumerate(zip(fine, main)) if a and not b])
            rows.append(observed)
        summary = dict(questions=len(rows), raw=summarize([r['raw'] for r in rows]),
                       coarse=summarize([r['coarse'] for r in rows]), variants={})
        for arm in ('archived', 'control', 'candidate', 'candidate_cost_adjusted'):
            valid = [r for r in rows if r['variants'][arm] is not None]
            variants = [r['variants'][arm] for r in valid]
            value = dict(evaluated_questions=len(valid), unavailable_questions=[r['qa_id'] for r in rows
                         if r['variants'][arm] is None], stages={})
            for key in ('fine', 'main', 'coarse_to_fine_missing', 'fine_to_main_missing'):
                if key in ('fine', 'main'):
                    value['stages'][key] = summarize([v[key] for v in variants])
                else:
                    value['stages'][key] = dict(questions=sum(bool(v[key]) for v in variants),
                        turns=sum(len(v[key]) for v in variants),
                        instances=[dict(qa_id=r['qa_id'], turn_indices=r['variants'][arm][key])
                                   for r in valid if r['variants'][arm][key]])
            summary['variants'][arm] = value
        cohorts[cohort] = summary
        all_rows.extend(rows)
        paths.extend([directory / 'plan.json', directory / 'result.json', stage_path])
    output = dict(status='complete', cohorts=cohorts, rows=all_rows, limitation=__doc__,
                  input_sha256={str(path.resolve()): sha(path) for path in paths})
    with (root / 'funnel-analysis.json').open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps(cohorts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
