"""Count complete user reference sets in the six frozen residual proposals.

Reference-question presence, complete sets and individual turns are separate.
This audit scores saved coverage, not answer accuracy or personal entailment.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main(args):
    files = ('semantic-source-context-proposal', 'exchange-source-context-proposal',
             'semantic-balanced-context-proposal', 'exchange-balanced-context-proposal')
    rows, inputs = [], []
    baseline = None
    for name in files:
        path = args.run / (name + '.json')
        data = json.loads(path.read_text())
        cases = data['rows']
        if len(cases) != 56 or len({r['qa_id'] for r in cases}) != 56:
            raise ValueError('Require full 56-question proposals')
        current = {r['qa_id']: r['coverage']['baseline'] for r in cases}
        if baseline is not None and current != baseline:
            raise ValueError('Different original coverage')
        baseline = current
        inputs.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            script_sha256=data['script_sha256'], helper_sha256=data['common_helper_sha256']))
        for summary in data['summary']:
            mode = summary['mode']
            metrics = {}
            for variant in ('baseline', mode):
                turns = [r['coverage'][variant]['user'] for r in cases]
                metrics[variant] = dict(reference_questions=sum(map(any, turns)),
                    user_turns=sum(map(sum, turns)), complete_sets=sum(bool(v) and all(v) for v in turns),
                    catalog_bytes=sum(r['catalog_bytes'][variant] for r in cases))
                for key in ('reference_questions', 'user_turns', 'catalog_bytes'):
                    if metrics[variant][key] != summary[key][variant]:
                        raise ValueError('Stored summary disagrees with rows')
            gained = [r['qa_id'] for r in cases if r['coverage'][mode]['user']
                and all(r['coverage'][mode]['user']) and not all(r['coverage']['baseline']['user'])]
            losses = [r['qa_id'] for r in cases if any(a and not b for role in ('user', 'assistant')
                for a, b in zip(r['coverage']['baseline'][role], r['coverage'][mode][role]))]
            if losses:
                raise ValueError('Original reference turn lost: '+str(losses))
            rows.append(dict(proposal=name, mode=mode, metrics=metrics,
                complete_set_gains=gained, turn_losses=losses,
                appended_questions=sum(bool(r['proposals'][mode]['appended_ids']) for r in cases)))
    result = dict(rows=rows, inputs=inputs, questions=56, variant_packets=336,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), limitation=__doc__)
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
