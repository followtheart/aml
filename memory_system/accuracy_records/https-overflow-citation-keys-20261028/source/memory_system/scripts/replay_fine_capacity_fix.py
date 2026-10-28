"""Compare current fine policy to a frozen run's implementation and CE scores.

Gold snippets are used only by the transport auditor, never by selection.
No provider calls or counterfactual answer-quality claims.
"""
import argparse
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import probe_ranking_capacity as probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.run / 'source/memory_system/app/fine_evidence.py'
    spec = importlib.util.spec_from_file_location('app._archived_fine_evidence', source)
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    replay = probe.replay
    def compare(search, fine_limit, llm_limit, policy='archived'):
        if policy == 'archived':
            with patch.object(probe.fine_evidence, 'propose', archived.propose):
                return replay(search, fine_limit, llm_limit, policy)
        return replay(search, fine_limit, llm_limit, policy)
    with patch.object(probe, 'replay', side_effect=compare), \
         patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
        out = probe.experiment(args.run, ['adaptive_shared'], [(24, 24)])
    out['baseline_source'] = str(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(baseline_reproduced=out['baseline_reproduced'],
                         questions=out['questions'], summary=out['summary']), indent=2))


if __name__ == '__main__':
    main()
