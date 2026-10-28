"""Use Search's query-balanced rank order for measured semantic source heads.

RRF60 can favor a source repeated across many similar queries. Compare the
existing Search convention: every query's first hit before any second hit.
Keep top six per query, eight nominated sources, one residual carrier and all
original packet limits. This is an offline alternative, not new measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import probe_semantic_source_context as semantic
from probe_source_rank_residual import main as score


def balanced_sources(queries):
    positions, seen = {}, set()
    for index, row in enumerate(queries):
        text = row['query']['text']
        if text in seen:
            continue
        seen.add(text)
        for rank, hit in enumerate(row['hits'][:6], 1):
            si = hit['source_index']
            positions[si] = min(positions.get(si, (rank, index)), (rank, index))
    return sorted(positions, key=lambda si: (*positions[si], si))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=Path('data/raw/personamem-v2.csv'))
    args = parser.parse_args()
    plan_path, ranks_path = args.probe / 'plan.json', args.probe / 'rankings.json'
    plan, ranks = json.loads(plan_path.read_text()), json.loads(ranks_path.read_text())
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != ranks['plan_sha256']:
        raise ValueError('Measured inputs changed')
    modes = tuple(ranks['rankings'][0]['modes'])
    with patch('socket.socket.connect', side_effect=AssertionError('Offline')), patch.object(semantic, 'ranked_sources', balanced_sources):
        score(args, preparer=semantic.preparer(plan, ranks, modes), modes=modes, limitation=__doc__, script_path=Path(__file__))
    result = json.loads(args.output.read_text())
    result['measured_inputs'] = dict(plan_sha256=ranks['plan_sha256'], rankings_sha256=hashlib.sha256(ranks_path.read_bytes()).hexdigest())
    result['carrier_proposer_sha256'] = hashlib.sha256(Path(semantic.__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
