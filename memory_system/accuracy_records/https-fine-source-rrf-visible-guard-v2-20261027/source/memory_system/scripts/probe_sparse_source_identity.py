"""Check sparse rarity using original source identities for profile quote views.

The general fine selector intentionally gives profile quotes separate coverage
identities. Sparse source rarity should instead deduplicate their real message
origins. This offline proposal canonicalizes only those keys, preserving text,
roles, dependencies, query terms and all packet limits. No provider calls.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

from probe_sparse_query_context import prepare, MODES
from probe_source_rank_residual import main as score
from app import fine_sparse_reserve, evidence_selection

original_nominate = fine_sparse_reserve.nominate


def canonical_nominate(rows, core, requirements, visible):
    by_id = {c['id']: c for c in [*rows, *core]}
    canonical = {}
    for mid, pieces in visible.items():
        result = []
        for (key, role), text in pieces:
            if not key.startswith('profile-quote:'):
                result.append(((key, role), text))
                continue
            for source in by_id[mid].get('_packet_item', {}).get('sources', []):
                if source.get('role') != role or text not in evidence_selection._text(source.get('content', '')):
                    continue
                sid = source.get('source_event_id')
                if not sid and source.get('request_id') is not None and source.get('message_index') is not None:
                    sid = f"{source['request_id']}:{source['message_index']}"
                if sid:
                    result.append(((sid, role), text))
        canonical[mid] = result
    return original_nominate(rows, core, requirements, canonical)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    with patch('socket.socket.connect', side_effect=AssertionError('Offline')), patch.object(
            fine_sparse_reserve, 'nominate', side_effect=canonical_nominate):
        score(p.parse_args(), preparer=prepare, modes=MODES, limitation=__doc__, script_path=Path(__file__))
