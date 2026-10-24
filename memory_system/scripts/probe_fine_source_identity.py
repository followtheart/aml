"""Count original sources, not profile-quote projections, in fine rarity.

Retain the existing 0.3 lexical ratio gate, core, ordering and all capacities.
Only sparse term document frequencies and their corpus-size threshold change.
This offline probe scores fine/main transport, not final or answer accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import probe_sparse_fine_ratio as probe
from app import fine_evidence, evidence_selection

ORIGINAL = fine_evidence.sparse_matches


def canonical_sparse(rows, requirements, visible):
    by_id = {row['id']: row for row in rows}
    canonical = {}
    for mid, fragments in visible.items():
        pieces = []
        for (key, role), text in fragments:
            if not key.startswith('profile-quote:'):
                pieces.append(((key, role), text))
                continue
            for source in by_id[mid].get('_packet_item', {}).get('sources', []):
                if source.get('role') != role or text not in evidence_selection._text(source.get('content', '')):
                    continue
                sid = source.get('source_event_id')
                if not sid and source.get('request_id') is not None and source.get('message_index') is not None:
                    sid = f"{source['request_id']}:{source['message_index']}"
                if sid:
                    pieces.append(((sid, role), text))
        canonical[mid] = pieces
    return ORIGINAL(rows, requirements, canonical)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=Path('data/raw/personamem-v2.csv'))
    args = parser.parse_args()
    args.bounded = False
    args.prototype = None
    with patch.object(probe, 'sparse_without_ratio', canonical_sparse), patch.object(probe, '__doc__', __doc__):
        probe.main(args)
    result = json.loads(args.output.read_text())
    result['proposal_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['variant_definition'] = {'baseline': 'original sparse matcher',
                                  'rare': 'same matcher with original-source identity for rarity only'}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
