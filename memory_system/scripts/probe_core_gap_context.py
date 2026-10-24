"""Fixed-packet probe: one complete CE-core outsider as retrieval context.

Keep the original three context seats and every existing item. Candidates must
rank within the existing core count in its recorded core pool, miss fine, and
have no unsatisfied recorded path dependency. Compare spare original context
bytes with a separately bounded 4 KiB allowance inside the unchanged request
byte/item caps. This increases local context allowance, not the global budget.
No model calls, gold selection, source clipping or factual-support promotion.
"""
import argparse
import copy
import math
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import main as score
from app import config, evidence_packet, fine_evidence, search_pipeline
from app.fine_context_lexical import propose as lexical
from app.fine_context_reserve import propose as reserve

MODES = ('core_gap_spare', 'core_gap_allowance')


def nominate(pool, packed, *, fine_ids, coarse_ids, selection, count):
    kept = {p['id'] for p in packed}
    excluded = set(fine_ids) | kept
    allowed = set(coarse_ids) & set(pool)
    ranks = {d['id']: d.get('rank') for d in selection['decisions']}
    eligible, inventory = [], []
    for mid, rank in sorted(ranks.items(), key=lambda p: p[1] if type(p[1]) is int else math.inf):
        if mid in excluded or mid not in allowed or type(rank) is not int or not 1 <= rank <= count:
            continue
        candidate = pool[mid]
        value = candidate.get('_ce_score')
        group, missing = fine_evidence.closure(candidate, pool)
        reason = ('unscored' if type(value) not in (int, float) or not math.isfinite(value)
                  else 'missing_dependency' if missing or not group <= kept | {mid}
                  else 'nominated')
        inventory.append(dict(candidate_id=mid, core_rank=rank, ce_score=value,
            dependency_ids=sorted(group - {mid}), missing_ids=sorted(missing), reason=reason))
        if reason == 'nominated':
            eligible.append(candidate)
    return eligible, inventory


def prepare(t):
    ctx = t['cascade']['conversation_context']
    if ctx['policy'] != 'fine_user_lexical_context_v1':
        raise ValueError('Require original three context seats')
    base = [dict(p, packet_hash=ctx['base_packet_hash'])
            for p in t['returned'][:len(ctx['base_packet_ids'])]]
    assert evidence_packet.digest(base) == ctx['base_packet_hash']
    pool = {c['id']: c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']), copy.deepcopy(t['plan']))
            if c.get('view_status') != 'stale' and '_packet_item' in c}
    fine_ids = t['cascade']['fine']['selected_ids']
    fine = [pool[mid] for mid in fine_ids if mid in pool and mid not in t['selection']['selected_ids']]
    kwargs = dict(user_id=t['user_id'], top_k=t['top_k'], byte_budget=t['coverage_manifest']['token_budget'])
    prefix, old = lexical(base, fine, **kwargs, reserve_bytes=5120, query=t['query'], options=t['options'])
    digest = evidence_packet.digest(prefix)
    assert [dict(p, packet_hash=digest) for p in prefix] == t['returned']
    nominees, inventory = nominate(pool, prefix, fine_ids=fine_ids,
        coarse_ids=t['cascade']['coarse']['selected_ids'], selection=t['cascade']['fine']['selection'],
        count=config.CASCADE_FINE_LIMIT)
    packets, proposals = dict(baseline=prefix), {}
    remaining = 5120-old['added_bytes']
    for mode, allowance in [('core_gap_spare', remaining), ('core_gap_allowance', 4096)]:
        packets[mode], proposals[mode] = reserve(prefix, nominees, **kwargs, reserve_bytes=allowance, limit=1)
        assert packets[mode][:len(prefix)] == prefix
        assert proposals[mode]['added_bytes'] <= allowance
    return dict(packets=packets, proposals=proposals, inventory=inventory, remaining_bytes=remaining)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    with patch('socket.socket.connect', side_effect=AssertionError('Offline only')):
        score(parser.parse_args(), preparer=prepare, modes=MODES, limitation=__doc__, script_path=Path(__file__))
