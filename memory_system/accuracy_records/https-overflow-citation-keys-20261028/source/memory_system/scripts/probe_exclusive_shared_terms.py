"""Compare shared option terms exclusive to one prepared source with marginal RRF.

This is a stricter follow-up to shared-option term rarity: an unseen term must
occur in at least two original options and exactly one original observation in
the prepared coarse corpus. It nominates a complete carrier, not a user fact.
Use the same original prefix, remaining 5 KiB budget and one appended item.
The comparator is the existing literal-query marginal residual algorithm.
"""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import main as score
from probe_shared_option_terms import prepare as original_prepare
from app import marginal_query_context, search_pipeline
from app.fine_context_reserve import propose as reserve

MODES = ('exclusive_shared', 'literal_marginal')


def prepare(trace):
    original = original_prepare(trace)
    prefix = original['packets']['baseline']
    detail = original['proposals']['shared_options']
    eligible_terms = {term for term, count in detail['source_frequency'].items()
        if count == 1 and detail['option_frequency'][term] >= 2}
    pool = {c['id']: c for c in search_pipeline._foresight_filter(
        copy.deepcopy(trace['ranked']), copy.deepcopy(trace['plan']))
        if c.get('view_status') != 'stale' and '_packet_item' in c}
    inventory = copy.deepcopy(original['inventory'])
    for candidate in inventory:
        candidate['exclusive_shared_terms'] = sorted(eligible_terms & set(candidate['matched_terms']['shared_options']))
    ordered = sorted([c for c in inventory if c['exclusive_shared_terms']],
        key=lambda c: (-len(c['exclusive_shared_terms']), -c['ce_score'], c['candidate_id']))
    marginal, nomination = marginal_query_context.nominate(pool, prefix,
        routes=trace['routes'], query_specs=trace['query_specs'], query=trace['query'],
        options=trace['options'], coarse_ids=trace['cascade']['coarse']['selected_ids'],
        selected_ids=trace['selection']['selected_ids'], user_id=trace['user_id'])
    packets, proposals = {'baseline': prefix}, {}
    for mode, candidates in [('exclusive_shared', [pool[c['candidate_id']] for c in ordered]),
                             ('literal_marginal', marginal)]:
        packets[mode], proposals[mode] = reserve(prefix, candidates, user_id=trace['user_id'],
            top_k=trace['top_k'], byte_budget=trace['coverage_manifest']['token_budget'],
            reserve_bytes=original['remaining_bytes'], limit=1)
        proposals[mode]['ordered_ids'] = [c['id'] for c in candidates]
        if packets[mode][:len(prefix)] != prefix:
            raise ValueError('Prefix changed')
    proposals['exclusive_shared'].update(corpus_observations=detail['corpus_observations'],
        eligible_terms=sorted(eligible_terms), source_frequency=detail['source_frequency'],
        option_frequency=detail['option_frequency'])
    proposals['literal_marginal']['nomination'] = nomination
    return dict(packets=packets, proposals=proposals, inventory=inventory,
        remaining_bytes=original['remaining_bytes'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=Path('data/raw/personamem-v2.csv'))
    with patch('socket.socket.connect', side_effect=AssertionError('Offline exclusive shared source study')):
        score(parser.parse_args(), preparer=prepare, modes=MODES, limitation=__doc__, script_path=Path(__file__))
