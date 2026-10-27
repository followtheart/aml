import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import fine_evidence
from fine_type_exchange import choose, apply


def candidate(mid, score, kind='episode', *, source=None, dependencies=()):
    return dict(id=mid, type=kind, _ce_score=score, _fused=0, content=mid, _rank_text=mid,
        _relation_path_ids=list(dependencies),
        _packet_item=dict(sources=[dict(source_event_id=source or mid, role='assistant', content=mid)]))


def metadata(rows, fine, core=(), *, capacity=4):
    return dict(reservations=[], decisions=[dict(id=row['id'], selected=row in fine) for row in rows],
        adaptive=dict(core_ids=list(core), capacity=capacity, proposed_ids=[r['id'] for r in fine],
            rejected=[], additions=[dict(candidate_id=r['id'], ids=[r['id']], group_ids=[r['id']])
                                    for r in fine if r['id'] not in core]))


class TypeExchangeTests(unittest.TestCase):
    def test_higher_cross_type_nominee_does_not_steal_episode_seat(self):
        donor, fact, episode = candidate('d', .1), candidate('f', .9, 'fact'), candidate('e', .8)
        rows = [donor, fact, episode]
        result = choose(rows, [donor], metadata(rows, [donor]), {'d'}, fine_evidence)
        self.assertEqual(result['swap']['added_id'], 'e')

    def test_core_and_selected_reservations_are_protected(self):
        donor, nominee = candidate('d', .1), candidate('n', .9)
        rows = [donor, nominee]
        for core in (True, False):
            meta = metadata(rows, [donor], ['d'] if core else [])
            if not core:
                meta['reservations'] = [dict(candidate_id='d', selected=True)]
            self.assertIsNone(choose(rows, [donor], meta, {'d'}, fine_evidence)['swap'])

    def test_dependencies_and_represented_sources_do_not_replace(self):
        donor = candidate('d', .1)
        dependent = candidate('n', .9, dependencies=['missing'])
        rows = [donor, dependent]
        self.assertIsNone(choose(rows, [donor], metadata(rows, [donor]), {'d'}, fine_evidence)['swap'])
        redundant = candidate('r', .9, source='d')
        redundant['_rank_text'] = redundant['_packet_item']['sources'][0]['content'] = 'd'
        rows = [donor, redundant]
        self.assertIsNone(choose(rows, [donor], metadata(rows, [donor]), {'d'}, fine_evidence)['swap'])

    def test_missing_type_equal_score_and_nonfinite_scores_reject(self):
        donor = candidate('d', .1)
        for kind, score in [(None, .9), ('episode', .1), ('episode', float('nan'))]:
            nominee = candidate('n', score, kind)
            rows = [donor, nominee]
            self.assertIsNone(choose(rows, [donor], metadata(rows, [donor]), {'d'}, fine_evidence)['swap'])

    def test_cutoff_prevents_deep_same_type_promotion(self):
        donor, fact, nominee = candidate('d', .1), candidate('f', .9, 'fact'), candidate('n', .8)
        rows = [donor, fact, nominee]
        self.assertIsNone(choose(rows, [donor], metadata(rows, [donor], capacity=1), {'d'}, fine_evidence)['swap'])

    def test_exchange_keeps_order_types_sources_and_input_objects(self):
        core, donor, tail, nominee = [candidate(k, s) for k, s in [('c', .95), ('d', .1), ('t', .5), ('n', .9)]]
        rows, fine = [core, donor, tail, nominee], [core, donor, tail]
        meta = metadata(rows, fine, ['c'])
        before = copy.deepcopy((rows, fine, meta))
        proposal = choose(rows, fine, meta, {'d'}, fine_evidence)
        result, updated = apply(rows, fine, meta, proposal)
        self.assertEqual([r['id'] for r in result], ['c', 'n', 't'])
        self.assertEqual(result[1], nominee)
        self.assertEqual((rows, fine, meta), before)
        self.assertEqual(updated['adaptive']['additions'][0]['candidate_id'], 'n')
        self.assertEqual(apply(rows, fine, meta, dict(swap=None)), (fine, meta))


if __name__ == '__main__':
    unittest.main()
