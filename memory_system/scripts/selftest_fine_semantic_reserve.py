"""Semantic inspection must preserve core, capacity, provenance and dependencies."""
import copy
import unittest
from unittest.mock import patch

import selftest_fine_selection as selection_tests
from selftest_fine_selection import candidate, source
from app import config, fine_evidence


class SemanticReserveTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for i, mid in enumerate('abcdefgh'):
            text = ('En mi sala hay fotografías de mi familia.' if mid in 'cd'
                    else f'I read market analysis report {mid}.')
            c = candidate(mid, 1-i*.1, [source(mid, text)])
            c['_ce_score'] = 1-i*.1
            rows.append(c)
        return rows

    def select(self, rows, enabled, **kw):
        args = dict(rescue_ids={'e', 'f'}, extra_limit=2, semantic_reserve=enabled)
        args.update(kw)
        return fine_evidence.propose(rows, 4, [dict(id='question', text='market analysis')], **args)

    def test_one_semantic_seat_preserves_core_and_sources(self):
        rows = self.fixture()
        before = copy.deepcopy(rows)
        old, old_trace = self.select(rows, False)
        new, trace = self.select(rows, True)
        self.assertNotIn('c', [r['id'] for r in old])
        self.assertIn('c', [r['id'] for r in new])
        self.assertNotIn('d', [r['id'] for r in new])
        self.assertEqual(old_trace['adaptive']['core_ids'], trace['adaptive']['core_ids'])
        self.assertEqual(len(new), 6)
        self.assertEqual(rows, before)
        self.assertEqual(sum(a['reason']=='ce_core_insurance' for a in trace['adaptive']['additions']), 1)
        self.assertFalse(rows[2].get('_supported_coverage_ids'))

    def test_rank_scope_is_original_core_pool(self):
        rows = self.fixture()
        outside = candidate('outside', 2, [source('outside', 'A distinct semantic match.')])
        outside['_ce_score'] = 2
        new, trace = self.select([outside]+rows, True, core_pool_ids={r['id'] for r in rows})
        self.assertEqual(trace['adaptive']['semantic_reserve']['eligible_ids'], list('abcd'))
        self.assertEqual(next(a['candidate_id'] for a in trace['adaptive']['additions']
                              if a['reason']=='ce_core_insurance'), 'c')

    def test_zero_extra_does_not_change_core(self):
        rows = self.fixture()
        old, _ = self.select(rows, False, extra_limit=0)
        new, trace = self.select(rows, True, extra_limit=0)
        self.assertEqual(old, new)
        self.assertFalse(trace['adaptive']['semantic_reserve']['used'])

    def test_hidden_and_unscored_sources_do_not_get_semantic_seat(self):
        for mode in ('hidden', 'unscored'):
            with self.subTest(mode=mode):
                rows = self.fixture()
                if mode == 'hidden':
                    rows[2]['_rank_text'] = 'Nothing visible here.'
                else:
                    rows[2]['_ce_score'] = None
                _, trace = self.select(rows, True)
                self.assertNotIn('c', trace['adaptive']['semantic_reserve']['eligible_ids'])

    def test_dependency_completion_outranks_semantic_seat(self):
        rows = self.fixture()
        rows[0]['_relation_path_ids'] = ['a', 'g']
        new, trace = self.select(rows, True, extra_limit=1)
        self.assertIn('g', [r['id'] for r in new])
        self.assertNotIn('c', [r['id'] for r in new])
        self.assertEqual(trace['adaptive']['additions'][0]['reason'], 'path_completion')

    def test_semantic_seat_cannot_split_a_dependency_group(self):
        rows = self.fixture()
        rows[2]['_relation_path_ids'] = ['c', 'g']
        new, trace = self.select(rows, True, extra_limit=1)
        self.assertNotIn('c', [r['id'] for r in new])
        self.assertTrue(all(len(a['ids']) == 1 for a in trace['adaptive']['additions']))

    def test_already_represented_source_is_not_semantic_gain(self):
        rows = self.fixture()
        rows[2]['_packet_item'] = copy.deepcopy(rows[0]['_packet_item'])
        rows[2]['_rank_text'] = rows[0]['_rank_text']
        _, trace = self.select(rows, True)
        self.assertNotIn('c', [a['candidate_id'] for a in trace['adaptive']['additions']
                               if a['reason']=='ce_core_insurance'])


class SemanticReserveIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_production_flag_reaches_only_adaptive_fine(self):
        helper = selection_tests.CascadeCoverageSelectionTests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        rows = [candidate(f'r{i}', 1-i*.01, [source(f's{i}', f'Independent passage {i}.')])
                for i in range(30)]
        rows[16]['_fused'] = 3
        rows[17]['_fused'] = 2
        for adaptive, enabled in [(True, False), (True, True), (False, True)]:
            with self.subTest(adaptive=adaptive, enabled=enabled), patch.multiple(config,
                    CASCADE_FINE_LIMIT=16, CASCADE_LLM_LIMIT=16, CASCADE_FUSED_RESERVE=2,
                    FINE_EVIDENCE_ADAPTIVE=adaptive, FINE_EXTRA_LIMIT=8,
                    FINE_COARSE_CORE_QUOTA=False, COARSE_EVIDENCE_EXTEND=False,
                    FINE_SEMANTIC_RESERVE=enabled):
                _, plan, _ = await helper.run_cascade(rows, [])
            selection = plan['_cascade']['fine']['selection'].get('adaptive', {})
            self.assertEqual('semantic_reserve' in selection, adaptive and enabled)
            if adaptive and enabled:
                self.assertTrue(selection['semantic_reserve']['used'])
                self.assertEqual(selection['semantic_reserve']['rank_scope'], 'core_pool')


if __name__ == '__main__':
    unittest.main()
