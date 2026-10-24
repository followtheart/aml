"""Weak promotion bounds must not disable witnessed evidence or CE-failure rescue."""
import copy
import unittest
from unittest.mock import patch

import selftest_fine_selection as selection_tests
from selftest_fine_selection import candidate, source
from app import config, fine_evidence


class SoftPromotionTests(unittest.TestCase):
    def rows(self):
        rows = [candidate(f'r{i}', 1-i/100, [source(str(i), f'Independent passage {i}.')])
                for i in range(30)]
        for row in rows:
            row['_ce_score'] = row['_fixture_score']
        return rows

    def test_zero_and_one_promotion_keep_count_and_sources(self):
        rows = self.rows(); original = copy.deepcopy(rows)
        for limit, expected in [(None, set(range(14)) | {16, 17}),
                                (1, set(range(15)) | {16}), (0, set(range(16)))]:
            selected, trace = fine_evidence.propose(rows, 16, [], fused_ids=['r16','r17'],
                extra_limit=0, soft_promotion_limit=limit)
            self.assertEqual({r['id'] for r in selected}, {f'r{i}' for i in expected})
            self.assertEqual(len(selected), 16)
            self.assertEqual(trace['soft_reservation_policy']['limit'], 2 if limit is None else limit)
        self.assertEqual(rows, original)

    def test_zero_weak_promotions_preserves_strong_witness_and_unscored_rescue(self):
        rows = self.rows()
        rows[-1]['_supported_coverage_ids'] = ['need']
        rows[-2]['_ce_score'] = None
        selected, trace = fine_evidence.propose(rows, 16, [dict(id='need', text='Required detail')],
            fused_ids=['r16','r17','r29'], rescue_ids=['r28'], extra_limit=0, soft_promotion_limit=0)
        self.assertTrue({'r28','r29'} <= {r['id'] for r in selected})
        self.assertEqual(len(selected), 16)
        self.assertEqual(set(trace['adaptive']['core_ids']), {r['id'] for r in selected})

    def test_dependencies_still_use_bounded_supplemental_seats(self):
        rows = self.rows(); rows[0]['_relation_path_ids'] = ['r0','r25']
        selected, trace = fine_evidence.propose(rows, 16, [], fused_ids=['r16','r17'],
            extra_limit=1, soft_promotion_limit=0)
        self.assertEqual(len(selected), 17)
        self.assertIn('r25', {r['id'] for r in selected})
        self.assertEqual(trace['adaptive']['additions'][0]['reason'], 'path_completion')

    def test_invalid_limits_are_rejected(self):
        for limit in (-1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                fine_evidence.propose(self.rows(), 16, [], soft_promotion_limit=limit)


class SoftPromotionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_flag_limits_adaptive_core_only(self):
        helper = selection_tests.CascadeCoverageSelectionTests()
        helper.setUp(); self.addCleanup(helper.doCleanups)
        rows = SoftPromotionTests().rows()
        rows[16]['_fused'] = 3; rows[17]['_fused'] = 2
        for adaptive, enabled, limit in [(True,False,2),(True,True,1),(False,True,2)]:
            with self.subTest(adaptive=adaptive, enabled=enabled), patch.multiple(config,
                    CASCADE_FINE_LIMIT=16, CASCADE_LLM_LIMIT=16, CASCADE_FUSED_RESERVE=2,
                    FINE_EVIDENCE_ADAPTIVE=adaptive, FINE_EXTRA_LIMIT=8,
                    FINE_COARSE_CORE_QUOTA=False, COARSE_EVIDENCE_EXTEND=False,
                    FINE_WEAK_PROMOTION_ONE=enabled):
                _, plan, _ = await helper.run_cascade(rows, [])
            fine = plan['_cascade']['fine']
            self.assertEqual(fine['selection']['soft_reservation_policy']['limit'],limit)
            core = fine['selection'].get('adaptive',{}).get('core_ids',fine['selected_ids'])
            self.assertEqual(len(set(core)&{'r16','r17'}),limit)


if __name__ == '__main__':
    unittest.main()
