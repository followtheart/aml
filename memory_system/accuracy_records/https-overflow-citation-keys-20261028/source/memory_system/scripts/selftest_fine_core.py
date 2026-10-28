"""Preserve the original coarse pool's core without bypassing fine admission."""
import copy
import unittest
from unittest.mock import patch

import selftest_fine_selection as selection_tests
from selftest_fine_selection import candidate, source
from app import cascade_rerank as cascade, config, fine_evidence


class FineCoreTests(unittest.TestCase):
    def rows(self):
        rows=[candidate(f'r{i}',.8-i*.01,[source(f's{i}',f'I study market detail {i}.')]) for i in range(6)]
        for r in rows:r['_ce_score']=r['_fixture_score']
        extra=candidate('extra',.99,[source('new','I study a new market detail.')])
        extra['_ce_score']=.99
        return rows+[extra]

    def test_new_top_score_competes_only_for_extra_seat(self):
        rows=self.rows(); before=copy.deepcopy(rows)
        normal,_=fine_evidence.propose(rows,3,[],extra_limit=1)
        proposed,trace=fine_evidence.propose(rows,3,[],extra_limit=1,core_pool_ids=[r['id'] for r in rows[:-1]])
        self.assertEqual(normal[0]['id'],'extra')
        self.assertEqual([r['id'] for r in proposed],['r0','r1','r2','extra'])
        self.assertEqual({d['id'] for d in trace['decisions']},{r['id'] for r in rows})
        self.assertEqual(rows,before)

    def test_complete_dependency_uses_extra_seat_before_optional_gain(self):
        rows=self.rows();rows[0]['_relation_path_ids']=['r0','extra']
        proposed,trace=fine_evidence.propose(rows,3,[],extra_limit=1,core_pool_ids=[r['id'] for r in rows[:-1]])
        self.assertEqual({r['id'] for r in proposed},{'r0','r1','r2','extra'})
        self.assertEqual(trace['adaptive']['additions'][0]['reason'],'path_completion')
        self.assertEqual(trace['adaptive']['path_groups'],[['extra','r0']])

    def test_extra_limit_and_invalid_pool_are_enforced(self):
        rows=self.rows()
        proposed,_=fine_evidence.propose(rows,3,[],extra_limit=0,core_pool_ids=[r['id'] for r in rows[:-1]])
        self.assertEqual([r['id'] for r in proposed],['r0','r1','r2'])
        with self.assertRaises(ValueError):fine_evidence.propose(rows,3,[],core_pool_ids=['missing'])

    def test_unscored_rescue_from_original_pool_stays_in_core(self):
        rows=self.rows();rows[5]['_ce_score']=None
        proposed,trace=fine_evidence.propose(rows,3,[],extra_limit=1,rescue_ids=['r5'],core_pool_ids=[r['id'] for r in rows[:-1]])
        self.assertIn('r5',trace['adaptive']['core_ids'])
        self.assertEqual(len(proposed),4)


class FineCoreIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_flag_uses_current_baseline_pool_and_does_nothing_without_extension(self):
        helper=selection_tests.CascadeCoverageSelectionTests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        rows=[candidate(f'r{i}',1-i*.01,[source(f's{i}',f'I study market detail {i}.')]) for i in range(22)]
        rows[18]['_fixture_score']=2.0
        with patch.multiple(config,CASCADE_COARSE_LIMIT=16,CASCADE_FINE_LIMIT=8,CASCADE_LLM_LIMIT=8,
                FINE_EVIDENCE_ADAPTIVE=True,FINE_EXTRA_LIMIT=4,FINE_COARSE_CORE_QUOTA=True,
                COARSE_EVIDENCE_EXTEND=True):
            _,plan,_=await helper.run_cascade(rows,[])
        trace=plan['_cascade'];a=trace['fine']['selection']['adaptive']
        self.assertEqual(a['core_policy'],'restricted_core_pool_v1')
        self.assertEqual(set(a['core_ids'])-set(trace['coarse']['baseline_ids']),{'r18'})
        self.assertEqual(a['new_coarse_core_limit'],1)
        self.assertIn('r18',trace['fine']['selected_ids'])
        with patch.multiple(config,FINE_COARSE_CORE_QUOTA=True,COARSE_EVIDENCE_EXTEND=False,
                            FINE_EVIDENCE_ADAPTIVE=True):
            _,plan,_=await helper.run_cascade(rows,[])
        self.assertNotIn('core_policy',plan['_cascade']['fine']['selection']['adaptive'])


if __name__=='__main__':unittest.main()
