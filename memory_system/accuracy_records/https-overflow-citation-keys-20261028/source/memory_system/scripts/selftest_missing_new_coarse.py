"""Missing-only nomination must not promote an existing fine candidate."""
import copy
import unittest
from unittest.mock import patch

import selftest_new_coarse_insurance as insurance_tests
from selftest_fine_selection import source,candidate
import selftest_fine_selection as selection_tests
from app import config,fine_evidence


class MissingOnlyTests(unittest.TestCase):
    def fixture(self):return insurance_tests.NewCoarseInsuranceTests().fixture()

    def select(self,rows,enabled,**kw):
        options=dict(extra_limit=2,core_pool_ids=set('abcdefgh'),new_coarse_missing_only=enabled)
        options.update(kw)
        return fine_evidence.propose(rows,4,[dict(id='question',text='market analysis')],**options)

    def replace(self,rows,index,text):
        mid=rows[index]['id'];value=rows[index]['_ce_score']
        rows[index]=candidate(mid,value,[source(mid,text)]);rows[index]['_ce_score']=value

    def test_existing_fine_head_cannot_take_insurance_seat(self):
        rows=self.fixture();self.replace(rows,0,'I read market analysis x.')
        frozen=copy.deepcopy(rows);baseline,_=self.select(rows,False);new,trace=self.select(rows,True)
        self.assertIn('x',[c['id'] for c in baseline])
        info=trace['adaptive']['new_coarse_insurance']
        self.assertNotIn('x',info['eligible_ids']);self.assertIn('y',info['eligible_ids'])
        used=[a['candidate_id'] for a in trace['adaptive']['additions'] if a['reason']=='new_coarse_core_insurance']
        self.assertEqual(used,['y']);self.assertEqual(rows,frozen)

    def test_all_heads_already_selected_is_exact_noop(self):
        rows=self.fixture()
        for i in (0,1):self.replace(rows,i,'I read market analysis '+rows[i]['id'])
        baseline,old=self.select(rows,False);new,trace=self.select(rows,True)
        self.assertEqual(new,baseline)
        self.assertFalse(trace['adaptive']['new_coarse_insurance']['used'])
        self.assertEqual(trace['adaptive']['additions'],old['adaptive']['additions'])
        self.assertEqual(trace['adaptive']['new_coarse_insurance']['baseline_fine_ids'],[c['id'] for c in baseline])

    def test_new_missing_candidate_keeps_core_and_capacity(self):
        rows=self.fixture();baseline,old=self.select(rows,False);new,trace=self.select(rows,True)
        self.assertEqual(trace['adaptive']['core_ids'],old['adaptive']['core_ids'])
        self.assertEqual(trace['adaptive']['capacity'],old['adaptive']['capacity'])
        self.assertEqual(len(new),len(baseline))
        self.assertTrue(trace['adaptive']['new_coarse_insurance']['missing_only'])
        self.assertIn('x',[c['id'] for c in new]);self.assertNotIn('x',[c['id'] for c in baseline])

    def test_missing_dependency_does_not_split_or_promote(self):
        rows=self.fixture()
        for c in rows[:2]:c['_relation_path_ids']=[c['id'],'absent']
        baseline,_=self.select(rows,False);new,trace=self.select(rows,True)
        self.assertEqual(new,baseline);self.assertFalse(trace['adaptive']['new_coarse_insurance']['used'])

    def test_zero_extra_or_no_quota_does_not_create_capacity(self):
        for kw in ({'extra_limit':0},{'core_pool_ids':None}):
            rows=self.fixture();baseline,_=self.select(rows,False,**kw);new,trace=self.select(rows,True,**kw)
            self.assertEqual(new,baseline);self.assertFalse(trace['adaptive']['new_coarse_insurance']['used'])

    def test_old_insurance_is_unchanged_when_missing_only_off(self):
        rows=self.fixture();options=dict(extra_limit=2,core_pool_ids=set('abcdefgh'),new_coarse_insurance=True)
        public=fine_evidence.propose(rows,4,[],**options)
        private=fine_evidence._propose(rows,4,[],**options)
        self.assertEqual(public,private)


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_only_has_independent_default_off_switch(self):
        helper=selection_tests.CascadeCoverageSelectionTests();helper.setUp();self.addCleanup(helper.doCleanups)
        rows=[candidate(f'r{i}',1-i*.01,[source(f's{i}',f'Independent passage {i}.')]) for i in range(30)]
        for adaptive,enabled in ((True,False),(True,True),(False,True)):
            with self.subTest(adaptive=adaptive,enabled=enabled),patch.multiple(config,
                    FINE_EVIDENCE_ADAPTIVE=adaptive,FINE_NEW_COARSE_INSURANCE=False,
                    FINE_NEW_COARSE_MISSING_ONLY=enabled,FINE_SEMANTIC_RESERVE=False,
                    COARSE_EVIDENCE_EXTEND=True,FINE_COARSE_CORE_QUOTA=True):
                _,plan,_=await helper.run_cascade(rows,[])
            trace=plan['_cascade']['fine']['selection'].get('adaptive',{})
            self.assertEqual('new_coarse_insurance' in trace,adaptive and enabled)
            if adaptive and enabled:self.assertTrue(trace['new_coarse_insurance']['missing_only'])


if __name__=='__main__':unittest.main()
