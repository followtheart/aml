"""High CE heads excluded by the quota share one bounded extra seat."""
import copy
import unittest
from unittest.mock import patch

import selftest_fine_selection as selection_tests
from selftest_fine_selection import candidate,source
from app import config,fine_evidence


class NewCoarseInsuranceTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for i,mid in enumerate('xyabcdefgh'):
            text=f'独立原话 {mid}。' if mid in 'xy' else f'I read market analysis {mid}.'
            row=candidate(mid,1-i*.05,[source(mid,text)]);row['_ce_score']=1-i*.05;rows.append(row)
        return rows

    def select(self,rows,enabled,**kw):
        options=dict(extra_limit=2,core_pool_ids=set('abcdefgh'),new_coarse_insurance=enabled)
        options.update(kw)
        return fine_evidence.propose(rows,4,[dict(id='question',text='market analysis')],**options)

    def test_one_seat_without_changing_core_sources_or_scores(self):
        rows=self.fixture();frozen=copy.deepcopy(rows)
        old,old_trace=self.select(rows,False);new,trace=self.select(rows,True)
        self.assertNotIn('x',[r['id'] for r in old]);self.assertIn('x',[r['id'] for r in new])
        self.assertNotIn('y',[r['id'] for r in new])
        self.assertEqual(trace['adaptive']['core_ids'],old_trace['adaptive']['core_ids'])
        self.assertEqual(len(new),6);self.assertEqual(rows,frozen)
        self.assertEqual(trace['adaptive']['new_coarse_insurance']['eligible_ids'],['x','y'])
        self.assertEqual(sum(a['reason']=='new_coarse_core_insurance' for a in trace['adaptive']['additions']),1)

    def test_no_core_quota_means_no_eligible_candidates(self):
        rows=self.fixture();old,_=self.select(rows,False,core_pool_ids=None)
        new,trace=self.select(rows,True,core_pool_ids=None)
        self.assertEqual(old,new);self.assertEqual(trace['adaptive']['new_coarse_insurance']['eligible_ids'],[])

    def test_hidden_unscored_and_below_cutoff_are_ineligible(self):
        for mode in ('hidden','unscored','low'):
            rows=self.fixture()
            if mode=='hidden':rows[0]['_rank_text']='Hidden original'
            elif mode=='unscored':rows[0]['_ce_score']=None
            else:rows[0]['_ce_score']=0
            _,trace=self.select(rows,True)
            self.assertNotIn('x',trace['adaptive']['new_coarse_insurance']['eligible_ids'])

    def test_zero_extra_and_core_dependencies_have_priority(self):
        rows=self.fixture();old,_=self.select(rows,False,extra_limit=0)
        new,trace=self.select(rows,True,extra_limit=0)
        self.assertEqual(old,new);self.assertFalse(trace['adaptive']['new_coarse_insurance']['used'])
        rows[2]['_relation_path_ids']=['a','h']
        new,trace=self.select(rows,True,extra_limit=1)
        self.assertIn('h',[r['id'] for r in new]);self.assertFalse(trace['adaptive']['new_coarse_insurance']['used'])

    def test_no_split_dependency_group(self):
        rows=self.fixture()
        for row in rows[:2]:row['_relation_path_ids']=[row['id'],'h']
        _,trace=self.select(rows,True,extra_limit=1)
        self.assertFalse(trace['adaptive']['new_coarse_insurance']['used'])

    def test_duplicate_visible_source_is_not_a_gain(self):
        rows=self.fixture()
        rows[0]['_packet_item']=copy.deepcopy(rows[2]['_packet_item']);rows[0]['_rank_text']=rows[2]['_rank_text']
        _,trace=self.select(rows,True)
        self.assertNotIn('x',[a['candidate_id'] for a in trace['adaptive']['additions'] if a['reason']=='new_coarse_core_insurance'])

    def test_both_insurance_policies_share_one_extra_seat(self):
        rows=self.fixture()
        _,trace=self.select(rows,True,semantic_reserve=True,rescue_ids={'g','h'})
        self.assertEqual(sum(a['reason'] in ('ce_core_insurance','new_coarse_core_insurance') for a in trace['adaptive']['additions']),1)


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_flag_reaches_only_adaptive_fine(self):
        helper=selection_tests.CascadeCoverageSelectionTests();helper.setUp();self.addCleanup(helper.doCleanups)
        rows=[candidate(f'r{i}',1-i*.01,[source(f's{i}',f'Independent passage {i}.')]) for i in range(30)]
        for adaptive,enabled in ((True,False),(True,True),(False,True)):
            with self.subTest(adaptive=adaptive,enabled=enabled),patch.multiple(config,
                    FINE_EVIDENCE_ADAPTIVE=adaptive,FINE_NEW_COARSE_INSURANCE=enabled,
                    FINE_SEMANTIC_RESERVE=False,COARSE_EVIDENCE_EXTEND=True,FINE_COARSE_CORE_QUOTA=True):
                _,plan,_=await helper.run_cascade(rows,[])
            trace=plan['_cascade']['fine']['selection'].get('adaptive',{})
            self.assertEqual('new_coarse_insurance' in trace,adaptive and enabled)


if __name__=='__main__':unittest.main()
