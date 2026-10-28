"""Async admission reservations, fallbacks and request isolation; no network."""
import asyncio,copy,sys,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget,config,joint_source_admission as module

class ExecutorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.flags=patch.multiple(config,JOINT_SOURCE_RANK_ADMISSION=True,FINE_EVIDENCE_ADAPTIVE=True,
            LISTWISE_STAGED=False,SOURCE_CE_LATE_ADMISSION=False,RERANK_DEADLINE_SECONDS=1,CE_DEADLINE_SECONDS=2)
        self.flags.start();self.addCleanup(self.flags.stop)
        self.limits=budget.Budget(time.monotonic()+20,10,10000)
        token=budget.current.set(self.limits);self.addCleanup(budget.current.reset,token)
        self.req=SimpleNamespace(user_id='u');self.rows=[{'id':'a'},{'id':'b'}]
        self.trace=dict(coarse={},fine=dict(selected_ids=['a'],selection=dict(reservations=[],decisions=[])),
            omitted=[dict(id='b',stage='fine',reason='old')],reservation_missing=[])
        self.plan={'_cascade':self.trace};self.cascade=SimpleNamespace()
        self.job=dict(ce_query='query',protected_ids=[],source=dict(views={'v':{'text':'Full user source'}},members={'b':['v']}),
            batches=[['v']],batch_bounds=[100],main_reservation=500,scheduling=dict(eligible=True,reason=None))
        self.preparation=patch.object(module.joint_source_selection,'prepare',return_value=self.job)
        self.preparation.start();self.addCleanup(self.preparation.stop)
        self.selection=patch.object(module.joint_source_selection,'select',side_effect=self.result)
        self.selection.start();self.addCleanup(self.selection.stop)
        self.provider=patch.object(module.cross_encoder,'rerank',side_effect=self.score)
        self.provider.start();self.addCleanup(self.provider.stop)
    async def score(self,query,texts,**kwargs):
        current=budget.current.get();self.assertEqual(current.reserved_calls,1);self.assertEqual(current.reserved_tokens,500)
        self.assertEqual(kwargs['rate_limit_retries'],0);current.before_call();current.tokens+=7
        await asyncio.sleep(0)
        return [.5]
    def result(self,*args,**kwargs):
        coarse=args[2]
        return dict(status='selected',reason=None,fine=coarse,submitted=[coarse[1]],plan={'_fine_path_groups':[]},
            proposal={},guard={},protected_fine_ids=[],mandatory_main_ids=[],decisions=[],rejections=[],
            request=dict(prompt='complete',reservation=500),selection_seconds=0)
    async def run_apply(self):
        return await module.apply(self.req,self.plan,self.rows,self.rows[:1],self.rows[:1],set(),{'a'},{},self.cascade)
    def assert_released(self):
        self.assertEqual((self.limits.reserved_calls,self.limits.reserved_tokens),(0,0))
    async def test_disabled_preserves_all_state_without_provider(self):
        with patch.object(config,'JOINT_SOURCE_RANK_ADMISSION',False):
            before=copy.deepcopy(self.trace);result=await self.run_apply()
        self.assertEqual(result[1],[self.rows[0]])
        self.assertEqual(self.trace['fine'],before['fine']);self.assertEqual(self.limits.calls,0)
        self.assertEqual(self.trace['joint_source_admission']['status'],'disabled')
    async def test_success_commits_actual_members_and_keeps_original_diagnostics(self):
        trace=self.plan['_cascade'];result=await self.run_apply()
        self.assertIs(self.plan['_cascade'],trace);self.assertEqual(result[1],[self.rows[1]])
        self.assertEqual(trace['fine']['selected_ids'],['a','b'])
        self.assertEqual(trace['joint_source_admission']['original_stages']['fine']['selected_ids'],['a'])
        self.assertEqual([(r['id'],r['stage']) for r in trace['omitted']],[('a','listwise')])
        self.assertEqual((self.limits.calls,self.limits.tokens),(1,7));self.assert_released()
    async def test_provider_failure_falls_back_without_refunding_cost(self):
        async def fail(*a,**k):self.limits.before_call();self.limits.tokens+=3;raise ValueError('provider')
        with patch.object(module.cross_encoder,'rerank',side_effect=fail):result=await self.run_apply()
        self.assertEqual(result[1],self.rows[:1]);self.assertEqual(self.limits.tokens,3);self.assert_released()
        self.assertEqual(self.trace['joint_source_admission']['batches'][0]['status'],'error')
    async def test_cancel_releases_reservations_and_propagates(self):
        async def cancel(*a,**k):raise asyncio.CancelledError()
        with patch.object(module.cross_encoder,'rerank',side_effect=cancel):
            with self.assertRaises(asyncio.CancelledError):await self.run_apply()
        self.assert_released();self.assertEqual(self.trace['fine']['selected_ids'],['a'])
    async def test_timeout_releases_reservations_and_retains_original(self):
        async def wait(*a,**k):await asyncio.Event().wait()
        with patch.object(config,'CE_DEADLINE_SECONDS',.01),patch.object(module.cross_encoder,'rerank',side_effect=wait):
            result=await self.run_apply()
        self.assertEqual(result[1],self.rows[:1]);self.assert_released()
        self.assertEqual(self.trace['joint_source_admission']['reason'],'TimeoutError')
    async def test_partial_reservation_failure_releases_call_slot(self):
        with patch.object(self.limits,'reserve_tokens',side_effect=budget.BudgetExceeded('full')):
            result=await self.run_apply()
        self.assertEqual(result[1],self.rows[:1]);self.assertEqual(self.limits.calls,0);self.assert_released()
    async def test_incomplete_batch_does_not_select(self):
        with patch.object(module.cross_encoder,'rerank',return_value=[]):result=await self.run_apply()
        self.assertEqual(result[1],self.rows[:1]);self.assert_released()
    async def test_post_selection_deadline_rejects_proposal(self):
        def late(*a,**k):
            result=self.result(*a,**k);self.limits.deadline=time.monotonic()+.1;return result
        with patch.object(module.joint_source_selection,'select',side_effect=late):result=await self.run_apply()
        self.assertEqual(result[1],self.rows[:1]);self.assert_released()
    async def test_constraint_fallback_keeps_original(self):
        with patch.object(module.joint_source_selection,'select',return_value=dict(status='fallback',reason='constraint_catalog_loss')):
            result=await self.run_apply()
        self.assertEqual(result[1],self.rows[:1]);self.assert_released()
    async def test_preflight_budget_skip_has_no_calls(self):
        self.job['scheduling']=dict(eligible=False,reason='token_headroom')
        result=await self.run_apply();self.assertEqual(result[1],self.rows[:1]);self.assertEqual(self.limits.calls,0)
    async def test_interleaved_requests_keep_separate_budgets_and_candidates(self):
        async def one(prefix):
            limits=budget.Budget(time.monotonic()+20,10,10000);token=budget.current.set(limits)
            try:
                rows=[{'id':prefix+'0'},{'id':prefix+'1'}]
                result,details=await module.propose(self.req,self.plan,rows,rows[:1],rows[:1],set(),self.cascade)
                self.assertEqual(result['submitted'],[rows[1]])
                self.assertEqual((limits.calls,limits.tokens,limits.reserved_calls,limits.reserved_tokens),(1,7,0,0))
                return details['original_main_ids']
            finally:budget.current.reset(token)
        self.assertEqual(await asyncio.gather(one('x'),one('y')),[['x0'],['y0']])
if __name__=='__main__':unittest.main()
