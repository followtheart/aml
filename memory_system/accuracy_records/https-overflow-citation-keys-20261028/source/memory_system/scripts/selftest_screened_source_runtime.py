import asyncio
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget, config, screened_source_nomination as runtime


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original=[dict(id='original',content='unchanged')]
        self.nominee=dict(id='new',content='complete source')
        self.job=dict(ce_query='q',decisions=[],views={'a':dict(text='a'),'b':dict(text='b')},
            batches=[['a'],['b']],batch_bounds=[10,10],
            scheduling=dict(eligible=True,main_reservation=100),
            nominees=[dict(id='new',ce_rank=20,source_view_ids=['a','b'],candidate=self.nominee)],
            original_display_audit=dict(decoded_graph={}))

    async def call(self, *, fail=False, tokens=150, calls=3, cancel=False):
        seen=[]
        async def ce(query,docs,stage,**kw):
            seen.append(docs)
            self.assertEqual(kw,dict(rate_limit_retries=0))
            limits=budget.current.get()
            self.assertEqual((limits.reserved_calls,limits.reserved_tokens),(1,100))
            limits.before_call();limits.tokens+=10
            if len(seen)==2 and fail:raise ValueError('Invalid source response')
            if cancel:raise asyncio.CancelledError()
            return [.9]
        with patch.multiple(config,SOURCE_CE_LATE_ADMISSION=True,LISTWISE_STAGED=False), \
                patch.object(runtime,'prepare',return_value=copy.deepcopy(self.job)), \
                patch.object(runtime.cross_encoder,'rerank',side_effect=ce), \
                patch.object(runtime.local_candidate_display,'request',return_value=(dict(prompt='complete',reservation=100),dict(decoded_graph={}))), \
                budget.scope(seconds=100,calls=calls,tokens=tokens) as limits:
            try:
                result=await runtime.append(SimpleNamespace(),{},[],[],self.original,SimpleNamespace())
            finally:
                self.assertEqual((limits.reserved_calls,limits.reserved_tokens),(0,0))
        return result,seen,limits

    async def test_original_prefix_and_source_scores_do_not_mutate_candidates(self):
        before=copy.deepcopy(self.original)
        (rows,trace),seen,limits=await self.call()
        self.assertEqual(rows,self.original+[self.nominee])
        self.assertEqual(self.original,before)
        self.assertEqual(trace['status'],'admitted')
        self.assertEqual(limits.calls,2)
        self.assertEqual(seen,[['a'],['b']])

    async def test_partial_source_failure_keeps_original_identity_and_display(self):
        (rows,trace),_,_=await self.call(fail=True)
        self.assertIs(rows,self.original)
        self.assertEqual(trace['status'],'fallback')
        self.assertIsNone(trace['addition'])
        self.assertFalse(trace['display_ids'])

    async def test_all_source_budget_required_before_first_call(self):
        for limits in (dict(tokens=119),dict(calls=2)):
            (rows,trace),seen,_=await self.call(**limits)
            self.assertIs(rows,self.original)
            self.assertEqual(trace['status'],'skipped')
            self.assertEqual(seen,[])

    async def test_cancellation_propagates_and_releases_main_reservations(self):
        with self.assertRaises(asyncio.CancelledError):await self.call(cancel=True)

    async def test_default_off_never_prepares_or_calls(self):
        with patch.object(config,'SOURCE_CE_LATE_ADMISSION',False), \
                patch.object(runtime,'prepare',side_effect=AssertionError('disabled')):
            rows,trace=await runtime.append(None,{},[],[],self.original,None)
        self.assertIs(rows,self.original)
        self.assertEqual(trace['status'],'disabled')


if __name__=='__main__':unittest.main()
