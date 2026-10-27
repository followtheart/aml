"""Safety checks for task-local ordering and reserved source-scoring budgets."""
import asyncio
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, fine_evidence, response_observer
from app import fine_source_rrf as runtime, user_source_order


class IsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_isolation_nested_scopes_and_cancellation(self):
        a, b = dict(id='a', _ce_score=2, _fused=1), dict(id='b', _ce_score=1, _fused=0)
        ready, finish = asyncio.Event(), asyncio.Event()
        async def scoped():
            with fine_evidence.ordered_by({'a':1, 'b':2}):
                self.assertGreater(fine_evidence.key(b), fine_evidence.key(a))
                ready.set()
                await finish.wait()
        task = asyncio.create_task(scoped())
        await ready.wait()
        self.assertGreater(fine_evidence.key(a), fine_evidence.key(b))
        with fine_evidence.ordered_by({'a':1, 'b':2}):
            with fine_evidence.ordered_by(None):
                self.assertGreater(fine_evidence.key(a), fine_evidence.key(b))
            self.assertGreater(fine_evidence.key(b), fine_evidence.key(a))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(fine_evidence._rank_override.get())
        self.assertGreater(fine_evidence.key(a), fine_evidence.key(b))


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.job = dict(source=dict(views={'a':dict(text='a'), 'b':dict(text='b')},
                                   members={'a':['a'], 'b':['b']}),
                        protected_ids=[], query='q', batches=[['a'], ['b']],
                        main_reservation=100, conservative_bound=4, seconds=10, reason=None)
        self.baseline = dict(fine=[dict(id='a')], submitted=[dict(id='a')])
        self.config = dict(FAKE=False, CE_MODEL='qwen3-rerank', LLM_MODEL='gpt-4o-mini',
                           CE_API_FORMAT='cohere', OPTION_MARGIN_SOURCE_ADMISSION=False,
                           JOINT_SOURCE_RANK_ADMISSION=False, SOURCE_CE_LATE_ADMISSION=False,
                           RERANK_DEADLINE_SECONDS=10)

    async def invoke(self, *, failure=None, tokens=120, calls=3):
        plan, seen = {'_cascade':{}}, []
        original = copy.deepcopy(self.baseline)
        selected = dict(plan={'_cascade':{}}, fine=[dict(id='b')], submitted=[dict(id='b')])
        async def rerank(query, texts, *, stage, rate_limit_retries):
            self.assertEqual(rate_limit_retries, 0)
            self.assertEqual(stage, 'search.fine_source_rrf')
            limits = budget.current.get()
            self.assertEqual((limits.reserved_calls, limits.reserved_tokens), (1, 100))
            seen.append(texts)
            limits.before_call()
            limits.tokens += 2
            if failure == 'cancel':
                raise asyncio.CancelledError()
            if failure == 'partial' and len(seen) == 2:
                raise ValueError('invalid provider response')
            if failure == 'deadline':
                limits.deadline = __import__('time').monotonic() + 1
            response_observer.record('rerank', {'results':[dict(index=0, relevance_score=.9)]},
                                     runtime.wire_body(query, texts))
            return [.9]
        with patch.multiple(config, **self.config), \
             patch.object(runtime, 'prepare', return_value=copy.deepcopy(self.job)), \
             patch.object(runtime.cross_encoder, 'rerank', side_effect=rerank), \
             patch.object(runtime, 'propose', return_value=(selected, {}, {}, None)) as propose, \
             budget.scope(seconds=100, calls=calls, tokens=tokens) as limits:
            try:
                result = await runtime.apply(SimpleNamespace(), plan, [], self.baseline, SimpleNamespace())
            finally:
                self.assertEqual((limits.reserved_calls, limits.reserved_tokens), (0, 0))
                self.assertEqual(self.baseline, original)
                self.assertIsNone(response_observer._sink.get())
        return result, plan['_cascade']['fine_source_rrf'], seen, limits, propose.call_count

    async def test_serial_calls_preserve_main_and_capture_identity(self):
        result, trace, seen, limits, proposals = await self.invoke()
        self.assertEqual(seen, [['a'], ['b']])
        self.assertEqual(trace['status'], 'selected')
        self.assertIs(result['plan']['_cascade']['fine_source_rrf'], trace)
        self.assertEqual((trace['charged_calls'], trace['charged_tokens']), (2, 4))
        self.assertEqual((limits.calls, limits.tokens, proposals), (2, 4, 1))
        for batch, event in zip(trace['batches'], trace['responses'], strict=True):
            self.assertEqual(batch['request_sha256'], event['request_sha256'])
            self.assertEqual(batch['index'], event['batch'])

    async def test_partial_failure_falls_back_without_proposal(self):
        result, trace, seen, limits, proposals = await self.invoke(failure='partial')
        self.assertIsNone(result)
        self.assertEqual(trace['status'], 'fallback')
        self.assertEqual((len(seen), limits.calls, proposals), (2, 2, 0))

    async def test_cancellation_rethrows_and_cleans_reservations(self):
        with self.assertRaises(asyncio.CancelledError):
            await self.invoke(failure='cancel')

    async def test_token_reservation_failure_releases_call_reservation(self):
        result, trace, seen, _, proposals = await self.invoke(tokens=99)
        self.assertIsNone(result)
        self.assertEqual(trace['reason'], 'BudgetExceeded')
        self.assertEqual((seen, proposals), ([], 0))

    async def test_incremental_budget_never_spends_reserved_main(self):
        result, trace, seen, limits, proposals = await self.invoke(tokens=103)
        self.assertIsNone(result)
        self.assertEqual(seen, [['a']])
        self.assertEqual(proposals, 0)
        self.assertGreaterEqual(limits.max_tokens-limits.tokens, 100)

    async def test_settled_deadline_falls_back_before_selection(self):
        result, trace, _, _, proposals = await self.invoke(failure='deadline')
        self.assertIsNone(result)
        self.assertEqual(trace['reason'], 'settled_deadline')
        self.assertEqual(proposals, 0)

    async def test_skip_has_no_provider_calls(self):
        self.job['reason'] = 'token_headroom'
        result, trace, seen, _, proposals = await self.invoke()
        self.assertIsNone(result)
        self.assertEqual(trace['status'], 'skipped')
        self.assertEqual((seen, proposals), ([], 0))


class PolicyTests(unittest.TestCase):
    def test_full_source_phase_budget_checked_before_dispatch(self):
        source = dict(views={'a':dict(text='a'), 'b':dict(text='b')}, members={'a':['a'], 'b':['b']})
        plan = {'_cascade':{'cross_encoder':{'query':'q'}}}
        cascade = SimpleNamespace(listwise_request=lambda *a: dict(reservation=100))
        cases = [(dict(token_headroom=103, call_headroom=3, deadline_seconds=100), 'token_headroom'),
                 (dict(token_headroom=104, call_headroom=2, deadline_seconds=100), 'call_headroom'),
                 (dict(token_headroom=104, call_headroom=3, deadline_seconds=12), 'deadline_headroom'),
                 (dict(token_headroom=104, call_headroom=3, deadline_seconds=100), None)]
        with patch.object(runtime.user_source_order, 'views', return_value=source), \
             patch.multiple(config, CE_BATCH_SIZE=1, RERANK_DEADLINE_SECONDS=10):
            for headroom, reason in cases:
                job = runtime.prepare(SimpleNamespace(user_id='u'), plan, [], [], cascade, headroom)
                self.assertEqual(job['reason'], reason)
                self.assertEqual(job['batches'], [['a'], ['b']])

    def test_invalid_scores_keep_original_order(self):
        rows = [dict(id='a', _ce_score=2), dict(id='p', _ce_score=1), dict(id='b', _ce_score=0)]
        source = dict(views={'a':{}, 'b':{}}, members={'a':['a'], 'b':['b']})
        for scores in ({'a':1}, {'a':float('nan'), 'b':2}, {'a':True, 'b':2}, {'a':1, 'b':2, 'x':3}):
            result = user_source_order.fused_order(rows, source, scores)
            self.assertEqual(result['status'], 'fallback')
            self.assertEqual(result['ordered_ids'], ['a', 'p', 'b'])

    def test_reader_constraint_requires_same_source_and_original_text(self):
        card = dict(request_id='r', message_index=0, source_event_id='s', role='user', text='forget my address')
        self.assertEqual(runtime.missing_constraints([card], [dict(card, text='Please forget my address now')]), [])
        for replacement in (dict(card, role='assistant'), dict(card, source_event_id='other'), dict(card, text='address')):
            self.assertEqual(runtime.missing_constraints([card], [replacement]), [card])


if __name__ == '__main__':
    unittest.main()
