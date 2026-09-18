"""Cascade provider budgets, cancellation and full evidence mapping; offline."""
import asyncio
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, cascade_rerank as cascade, config, cross_encoder, llm, metrics, schemas
from selftest_graph_cascade import documents_in, permutation, selected


class RerankBudgetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        settings = patch.multiple(config, CASCADE_COARSE_LIMIT=50, CASCADE_FINE_LIMIT=12,
            CASCADE_LLM_LIMIT=10, CE_BATCH_SIZE=8, CE_MAX_REQUEST_BYTES=100000,
            CE_MAX_DOCUMENT_BYTES=20000, RERANK_MAX_PROMPT_BYTES=24000,
            RERANK_REPAIR_MAX_CALLS=1, RERANK_CONCURRENCY=2)
        settings.start()
        self.addCleanup(settings.stop)

    def req(self):
        return schemas.SearchRequest(user_id='u', query='Find the Kyoto evidence', top_k=50)

    def rows(self, count=80):
        return [dict(id=f'm{i}', content=f'Kyoto record {i}', _fused=1-i/(count+1),
                     _rank_text=f'Kyoto evidence {i}: ' + 'Verbatim source context. ' * 28)
                for i in range(count)]

    async def listwise(self, prompt, *args, **kwargs):
        return permutation(len(documents_in(prompt)))

    async def test_large_pool_has_bounded_ce_batches_listwise_prompt_and_concurrency(self):
        active, peak, submitted, prompts = 0, 0, [], []
        async def ce(query, documents, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            submitted.extend(documents)
            try:
                self.assertLessEqual(len(documents), config.CE_BATCH_SIZE)
                self.assertLessEqual(cross_encoder.request_size(query, documents), config.CE_MAX_REQUEST_BYTES)
                await asyncio.sleep(0)
                return [.9] * len(documents)
            finally:
                active -= 1
        async def rank(prompt, *args, **kwargs):
            prompts.append(prompt)
            count = len(documents_in(prompt))
            self.assertLessEqual(count, 10)
            self.assertLessEqual(len(prompt.encode('utf-8')), 24000)
            self.assertLessEqual(kwargs['max_tokens'], 256)
            self.assertEqual(kwargs['attempts'], 1)
            self.assertLessEqual(kwargs['timeout'], 20)
            return permutation(count)
        with patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(llm, 'complete_json', side_effect=rank):
            result = await cascade.rank(self.req(), {}, self.rows())
        self.assertEqual(len(submitted), 50)
        self.assertEqual(len(set(submitted)), 50)
        self.assertLessEqual(peak, 2)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(len(selected(result)), 10)

    async def test_completed_ce_batches_map_scores_to_original_candidate(self):
        async def ce(query, documents, **kwargs):
            if kwargs['stage'].endswith('_1'):
                await asyncio.sleep(.01)
            return [12.5 if 'evidence 37:' in text else -3. for text in documents]
        with patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(llm, 'complete_json', side_effect=self.listwise):
            result = selected(await cascade.rank(self.req(), {}, self.rows()))
        self.assertEqual(result[0]['id'], 'm37')
        self.assertEqual(result[0]['_final'], 12.5)
        self.assertTrue(all(row['_final'] == -3. for row in result[1:]))

    async def test_failed_ce_batch_gets_a_listwise_opportunity_without_discarding_successes(self):
        failed, completed, seen = set(), set(), []
        async def ce(query, documents, **kwargs):
            ids = {f'm{i}' for text in documents for i in re.findall(r'evidence (\d+):', text)}
            if kwargs['stage'].endswith('_1'):
                failed.update(ids)
                raise ConnectionError('offline')
            completed.update(ids)
            await asyncio.sleep(0)
            return [.6] * len(documents)
        async def rank(prompt, *args, **kwargs):
            texts = documents_in(prompt)
            seen.extend(f'm{i}' for text in texts for i in re.findall(r'evidence (\d+):', text))
            return permutation(len(texts))
        plan = {}
        with patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(llm, 'complete_json', side_effect=rank):
            result = selected(await cascade.rank(self.req(), plan, self.rows()))
        self.assertTrue(failed & set(seen))
        self.assertTrue(completed & set(seen))
        self.assertTrue(any(row.get('_final') is None for row in result))
        self.assertTrue(any(row.get('_final') == .6 for row in result))
        self.assertEqual(plan['_cascade']['cross_encoder']['status'], 'partial')

    async def test_ce_deadline_preserves_finished_batches_and_cancels_hanging_work(self):
        cancelled = []
        async def ce(query, documents, **kwargs):
            if kwargs['stage'].endswith('_1'):
                return [.9] * len(documents)
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(kwargs['stage'])
        plan = {}
        with patch.object(config, 'CE_DEADLINE_SECONDS', .025), \
                patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(llm, 'complete_json', side_effect=self.listwise):
            result = selected(await cascade.rank(self.req(), plan, self.rows()))
        self.assertTrue(cancelled)
        self.assertTrue(any(row.get('_final') == .9 for row in result))
        self.assertEqual(plan['_cascade']['cross_encoder']['scored_count'], 8)
        self.assertTrue(all(entry['status'] != 'running'
                            for entry in plan['_cascade']['cross_encoder']['batches']))

    async def test_ce_and_listwise_share_the_provider_call_budget(self):
        paid = []
        async def ce(query, documents, **kwargs):
            budget.current.get().before_call()
            paid.append(kwargs['stage'])
            return [.8] * len(documents)
        async def rank(prompt, *args, **kwargs):
            budget.current.get().before_call()
            paid.append(kwargs['stage'])
            return await self.listwise(prompt)
        with budget.scope(seconds=30, calls=2, tokens=64000) as limits, \
                patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(llm, 'complete_json', side_effect=rank):
            result = selected(await cascade.rank(self.req(), {}, self.rows()))
        self.assertEqual(len(paid), 2)
        self.assertEqual(limits.calls, 2)
        self.assertTrue(result)
        self.assertTrue(any(row.get('_final') == .8 for row in result))

    async def test_ce_usage_can_exhaust_the_shared_token_budget_before_listwise(self):
        paid = []
        async def ce(query, documents, **kwargs):
            limits = budget.current.get()
            limits.before_call()
            paid.append(kwargs['stage'])
            limits.tokens = limits.max_tokens
            return [.7] * len(documents)
        async def rank(prompt, *args, **kwargs):
            budget.current.get().before_call()
            paid.append(kwargs['stage'])
            return await self.listwise(prompt)
        with budget.scope(seconds=30, calls=12, tokens=64000), \
                patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(llm, 'complete_json', side_effect=rank):
            result = selected(await cascade.rank(self.req(), {}, self.rows(4)))
        self.assertEqual(paid, ['search.cross_encoder.batch_1'])
        self.assertTrue(result)
        self.assertTrue(all(row['_final'] == .7 for row in result))

    async def test_expired_shared_deadline_never_starts_a_provider(self):
        with budget.scope(seconds=-1, calls=12, tokens=64000), \
                patch.object(cross_encoder, 'rerank', AsyncMock()) as ce, \
                patch.object(llm, 'complete_json', AsyncMock()) as rank:
            with self.assertRaises(TimeoutError):
                await cascade.rank(self.req(), {}, self.rows())
        ce.assert_not_awaited()
        rank.assert_not_awaited()

    async def test_listwise_checks_full_token_reservation_before_dispatch(self):
        plan = {}
        rows = [dict(id='m0', content='Kyoto evidence', _fused=1.)]
        # Enough for the CE input and positive tokens remain, but not enough
        # for the listwise prompt, schema, framing and output allowance.
        with budget.scope(seconds=30, calls=12, tokens=1000) as limits, \
                patch.object(cross_encoder, 'rerank', AsyncMock(return_value=[.7])) as ce, \
                patch.object(llm, 'complete_json', AsyncMock()) as rank:
            result = selected(await cascade.rank(self.req(), plan, rows))
        ce.assert_awaited_once()
        rank.assert_not_awaited()
        self.assertEqual(limits.tokens, 0)
        self.assertEqual(limits.reserved_tokens, 0)
        self.assertEqual(plan['_cascade']['listwise']['status'], 'fallback')
        self.assertEqual(plan['_cascade']['listwise']['errors'], [
            {'stage': 'listwise', 'error': 'BudgetExceeded'}])
        self.assertEqual([row['id'] for row in result], ['m0'])
        self.assertEqual(result[0]['_final'], .7)

    async def test_oversized_single_source_is_not_sent_or_silently_clipped(self):
        text = '原始引用。' * 10000
        rows = [dict(id='oversized', content='quote', _rank_text=text)]
        plan = {}
        with patch.object(cross_encoder, 'rerank', AsyncMock()) as ce, \
                patch.object(llm, 'complete_json', AsyncMock()) as rank:
            result = await cascade.rank(self.req(), plan, rows)
        ce.assert_not_awaited()
        rank.assert_not_awaited()
        self.assertEqual(result[0]['_rank_text'], text)
        self.assertEqual(selected(result), [])
        self.assertTrue(plan['_cascade']['cross_encoder']['errors'])

    async def test_long_units_reduce_listwise_pool_without_clipping_source_text(self):
        rows = self.rows(8)
        for row in rows:
            row['_rank_text'] = row['content'] + ' ' + '完整原文段落。' * 160 + ' END-' + row['id']
        originals = {row['id']: row['_rank_text'] for row in rows}
        submitted = []
        async def ce(query, documents, **kwargs):
            self.assertTrue(all(text in originals.values() for text in documents))
            return [.8] * len(documents)
        async def rank(prompt, *args, **kwargs):
            self.assertLessEqual(len(prompt.encode('utf-8')), 24000)
            texts = documents_in(prompt)
            submitted.extend(texts)
            for text in texts:
                mid = re.search(r'\[id: ([^\]]+)\]', text)[1]
                self.assertTrue(text.endswith(originals[mid]))
            return permutation(len(texts))
        plan = {}
        with patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(llm, 'complete_json', side_effect=rank):
            result = selected(await cascade.rank(self.req(), plan, rows))
        self.assertGreater(len(submitted), 0)
        self.assertLess(len(submitted), len(rows))
        self.assertEqual(len(result), len(submitted))
        self.assertTrue(any(item['reason'] == 'prompt_budget' for item in plan['_cascade']['omitted']))

    async def test_listwise_provider_options_do_not_change_add_defaults(self):
        response = {'choices': [{'message': {'tool_calls': [
            {'function': {'arguments': '{"ranking":[0],"irrelevant":[],"groups":[]}'}}]}}]}
        call = AsyncMock(return_value=response)
        attempts = []
        async def measured(**kwargs):
            attempts.append(kwargs['attempts'])
            return await kwargs['call'](1)
        with patch.object(config, 'FAKE', False), \
                patch.object(metrics, 'measured_call', side_effect=measured), \
                patch.dict(sys.modules, {'litellm': SimpleNamespace(acompletion=call)}):
            await llm.complete_json('rank', schema=cascade.schema(1),
                                    max_tokens=256, timeout=20, attempts=1)
            self.assertEqual(call.call_args.kwargs['max_tokens'], min(256, config.LLM_JSON_MAX_TOKENS))
            self.assertEqual(call.call_args.kwargs['timeout'], 20)
            await llm.complete_json('extract', schema=llm.STRUCTURED_SCHEMAS['extraction'])
            self.assertEqual(call.call_args.kwargs['max_tokens'], config.LLM_JSON_MAX_TOKENS)
            self.assertEqual(call.call_args.kwargs['timeout'], 180)
        self.assertEqual(attempts, [1, 2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
