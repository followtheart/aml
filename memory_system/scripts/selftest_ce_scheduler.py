"""CE wave scheduling through real HTTP/accounting code, without any providers."""
import asyncio
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

import httpx

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_read_text = Path.read_text
with patch.object(Path, 'read_text', lambda path, *a, **kw:
                  '' if path.name == '.env' else _read_text(path, *a, **kw)):
    from app import budget, cascade_rerank as cascade, config, cross_encoder, llm, schemas

_async_client = httpx.AsyncClient


class CESchedulerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        settings = patch.multiple(config, FAKE=False,
            CE_API_URL='https://ce-scheduler.example.test/v1/rerank',
            CE_API_KEY='scheduler-test-key', CE_MODEL='scheduler-test-model',
            CE_API_FORMAT='cohere', CE_TIMEOUT_SECONDS=2., CE_DEADLINE_SECONDS=2.,
            CE_BATCH_SIZE=24, CE_MAX_DOCUMENT_BYTES=10000, CE_MAX_REQUEST_BYTES=90000,
            CASCADE_COARSE_LIMIT=50, CASCADE_FINE_LIMIT=12, CASCADE_LLM_LIMIT=1,
            RERANK_CONCURRENCY=2, RERANK_MAX_PROMPT_BYTES=24000,
            RERANK_DEADLINE_SECONDS=2., RERANK_REPAIR_MAX_CALLS=0,
            PROVIDER_CONCURRENCY=4, PROVIDER_RPM=0, PROVIDER_SOFT_TPM=0,
            RATE_LIMIT_RETRIES=0)
        settings.start()
        self.addCleanup(settings.stop)
        self.submitted = []
        self.listwise_calls = 0
        self.listwise_prompts = []

    def rows(self, count=45, length=1999):
        return [dict(id=f'm{i:02}', content=f'document-{i:02} '.ljust(length, 'x'),
                     _fused=1-i/100) for i in range(count)]

    def req(self):
        return schemas.SearchRequest(user_id='u', query='q', top_k=50)

    def response(self, documents, tokens):
        # Deliberately return reversed indices to exercise real score mapping.
        self.assertLessEqual(tokens, cross_encoder.request_size('q', documents))
        return httpx.Response(200, json={'results': [
            {'index': i, 'relevance_score': float(re.search(r'document-(\d+)', documents[i])[1])}
            for i in reversed(range(len(documents)))], 'usage': {'total_tokens': tokens}})

    async def listwise(self, prompt, *args, **kwargs):
        limits = budget.current.get()
        if limits:
            limits.before_call()
        self.listwise_calls += 1
        self.listwise_prompts.append(prompt)
        count = len(re.findall(r'^\d+:', prompt, re.M))
        return dict(ranking=list(range(count)), irrelevant=[], groups=[])

    @contextmanager
    def transport(self, handler):
        async def checked(request):
            self.assertEqual(str(request.url), config.CE_API_URL)
            self.assertEqual(request.headers['authorization'], 'Bearer scheduler-test-key')
            body = json.loads(request.content)
            self.assertEqual(body['query'], 'q')
            documents = body['documents']
            self.assertLessEqual(len(documents), config.CE_BATCH_SIZE)
            self.assertLessEqual(cross_encoder.request_size('q', documents), config.CE_MAX_REQUEST_BYTES)
            limits = budget.current.get()
            self.assertGreater(limits.reserved_tokens, 0)
            self.assertLessEqual(limits.tokens + limits.reserved_tokens, limits.max_tokens)
            self.submitted.append(documents)
            return await handler(documents, len(self.submitted))

        transport = httpx.MockTransport(checked)
        def client(*args, **kwargs):
            kwargs['transport'] = transport
            return _async_client(*args, **kwargs)
        with patch.object(httpx, 'AsyncClient', side_effect=client), \
                patch.object(llm, 'complete_json', side_effect=self.listwise):
            yield

    def assert_all_scored_once(self, originals, result, limits):
        sent = [text for batch in self.submitted for text in batch]
        identifier = lambda text: re.match(r'document-\d+', text)[0]
        self.assertEqual(Counter(map(identifier, sent)),
                         Counter(identifier(row['content']) for row in originals))
        expected = {identifier(row['content']): row['content'] for row in originals}
        for text in sent:
            self.assertEqual(text, expected[identifier(text)], 'source text was clipped or modified')
        self.assertEqual({row['id']: row.get('_ce_score') for row in result},
                         {row['id']: float(row['id'][1:]) for row in originals})
        self.assertLessEqual(limits.tokens, limits.max_tokens)
        self.assertLessEqual(limits.calls, limits.max_calls)
        self.assertEqual(limits.reserved_tokens, 0)
        self.assertEqual(limits.reserved_calls, 0)

    async def test_settled_wave_allows_every_candidate_to_be_scored_once(self):
        first_started, second_started = asyncio.Event(), asyncio.Event()
        release_first, release_second, first_returned = (asyncio.Event() for _ in range(3))
        async def handler(documents, index):
            if index == 1:
                first_started.set()
                await release_first.wait()
                first_returned.set()
            elif index == 2:
                second_started.set()
                await release_second.wait()
            return self.response(documents, min(8000, cross_encoder.request_size('q', documents)))

        rows, plan = self.rows(), {}
        with budget.scope(seconds=15, calls=12, tokens=64000) as limits, self.transport(handler):
            limits.tokens = 2000
            task = asyncio.create_task(cascade.rank(self.req(), plan, rows))
            try:
                await asyncio.wait_for(first_started.wait(), 1)
                await asyncio.wait_for(second_started.wait(), 1)
                release_first.set()
                await asyncio.wait_for(first_returned.wait(), 1)
                # Let accounting settle while the second HTTP response is held.
                # Its release depends only on this test, never on a third call.
                await asyncio.sleep(.02)
                self.assertEqual(len(self.submitted), 2)
                release_second.set()
                result = await asyncio.wait_for(task, 2)
            finally:
                release_first.set()
                release_second.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assert_all_scored_once(rows, result, limits)
        self.assertEqual(plan['_cascade']['cross_encoder']['status'], 'ok')
        self.assertFalse(plan['_cascade']['cross_encoder']['errors'])
        self.assertEqual(limits.tokens, 2000 + sum(
            min(8000, cross_encoder.request_size('q', batch)) for batch in self.submitted))

    async def test_later_wave_shrinks_batches_after_usage_without_clipping_documents(self):
        both_started = asyncio.Event()
        async def handler(documents, index):
            if index <= 2:
                if index == 2:
                    both_started.set()
                await both_started.wait()
            return self.response(documents, 18000 if index <= 2 else len(documents) * 200)

        rows, plan = self.rows(), {}
        with budget.scope(seconds=15, calls=16, tokens=64000) as limits, self.transport(handler):
            limits.tokens = 2000
            result = await asyncio.wait_for(cascade.rank(self.req(), plan, rows), 2)
        self.assert_all_scored_once(rows, result, limits)
        self.assertGreater(len(self.submitted), 2)
        self.assertLess(max(map(len, self.submitted[2:])), min(map(len, self.submitted[:2])))
        self.assertEqual(plan['_cascade']['cross_encoder']['status'], 'ok')

    async def test_one_large_document_can_use_available_budget_without_two_slot_split(self):
        async def handler(documents, index):
            return self.response(documents, 500)
        rows, plan = self.rows(1, 7000), {}
        with budget.scope(seconds=15, calls=4, tokens=20000) as limits, self.transport(handler), \
                patch.object(config, 'RERANK_CONCURRENCY', 4):
            result = await cascade.rank(self.req(), plan, rows)
        self.assert_all_scored_once(rows, result, limits)

    async def test_genuinely_insufficient_tokens_skip_http_with_explicit_diagnostics(self):
        async def handler(documents, index):
            self.fail('insufficient input budget dispatched HTTP')
        rows, plan = self.rows(2), {}
        with budget.scope(seconds=15, calls=4, tokens=1000) as limits, self.transport(handler):
            result = await cascade.rank(self.req(), plan, rows)
        self.assertEqual(self.submitted, [])
        self.assertTrue(all(row.get('_ce_score') is None for row in result))
        self.assertEqual(plan['_cascade']['cross_encoder']['status'], 'fallback')
        self.assertTrue(plan['_cascade']['cross_encoder']['errors'])
        self.assertEqual(limits.calls, 0)
        self.assertEqual(limits.reserved_tokens, 0)

    async def test_call_cap_bounds_real_http_and_leaves_unscored_candidates_identifiable(self):
        async def handler(documents, index):
            return self.response(documents, 100)
        rows, plan = self.rows(12, 199), {}
        with budget.scope(seconds=15, calls=2, tokens=64000) as limits, self.transport(handler), \
                patch.object(config, 'CE_BATCH_SIZE', 4):
            result = await cascade.rank(self.req(), plan, rows)
        self.assertGreater(len(self.submitted), 0)
        self.assertLessEqual(len(self.submitted) + self.listwise_calls, 2)
        self.assertEqual(limits.calls, len(self.submitted) + self.listwise_calls)
        sent = [text for batch in self.submitted for text in batch]
        self.assertEqual(len(sent), len(set(sent)))
        self.assertLess(len(sent), len(rows))
        self.assertTrue(any(row.get('_ce_score') is None for row in result))
        self.assertTrue(plan['_cascade']['cross_encoder']['errors'])
        self.assertEqual(limits.reserved_tokens, 0)

    async def test_ce_rate_limit_retry_cannot_spend_reserved_listwise_call(self):
        async def handler(documents, index):
            if index == 1:
                return httpx.Response(429, json={'error': 'controlled rate limit'})
            return self.response(documents, 100)
        rows, plan = self.rows(3, 199), {}
        with budget.scope(seconds=15, calls=2, tokens=64000) as limits, self.transport(handler), \
                patch.multiple(config, RATE_LIMIT_RETRIES=1,
                               RATE_LIMIT_BACKOFF_SECONDS=0, RATE_LIMIT_MAX_BACKOFF_SECONDS=0):
            result = await cascade.rank(self.req(), plan, rows)
        self.assertEqual(len(self.submitted), 1,
                         'a CE retry spent the call reserved for the final listwise stage')
        self.assertEqual(self.listwise_calls, 1)
        self.assertEqual(limits.calls, 2)
        self.assertEqual(plan['_cascade']['listwise']['status'], 'ok')
        self.assertTrue(any(error['error'] == 'BudgetExceeded'
                            for error in plan['_cascade']['cross_encoder']['errors']))
        self.assertTrue(any(row.get('_cascade_selected') for row in result))
        self.assertEqual(limits.reserved_tokens, 0)
        self.assertEqual(limits.reserved_calls, 0)

    async def test_listwise_token_budget_reduces_whole_pool_without_clipping_or_losing_ce_scores(self):
        async def handler(documents, index):
            return self.response(documents, len(documents) * 1750)
        rows, plan = self.rows(8), {}
        originals = {row['id']: row['content'] for row in rows}
        with budget.scope(seconds=15, calls=12, tokens=26000) as limits, self.transport(handler), \
                patch.object(config, 'CASCADE_LLM_LIMIT', 10):
            result = await cascade.rank(self.req(), plan, rows)
        self.assert_all_scored_once(rows, result, limits)
        self.assertEqual(self.listwise_calls, 1)
        listed = re.findall(r'^\d+: \[id: (m\d+)\] (.*)$', self.listwise_prompts[0], re.M)
        self.assertGreater(len(listed), 0)
        self.assertLess(len(listed), len(rows))
        for mid, text in listed:
            self.assertEqual(text, originals[mid], 'listwise shortened a source to fit the budget')
        self.assertEqual({mid for mid, _ in listed},
                         {row['id'] for row in result if row.get('_cascade_selected')})
        self.assertTrue(all(row['_final'] == float(row['id'][1:]) for row in result))
        self.assertEqual(plan['_cascade']['listwise']['status'], 'ok')
        self.assertTrue(any(item['stage'] == 'listwise' and item['reason'] == 'token_budget'
                            for item in plan['_cascade']['omitted']))
        self.assertEqual(limits.tokens, 14000)

    async def test_stage_timeout_keeps_completed_scores_and_releases_pending_reservations(self):
        active, cancelled = set(), set()
        async def handler(documents, index):
            if index == 1:
                return self.response(documents, 100)
            active.add(index)
            try:
                await asyncio.Event().wait()
            finally:
                active.remove(index)
                cancelled.add(index)
        rows, plan = self.rows(12, 199), {}
        with budget.scope(seconds=15, calls=8, tokens=64000) as limits, self.transport(handler), \
                patch.multiple(config, CE_BATCH_SIZE=4, CE_DEADLINE_SECONDS=.04):
            result = await asyncio.wait_for(cascade.rank(self.req(), plan, rows), 1)
        self.assertTrue(cancelled)
        self.assertFalse(active)
        self.assertEqual(sum(row.get('_ce_score') is not None for row in result), 4)
        self.assertEqual(plan['_cascade']['cross_encoder']['status'], 'partial')
        self.assertTrue(all(b['status'] != 'running'
                            for b in plan['_cascade']['cross_encoder']['batches']))
        self.assertEqual(limits.reserved_tokens, 0)
        self.assertEqual(limits.reserved_calls, 0)

    async def test_outer_cancellation_cleans_http_and_all_token_reservations(self):
        both_started = asyncio.Event()
        active, cancelled = set(), set()
        async def handler(documents, index):
            active.add(index)
            if index == 2:
                both_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                active.remove(index)
                cancelled.add(index)
        with budget.scope(seconds=15, calls=8, tokens=64000) as limits, self.transport(handler):
            task = asyncio.create_task(cascade.rank(self.req(), {}, self.rows()))
            try:
                await asyncio.wait_for(both_started.wait(), 1)
                self.assertGreater(limits.reserved_tokens, 0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertFalse(active)
        self.assertEqual(cancelled, {1, 2})
        self.assertEqual(limits.reserved_tokens, 0)
        self.assertEqual(limits.reserved_calls, 0)
        self.assertEqual(self.listwise_calls, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
