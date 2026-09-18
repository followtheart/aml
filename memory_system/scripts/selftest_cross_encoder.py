"""Cross-Encoder HTTP contracts using real httpx clients and local transports."""
import asyncio
import importlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_read_text = Path.read_text
with patch.object(Path, 'read_text', lambda path, *a, **kw:
                  '' if path.name == '.env' else _read_text(path, *a, **kw)):
    from app import budget, cascade_rerank, config, llm, schemas

_async_client = httpx.AsyncClient


class CrossEncoderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ce = importlib.import_module('app.cross_encoder')
        self.settings = patch.multiple(config, create=True, FAKE=False,
            CE_API_URL='https://rerank.example.test/v1/rerank',
            CE_API_KEY='unit-test-cross-encoder-key', CE_MODEL='unit-rerank-model',
            CE_API_FORMAT='cohere', CE_TIMEOUT_SECONDS=7.0, RATE_LIMIT_RETRIES=0)
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.documents = ['Alice founded Acme.', 'Acme is based in Oslo.', 'Bob likes pears.']
        self.query = 'Where is the company Alice founded based?'

    async def call(self, handler):
        transport = httpx.MockTransport(handler)

        def client(*args, **kwargs):
            kwargs['transport'] = transport
            return _async_client(*args, **kwargs)

        with patch.object(httpx, 'AsyncClient', side_effect=client):
            return await self.ce.rerank(self.query, self.documents, stage='test.cross_encoder')

    async def test_cohere_request_and_out_of_order_results_align_to_input(self):
        seen = []

        def handler(request):
            seen.append(request)
            self.assertEqual(str(request.url), config.CE_API_URL)
            self.assertEqual(request.method, 'POST')
            self.assertEqual(request.headers['authorization'], 'Bearer unit-test-cross-encoder-key')
            body = json.loads(request.content)
            self.assertEqual(body['model'], 'unit-rerank-model')
            self.assertEqual(body['query'], self.query)
            self.assertEqual(body['documents'], self.documents)
            self.assertEqual(body.get('top_n', len(self.documents)), len(self.documents))
            return httpx.Response(200, json={'results': [
                {'index': 2, 'relevance_score': 0.1},
                {'index': 0, 'relevance_score': 0.7},
                {'index': 1, 'relevance_score': 0.9}]})

        scores = await self.call(handler)
        self.assertEqual(scores, [0.7, 0.9, 0.1])
        self.assertEqual(len(seen), 1)
        self.assertLessEqual(seen[0].extensions['timeout']['read'], 7.0)

    async def test_dashscope_request_and_nested_output_align_to_input(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body['model'], 'unit-rerank-model')
            self.assertEqual(body['input'], {'query': self.query, 'documents': self.documents})
            self.assertEqual(body.get('parameters', {}).get('top_n', len(self.documents)), len(self.documents))
            return httpx.Response(200, json={'output': {'results': [
                {'index': 1, 'relevance_score': 0.8},
                {'index': 2, 'relevance_score': 0.2},
                {'index': 0, 'relevance_score': 0.6}]}})

        with patch.object(config, 'CE_API_FORMAT', 'dashscope'):
            self.assertEqual(await self.call(handler), [0.6, 0.8, 0.2])

    async def test_complete_finite_logits_are_preserved_without_probability_clipping(self):
        def handler(request):
            return httpx.Response(200, json={'results': [
                {'index': 0, 'relevance_score': -3.5},
                {'index': 1, 'relevance_score': 2.25},
                {'index': 2, 'relevance_score': 0.0}]})

        self.assertEqual(await self.call(handler), [-3.5, 2.25, 0.0])

    async def test_invalid_indices_and_non_finite_scores_are_rejected(self):
        valid = [dict(index=i, relevance_score=0.5) for i in range(3)]
        cases = {
            'duplicate': [valid[0], valid[0], valid[2]],
            'missing': valid[:2],
            'out_of_range': [valid[0], valid[1], dict(index=3, relevance_score=0.5)],
            'negative': [valid[0], valid[1], dict(index=-1, relevance_score=0.5)],
            'boolean_index': [valid[0], dict(index=True, relevance_score=0.5), valid[2]],
            'fractional_index': [valid[0], dict(index=1.5, relevance_score=0.5), valid[2]],
            'nan': [valid[0], valid[1], dict(index=2, relevance_score=float('nan'))],
            'infinity': [valid[0], valid[1], dict(index=2, relevance_score=float('inf'))],
            'missing_score': [valid[0], valid[1], dict(index=2)],
        }
        for name, results in cases.items():
            with self.subTest(name=name):
                # Raw content also permits deliberately invalid NaN/Infinity JSON
                # numbers, which a decoder can accept but the scorer must reject.
                def handler(request, results=results):
                    return httpx.Response(200, content=json.dumps({'results': results}),
                                          headers={'content-type': 'application/json'})
                with self.assertRaises(Exception):
                    await self.call(handler)

    async def test_http_failures_are_not_replaced_by_fake_scores(self):
        for status in (401, 429, 500):
            with self.subTest(status=status):
                seen = []

                def handler(request):
                    seen.append(request)
                    return httpx.Response(status, json={'error': 'controlled test failure'})

                with self.assertRaises(Exception):
                    await self.call(handler)
                self.assertTrue(seen)

    async def test_missing_key_rejects_before_network_and_does_not_borrow_other_host_key(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json={'results': []})

        with patch.object(config, 'CE_API_KEY', None), patch.object(
                config, 'LLM_API_KEY', 'must-not-send-to-rerank-host'), patch.object(
                config, 'LLM_API_BASE', 'https://chat.example.test/v1'):
            with self.assertRaises(Exception):
                await self.call(handler)
        self.assertEqual(seen, [])

    async def test_transport_timeout_is_not_replaced_by_fake_scores(self):
        def handler(request):
            raise httpx.ReadTimeout('controlled timeout', request=request)

        with self.assertRaises(httpx.TimeoutException):
            await self.call(handler)

    async def test_concurrent_requests_reserve_shared_input_budget_before_http(self):
        first_started, second_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        seen = []
        documents = ['x' * 60]
        query = 'q'
        estimate = self.ce.request_size(query, documents)
        self.assertEqual(estimate, 61)

        async def handler(request):
            seen.append(request)
            (first_started if len(seen) == 1 else second_started).set()
            await release.wait()
            return httpx.Response(200, json={'results': [{'index': 0, 'relevance_score': 0.4}],
                                             'usage': {'total_tokens': estimate}})

        transport = httpx.MockTransport(handler)

        def client(*args, **kwargs):
            kwargs['transport'] = transport
            return _async_client(*args, **kwargs)

        with budget.scope(seconds=15, calls=4, tokens=100) as limits, patch.object(
                httpx, 'AsyncClient', side_effect=client):
            first = asyncio.create_task(self.ce.rerank(query, documents, stage='test.cross_encoder.first'))
            second = second_observed = None
            try:
                await asyncio.wait_for(first_started.wait(), 5)
                second = asyncio.create_task(self.ce.rerank(query, documents, stage='test.cross_encoder.second'))
                second_observed = asyncio.create_task(second_started.wait())
                # An implementation with reservations rejects the second call;
                # without reservations both HTTP requests reach the barrier.
                done, _ = await asyncio.wait({second, second_observed}, timeout=5,
                                             return_when=asyncio.FIRST_COMPLETED)
                self.assertTrue(done, 'second request neither failed nor reached HTTP')
                release.set()
                results = await asyncio.gather(first, second, return_exceptions=True)
                self.assertLessEqual(limits.tokens, limits.max_tokens,
                                     f'{len(seen)} concurrent requests exceeded the shared token budget')
                self.assertEqual(len(seen), 1)
                self.assertEqual(results[0], [0.4])
                self.assertIsInstance(results[1], budget.BudgetExceeded)
                self.assertEqual(limits.reserved_tokens, 0)
            finally:
                release.set()
                pending = [task for task in (first, second) if task is not None]
                await asyncio.gather(*pending, return_exceptions=True)
                if second_observed is not None:
                    second_observed.cancel()
                    await asyncio.gather(second_observed, return_exceptions=True)

    async def test_failed_and_invalid_responses_release_reservation_for_the_next_request(self):
        estimate = self.ce.request_size(self.query, self.documents)
        valid = [{'index': i, 'relevance_score': .4} for i in range(len(self.documents))]
        for failure in ('http', 'invalid_scores'):
            with self.subTest(failure=failure):
                attempts = []

                def handler(request):
                    attempts.append(request)
                    if len(attempts) == 1:
                        if failure == 'http':
                            return httpx.Response(500, json={'error': 'controlled failure'})
                        return httpx.Response(200, json={'results': [], 'usage': {'total_tokens': 7}})
                    return httpx.Response(200, json={'results': valid, 'usage': {'total_tokens': 9}})

                with budget.scope(seconds=15, calls=3, tokens=estimate + 7) as limits:
                    with self.assertRaises(Exception):
                        await self.call(handler)
                    self.assertEqual(limits.reserved_tokens, 0)
                    self.assertEqual(limits.tokens, 0 if failure == 'http' else 7)
                    self.assertEqual(await self.call(handler), [.4] * len(self.documents))
                    self.assertEqual(limits.reserved_tokens, 0)
                    self.assertEqual(limits.tokens, 9 if failure == 'http' else 16)
                self.assertEqual(len(attempts), 2)

    async def test_cancellation_releases_reservation_without_poisoning_shared_budget(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        estimate = self.ce.request_size(self.query, self.documents)

        async def blocked(request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        def complete(request):
            return httpx.Response(200, json={'results': [
                {'index': i, 'relevance_score': .6} for i in range(len(self.documents))],
                'usage': {'total_tokens': 9}})

        with budget.scope(seconds=15, calls=3, tokens=estimate) as limits:
            task = asyncio.create_task(self.call(blocked))
            try:
                await asyncio.wait_for(started.wait(), 5)
                self.assertEqual(limits.reserved_tokens, estimate)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(cancelled.is_set())
                self.assertEqual(limits.reserved_tokens, 0)
                self.assertEqual(await self.call(complete), [.6] * len(self.documents))
                self.assertEqual(limits.reserved_tokens, 0)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_listwise_format_repair_charges_usage_of_the_malformed_response(self):
        def response(arguments, tokens):
            return {'choices': [{'message': {'tool_calls': [
                {'function': {'arguments': arguments}}]}}],
                'usage': {'prompt_tokens': tokens - 10, 'completion_tokens': 10,
                          'total_tokens': tokens}}

        provider = AsyncMock(side_effect=[response('not valid JSON', 300),
            response('{"ranking":[0],"irrelevant":[],"groups":[]}', 400)])
        plan = {}
        with budget.scope(seconds=30, calls=4, tokens=64000) as limits, patch.object(
                self.ce, 'rerank', AsyncMock(return_value=[.5])), patch.object(
                config, 'RERANK_REPAIR_MAX_CALLS', 1), patch.dict(
                sys.modules, {'litellm': SimpleNamespace(acompletion=provider)}):
            rows = await cascade_rerank.rank(schemas.SearchRequest(user_id='u', query='Where?'),
                plan, [dict(id='a', content='Kyoto', _fused=1.)])
        self.assertEqual(provider.await_count, 2)
        self.assertEqual(plan['_cascade']['listwise']['status'], 'recovered')
        self.assertEqual([row['id'] for row in rows if row.get('_cascade_selected')], ['a'])
        self.assertEqual(limits.reserved_tokens, 0)
        self.assertEqual(limits.tokens, 700,
                         'malformed provider output still consumed tokens before the repair')

    async def test_redirect_does_not_forward_credential_to_another_origin(self):
        seen = []

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(302, headers={'location': 'https://different.example.test/rerank'})

        with self.assertRaises(Exception):
            await self.call(handler)
        self.assertEqual(seen, ['https://rerank.example.test/v1/rerank'])

    def test_credential_helper_reuses_only_matching_origin(self):
        with patch.object(config, 'LLM_API_BASE', 'https://api.siliconflow.cn/v1'), patch.object(
                config, 'LLM_API_KEY', 'same-origin-unit-key'), patch.object(
                config, 'EMBED_API_BASE', None), patch.object(config, 'EMBED_API_KEY', None):
            for url, expected in (
                    ('https://api.siliconflow.cn/v1/rerank', 'same-origin-unit-key'),
                    ('https://api.siliconflow.cn.different.example/rerank', ''),
                    ('http://api.siliconflow.cn/v1/rerank', ''),
                    ('https://api.siliconflow.cn:444/v1/rerank', '')):
                with self.subTest(url=url), patch.dict(os.environ, {
                        'AML_CE_API_URL': url, 'AML_CE_API_KEY': '',
                        'AML_CE_MODEL': 'unit-rerank', 'AML_CE_API_FORMAT': 'cohere'}):
                    actual_url, key, model, _ = config._cross_encoder_settings()
                    self.assertEqual(actual_url, url)
                    self.assertEqual(key, expected)
                    self.assertEqual(model, 'unit-rerank')

    def test_explicit_cross_encoder_credential_takes_precedence(self):
        with patch.object(config, 'LLM_API_BASE', 'https://api.siliconflow.cn/v1'), patch.object(
                config, 'LLM_API_KEY', 'must-not-override-explicit-key'), patch.dict(os.environ, {
                    'AML_CE_API_URL': 'https://api.siliconflow.cn/v1/rerank',
                    'AML_CE_API_KEY': 'explicit-unit-key', 'AML_CE_MODEL': 'unit-rerank'}):
            self.assertEqual(config._cross_encoder_settings()[1], 'explicit-unit-key')

    async def test_only_explicit_fake_mode_avoids_http(self):
        def handler(request):
            self.fail('explicit fake mode attempted an HTTP request')

        with patch.object(config, 'FAKE', True), patch.object(config, 'CE_API_KEY', None):
            scores = await self.call(handler)
        self.assertEqual(len(scores), len(self.documents))
        self.assertTrue(all(isinstance(value, (float, int)) and math.isfinite(value) for value in scores))


if __name__ == '__main__':
    unittest.main()
