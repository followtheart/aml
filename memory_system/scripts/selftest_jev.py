"""Offline OpenRouter Decisions contract, parsing and budget regression tests."""
import asyncio
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_read_text = Path.read_text
with patch.object(Path, 'read_text', lambda path, *a, **kw:
                  '' if path.name == '.env' else _read_text(path, *a, **kw)):
    from app import budget, config, metrics

_async_client = httpx.AsyncClient


class JevTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.jev = importlib.import_module('app.jev')
        settings = patch.multiple(config, create=True, FAKE=False,
            JEV_API_KEY='unit-test-jev-key', JEV_MODEL='typesafe/jev-1.13',
            JEV_TIMEOUT_SECONDS=2.0, JEV_MAX_REQUEST_BYTES=48000,
            RATE_LIMIT_RETRIES=0, PROVIDER_RPM=0, PROVIDER_SOFT_TPM=0)
        settings.start()
        self.addCleanup(settings.stop)
        self.state = {'claim': 'The user likes cycling.', 'evidence': 'I love cycling.'}
        self.questions = {'claim_0': {'type': 'choice', 'instructions': 'Evaluate support.',
            'criteria': {'supported': 'Supports all details.', 'contradicted': 'Denies it.',
                         'insufficient': 'Does not establish it.'}}}
        self.response = {'id': 'decision-unit', 'model': 'typesafe/jev-1.13.0',
            'answers': {'claim_0': {'type': 'choice', 'choice': 'supported',
                'confidence': .8, 'probabilities': {'supported': .9,
                    'contradicted': .05, 'insufficient': .05}}},
            'usage': {'input_tokens': 11, 'output_tokens': 7, 'cost': .000001}}

    async def call(self, handler, *, state=None, questions=None):
        transport = httpx.MockTransport(handler)

        def client(*args, **kwargs):
            kwargs['transport'] = transport
            return _async_client(*args, **kwargs)

        with patch.object(httpx, 'AsyncClient', side_effect=client):
            return await self.jev.decide(self.state if state is None else state,
                self.questions if questions is None else questions)

    def success(self, request):
        return httpx.Response(200, content=json.dumps(self.response))

    async def test_request_contract_and_normalized_accounting(self):
        seen = []

        def handler(request):
            seen.append(request)
            self.assertEqual(str(request.url), 'https://openrouter.ai/api/alpha/decisions')
            self.assertEqual(request.method, 'POST')
            self.assertEqual(request.headers['authorization'], 'Bearer unit-test-jev-key')
            self.assertEqual(json.loads(request.content), {'model': config.JEV_MODEL,
                'state': self.state, 'questions': self.questions})
            return self.success(request)

        with budget.scope(seconds=5, calls=2, tokens=20000) as limits:
            result = await self.call(handler)
            self.assertEqual(limits.calls, 1)
            self.assertEqual(limits.tokens, 18)
            self.assertEqual(limits.reserved_tokens, 0)
        self.assertEqual(result['answers'], self.response['answers'])
        self.assertEqual(result['requested_model'], config.JEV_MODEL)
        self.assertEqual(result['resolved_model'], self.response['model'])
        self.assertEqual(result['model'], self.response['model'])
        self.assertEqual(result['id'], 'decision-unit')
        self.assertEqual(result['usage'], {'input_tokens': 11, 'output_tokens': 7,
            'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18,
            'cost': .000001})
        self.assertEqual(len(seen), 1)

    async def test_strict_choice_response_rejects_invalid_values_and_keeps_billed_usage(self):
        mutations = [
            lambda p: p.update(model=''),
            lambda p: p.update(id=False),
            lambda p: p.update(answers={}),
            lambda p: p['answers'].update(other=p['answers']['claim_0']),
            lambda p: p['answers']['claim_0'].update(type='score'),
            lambda p: p['answers']['claim_0'].update(choice='other'),
            lambda p: p['answers']['claim_0'].update(choice='contradicted'),
            lambda p: p['answers']['claim_0'].update(confidence=True),
            lambda p: p['answers']['claim_0'].update(confidence=float('nan')),
            lambda p: p['answers']['claim_0'].update(confidence=1.1),
            lambda p: p['answers']['claim_0']['probabilities'].pop('insufficient'),
            lambda p: p['answers']['claim_0']['probabilities'].update(extra=0),
            lambda p: p['answers']['claim_0']['probabilities'].update(supported=True),
            lambda p: p['answers']['claim_0']['probabilities'].update(supported=float('inf')),
            lambda p: p['answers']['claim_0']['probabilities'].update(supported=-.1),
            lambda p: p['answers']['claim_0']['probabilities'].update(supported=.6),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                payload = copy.deepcopy(self.response)
                mutate(payload)

                def handler(request):
                    return httpx.Response(200, content=json.dumps(payload))

                with budget.scope(seconds=5, calls=3, tokens=30000) as limits:
                    with self.assertRaises(ValueError):
                        await self.call(handler)
                    self.assertEqual(limits.tokens, 18 * limits.calls)
                    self.assertGreater(limits.calls, 0)
                    self.assertEqual(limits.reserved_tokens, 0)

    async def test_two_decimal_probability_rounding_and_ties_are_valid(self):
        self.response['answers']['claim_0'].update(choice='insufficient',
            probabilities={'supported': .33, 'contradicted': .33, 'insufficient': .33})
        self.assertEqual((await self.call(self.success))['answers']['claim_0']['choice'],
                         'insufficient')

    async def test_invalid_request_and_oversize_inputs_never_call_provider(self):
        invalid = [({}, self.questions), ('', self.questions), (self.state, {}),
            (self.state, {'': self.questions['claim_0']}),
            (self.state, {'claim': {'type': 'choice', 'instructions': '', 'criteria': {'a': None}}}),
            ({'number': float('nan')}, self.questions),
            (self.state, {'claim': {'type': 'choice', 'instructions': 'x', 'criteria': {'': 'x', 'b': 'y'}}})]
        for state, questions in invalid:
            with self.subTest(state=state, questions=questions):
                with self.assertRaises(ValueError):
                    await self.call(lambda r: self.fail('invalid input reached HTTP'),
                                    state=state, questions=questions)
        with patch.object(config, 'JEV_MAX_REQUEST_BYTES', 20):
            with self.assertRaises(ValueError):
                await self.call(lambda r: self.fail('oversize input reached HTTP'))

    async def test_fake_and_missing_key_never_borrow_chat_credentials_or_call_network(self):
        for fake, key in ((True, 'unit-key'), (False, None), (False, '')):
            with self.subTest(fake=fake, key=bool(key)), patch.object(config, 'FAKE', fake), \
                    patch.object(config, 'JEV_API_KEY', key), \
                    patch.object(config, 'LLM_API_KEY', 'must-not-borrow'):
                with self.assertRaises(self.jev.JevUnavailable):
                    await self.call(lambda r: self.fail('unconfigured/offline request reached HTTP'))

    async def test_redirects_never_follow_and_logs_do_not_include_key_state_or_error_body(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(302, headers={'location': 'https://other.example.test/'},
                                  text='private-error-body')

        with self.assertLogs('aml.metrics') as logs:
            with self.assertRaises(httpx.HTTPStatusError):
                await self.call(handler)
        self.assertTrue(seen)
        self.assertTrue(all(str(r.url) == self.jev.ENDPOINT for r in seen))
        output = '\n'.join(logs.output)
        for secret in ('unit-test-jev-key', 'private-error-body', 'I love cycling.'):
            self.assertNotIn(secret, output)

    async def test_missing_or_invalid_usage_uses_conservative_estimate(self):
        for usage in (None, {'input_tokens': True, 'output_tokens': -1, 'cost': float('nan')},
                      {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 1}):
            with self.subTest(usage=usage):
                self.response['usage'] = usage
                result = await self.call(self.success)
                actual = result['usage']
                self.assertEqual(actual['total_tokens'], actual['input_tokens'] + actual['output_tokens'])
                self.assertGreater(actual['total_tokens'], 0)
                self.assertNotIn('cost', actual)
                if usage and 'prompt_tokens' in usage:
                    self.assertEqual(actual['total_tokens'], 5)

    async def test_duplicate_json_answer_ids_are_rejected(self):
        answer = json.dumps(self.response['answers']['claim_0'])
        payload = '{"answers":{"claim_0":' + answer + ',"claim_0":' + answer + '}}'
        with self.assertRaises(ValueError):
            await self.call(lambda r: httpx.Response(200, content=payload))

    async def test_request_budget_rejects_before_network(self):
        with budget.scope(seconds=5, calls=2, tokens=1) as limits:
            with self.assertRaises(budget.BudgetExceeded):
                await self.call(lambda r: self.fail('token budget failure reached HTTP'))
            self.assertEqual(limits.reserved_tokens, 0)

    async def test_concurrent_dispatch_reserves_shared_input_output_budget(self):
        first_started, release = asyncio.Event(), asyncio.Event()
        seen = []
        _, input_bound = self.jev._request(self.state, self.questions)
        reservation = input_bound + self.jev._response_allowance(self.questions)

        async def handler(request):
            seen.append(request)
            first_started.set()
            await release.wait()
            return self.success(request)

        transport = httpx.MockTransport(handler)

        def client(*args, **kwargs):
            kwargs['transport'] = transport
            return _async_client(*args, **kwargs)

        with budget.scope(seconds=5, calls=3, tokens=reservation) as limits, \
                patch.object(httpx, 'AsyncClient', side_effect=client):
            first = asyncio.create_task(self.jev.decide(self.state, self.questions))
            try:
                await asyncio.wait_for(first_started.wait(), 1)
                self.assertEqual(limits.reserved_tokens, reservation)
                with self.assertRaises(budget.BudgetExceeded):
                    await self.jev.decide(self.state, self.questions)
                self.assertEqual(len(seen), 1)
                release.set()
                await first
                self.assertEqual(limits.tokens, 18)
                self.assertEqual(limits.reserved_tokens, 0)
            finally:
                release.set()
                await first

    async def test_malformed_json_charges_estimate_and_release_allows_next_call(self):
        with budget.scope(seconds=5, calls=3, tokens=30000) as limits:
            with self.assertRaises(metrics.ResponseParseError):
                await self.call(lambda r: httpx.Response(200, text='private-invalid-json'))
            self.assertGreater(limits.tokens, 0)
            self.assertEqual(limits.reserved_tokens, 0)
            before = limits.tokens
            await self.call(self.success)
            self.assertEqual(limits.tokens, before + 18)

    async def test_cancellation_releases_reservation(self):
        started = asyncio.Event()

        async def blocked(request):
            started.set()
            await asyncio.Event().wait()

        with budget.scope(seconds=5, calls=3, tokens=30000) as limits:
            task = asyncio.create_task(self.call(blocked))
            await asyncio.wait_for(started.wait(), 1)
            self.assertGreater(limits.reserved_tokens, 0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(limits.reserved_tokens, 0)
            await self.call(self.success)
            self.assertEqual(limits.tokens, 18)

    async def test_stage_deadline_includes_throttle_and_current_budget(self):
        for local_timeout, budget_seconds in ((.03, 3), (3, .03)):
            with self.subTest(local_timeout=local_timeout), \
                    patch.object(config, 'JEV_TIMEOUT_SECONDS', local_timeout):
                async def blocked_pacer(*args):
                    await asyncio.Event().wait()

                with budget.scope(seconds=budget_seconds, calls=3, tokens=30000) as limits, \
                        patch.object(metrics._Pacer, 'wait', side_effect=blocked_pacer):
                    with self.assertRaises(TimeoutError):
                        await self.call(lambda r: self.fail('blocked pacer reached HTTP'))
                    self.assertEqual(limits.reserved_tokens, 0)
                    self.assertEqual(limits.calls, 0)

    async def test_http_timeout_and_rate_limit_retry_obey_shared_budget(self):
        attempts = []

        def handler(request):
            attempts.append(request)
            if len(attempts) == 1:
                return httpx.Response(429, headers={'retry-after': '0'})
            return self.success(request)

        with patch.object(config, 'RATE_LIMIT_RETRIES', 1), \
                patch.object(config, 'RATE_LIMIT_BACKOFF_SECONDS', 0), \
                budget.scope(seconds=5, calls=2, tokens=30000) as limits:
            await self.call(handler)
            self.assertEqual(limits.calls, 2)
            self.assertEqual(limits.reserved_tokens, 0)
            self.assertEqual(limits.tokens, 18)

        async def blocked(request):
            await asyncio.Event().wait()

        with patch.object(config, 'JEV_TIMEOUT_SECONDS', .03), \
                budget.scope(seconds=5, calls=2, tokens=30000) as limits:
            with self.assertRaises(TimeoutError):
                await self.call(blocked)
            self.assertEqual(limits.reserved_tokens, 0)


if __name__ == '__main__':
    unittest.main()
