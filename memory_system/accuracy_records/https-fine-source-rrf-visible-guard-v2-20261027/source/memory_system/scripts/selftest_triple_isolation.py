"""Independent atomic-fact checks cannot inherit failed graph projections."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import add_pipeline as add, config, llm, memory_debug, schemas


class TripleIsolationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')).start()
        patch.multiple(config, FAKE=False, EXTRACT_TRIPLE_ISOLATION=True,
                       EXTRACT_SPAN_REFS=False, STORE_EPISODES=True).start()
        self.req = schemas.AddRequest(request_id='isolation', user_id='u', session_id='s',
            messages=[schemas.Message(role='user', content='I asked about tidal patterns yesterday.',
                                      timestamp=1700000000000)])
        self.raw = dict(content='The user asked about tidal patterns yesterday.', type='fact',
                        evidence=[dict(message_index=0, quote=self.req.messages[0].content)],
                        triples=[dict(subject='user', relation='owns', object='a boat')],
                        state=None, sensitivity='normal')
        self.events = []
        patch.object(memory_debug, 'extraction_event',
                     lambda stage, **fields: self.events.append(
                         copy.deepcopy(dict(stage=stage, **fields)))).start()

    async def extract(self, responses):
        provider = AsyncMock(side_effect=responses)
        with patch.object(llm, 'complete_json', provider):
            result = await add._extract_segment(self.req, None, 0, [0])
        return result, provider

    async def test_preserve_fact_only_after_independent_check(self):
        before = copy.deepcopy(self.raw)
        result, provider = await self.extract([
            dict(facts=[self.raw]), dict(valid=False, reason='Ownership triple is unsupported.'),
            dict(valid=True, reason='The inquiry and date are explicit.')])
        self.assertEqual(self.raw, before)
        self.assertEqual(len(result), 1)
        fact = result[0]
        self.assertEqual(fact['content'], self.raw['content'])
        self.assertEqual(fact['triples'], [])
        self.assertEqual(fact['_sources'], [0])
        self.assertEqual(fact['evidence'][0]['quote'], self.req.messages[0].content)
        self.assertEqual(fact['time_expression'], 'yesterday')
        self.assertEqual(fact['sensitivity'], 'normal')
        self.assertEqual([call.kwargs['stage'] for call in provider.await_args_list],
                         ['add.extract.segment_1', 'add.verify_evidence', 'add.verify_fact_without_triples'])
        omission = next(e for e in self.events if e['stage'] == 'semantic_triples_omitted')
        self.assertEqual(omission['omitted_triples'], self.raw['triples'])
        self.assertIn('Ownership', omission['reason'])
        self.assertIn('"triples": []', provider.await_args_list[-1].args[0])

    async def test_failed_second_check_preserves_only_raw_episode(self):
        result, provider = await self.extract([
            dict(facts=[self.raw]), dict(valid=False, reason='Unsupported claim.'),
            dict(valid=False, reason='The claim itself is also unsupported.')])
        self.assertEqual([f['type'] for f in result], ['episode'])
        self.assertIn(self.req.messages[0].content, result[0]['content'])
        self.assertEqual(provider.await_count, 3)
        self.assertFalse(any(e['stage'] == 'semantic_triples_omitted' for e in self.events))

    async def test_without_triples_or_when_disabled_no_extra_check(self):
        for settings, triples in ((dict(EXTRACT_TRIPLE_ISOLATION=False), self.raw['triples']),
                                  (dict(EXTRACT_TRIPLE_ISOLATION=True), [])):
            with self.subTest(settings=settings):
                raw = dict(self.raw, triples=triples)
                with patch.multiple(config, **settings):
                    result, provider = await self.extract([
                        dict(facts=[raw]), dict(valid=False, reason='Unsupported.')])
                self.assertEqual([f['type'] for f in result], ['episode'])
                self.assertEqual(provider.await_count, 2)

    async def test_accepted_fact_keeps_graph_and_cost(self):
        result, provider = await self.extract([
            dict(facts=[self.raw]), dict(valid=True, reason='Synthetic all-fields approval.')])
        self.assertEqual(result[0]['triples'], self.raw['triples'])
        self.assertEqual(provider.await_count, 2)

    async def test_invalid_original_quote_cannot_use_isolation(self):
        raw = dict(self.raw, evidence=[dict(message_index=0, quote='I own a boat.')])
        result, provider = await self.extract([dict(facts=[raw])])
        self.assertEqual([f['type'] for f in result], ['episode'])
        self.assertEqual(provider.await_count, 1)

    async def test_roles_state_and_privacy_remain_in_independent_request(self):
        self.req.messages[0].role = 'assistant'
        self.req.messages[0].sensitivity = 'sensitive'
        self.raw['sensitivity'] = 'sensitive'
        self.raw['state'] = dict(subject='user', attribute='home', value='Harbor City')
        fact = add._validate_fact(self.raw, self.req.messages, 0, self.req)
        original = copy.deepcopy(fact)
        with patch.object(add, '_verify_semantics', AsyncMock()) as verify:
            got = await add._isolate_triples(
                fact, [dict(message_index=0, role='assistant', content=self.req.messages[0].content)],
                ValueError('Invalid graph.'))
        self.assertEqual(fact, original)
        self.assertEqual(got, dict(original, triples=[]))
        self.assertEqual(got['epistemic_status'], 'inferred')
        self.assertEqual(got['state'], original['state'])
        self.assertEqual(got['sensitivity'], 'sensitive')
        self.assertEqual(verify.call_args.args[1][0]['role'], 'assistant')

    async def test_batch_rejection_isolates_only_failed_fact(self):
        other = dict(self.raw, content='The user asked about tidal patterns.',
                     time_expression=None, triples=[])
        result, provider = await self.extract([
            dict(facts=[self.raw, other]),
            dict(valid=False, reason='One graph is unsupported.'),
            dict(valid=False, reason='Ownership triple is unsupported.'),
            dict(valid=True, reason='Unchanged inquiry is supported.'),
            dict(valid=True, reason='Second inquiry is supported.')])
        self.assertEqual(len(result), 2)
        self.assertTrue(all(f['type'] == 'fact' and f['triples'] == [] for f in result))
        self.assertEqual(provider.await_count, 5)

    async def test_provider_failure_does_not_accept_fact(self):
        result, _ = await self.extract([
            dict(facts=[self.raw]), dict(valid=False, reason='Invalid graph.'),
            llm.LLMError('Provider unavailable.')])
        self.assertEqual([f['type'] for f in result], ['episode'])
        fallback = next(e for e in self.events if e['stage'] == 'segment_fallback')
        self.assertEqual(fallback['error_type'], 'LLMError')


if __name__ == '__main__':
    unittest.main()
