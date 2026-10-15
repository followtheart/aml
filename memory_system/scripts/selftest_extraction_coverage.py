"""Coverage retries preserve source authority and bound supplementary work."""
import asyncio
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import add_pipeline as add, budget, config, extraction_coverage as coverage, llm, schemas


class CoverageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')).start()
        patch.multiple(config, FAKE=False, STORE_EPISODES=True,
                       EXTRACT_SPAN_REFS=False, EXTRACT_COVERAGE_RETRY=True).start()
        self.req = schemas.AddRequest(request_id='coverage', user_id='u', session_id='s',
            messages=[schemas.Message(role=role, content=text) for role, text in [
                ('assistant', 'Earlier context.'), ('user', 'I collect blue ceramics.'),
                ('assistant', 'That sounds interesting.'), ('user', 'I study tidal patterns.'),
                ('assistant', 'Here is a general explanation.')]])
        self.segments = [[1, 2, 3, 4]]
        self.groups = [[add._episode(self.req, self.segments[0]),
                        dict(type='fact', content='The user studies tides.', _sources=[3])]]

    async def test_only_uncovered_user_exchange(self):
        self.assertEqual(list(coverage.uncovered_exchanges(self.req, self.segments, self.groups)),
                         [(0, [1, 2])])
        self.assertEqual(list(coverage.uncovered_exchanges(self.req, [[0, 1, 2]], [[]])), [])
        self.assertEqual(list(coverage.uncovered_exchanges(self.req, [[1, 3]], [[]])), [])

    async def test_real_validation_maps_original_indices_and_preserves_episode(self):
        before = copy.deepcopy(self.groups[0])
        raw = dict(type='fact', content='The user collects blue ceramics.', triples=[],
                   evidence=[dict(message_index=0, quote=self.req.messages[1].content)])
        with patch.object(llm, 'complete_json', AsyncMock(side_effect=[
                dict(facts=[raw]), dict(valid=True)])) as provider:
            await coverage.supplement(self.req, None, self.segments, self.groups,
                                      add._extract_segment_impl)
        self.assertEqual(self.groups[0][:2], before)
        self.assertEqual(len(self.groups[0]), 3)
        self.assertEqual(self.groups[0][-1]['_sources'], [1])
        self.assertEqual(self.groups[0][-1]['evidence'][0]['message_index'], 1)
        self.assertEqual(self.groups[0][-1]['_segment'], 0)
        self.assertEqual(provider.await_count, 2)

    async def test_rejected_claim_cannot_be_added(self):
        before = copy.deepcopy(self.groups)
        raw = dict(type='fact', content='The user collects blue ceramics.', triples=[],
                   evidence=[dict(message_index=0, quote='I own a museum.')])
        with patch.object(llm, 'complete_json', AsyncMock(return_value=dict(facts=[raw]))):
            await coverage.supplement(self.req, None, self.segments, self.groups,
                                      add._extract_segment_impl)
        self.assertEqual(self.groups, before)

    async def test_semantic_rejection_cannot_be_added(self):
        before = copy.deepcopy(self.groups)
        raw = dict(type='fact', content='The user sells blue ceramics.', triples=[],
                   evidence=[dict(message_index=0, quote=self.req.messages[1].content)])
        with patch.object(llm, 'complete_json', AsyncMock(side_effect=[dict(facts=[raw]),
                dict(valid=False), dict(valid=False)])):
            await coverage.supplement(self.req, None, self.segments, self.groups,
                                      add._extract_segment_impl)
        self.assertEqual(self.groups, before)

    async def test_caps_and_parent_budget(self):
        extractor = AsyncMock(return_value=[])
        await coverage.supplement(self.req, None, self.segments, [[]], extractor, max_exchanges=1)
        self.assertEqual(extractor.await_count, 1)
        with budget.scope() as parent:
            await coverage.supplement(self.req, None, self.segments, [[]], extractor)
            self.assertIs(budget.current.get(), parent)
        self.assertEqual(extractor.await_count, 1)
        async def consume(*args):
            budget.current.get().before_call()
            return []
        capped = AsyncMock(side_effect=consume)
        await coverage.supplement(self.req, None, self.segments, [[]], capped, calls=1)
        self.assertEqual(capped.await_count, 1)

    async def test_timeout_preserves_original_and_cancellation_propagates(self):
        before = copy.deepcopy(self.groups)
        async def slow(*args):
            await asyncio.sleep(1)
        await coverage.supplement(self.req, None, self.segments, self.groups, slow, seconds=.001)
        self.assertEqual(self.groups, before)
        with self.assertRaises(asyncio.CancelledError):
            await coverage.supplement(self.req, None, self.segments, self.groups,
                AsyncMock(side_effect=asyncio.CancelledError))
        self.assertIsNone(budget.current.get())

    async def test_disabled_path_never_retries(self):
        with patch.object(config, 'EXTRACT_COVERAGE_RETRY', False), \
                patch.object(add, '_segments', AsyncMock(return_value=self.segments)), \
                patch.object(add, '_extract_segment', AsyncMock(return_value=self.groups[0])), \
                patch.object(coverage, 'supplement', AsyncMock()) as retry:
            result = await add._extract(Mock(get_summary=Mock(return_value=None)), self.req)
        retry.assert_not_awaited()
        self.assertEqual(result['facts'], self.groups[0])

    async def test_pipeline_uses_deterministic_coverage_before_retry(self):
        req = schemas.AddRequest(request_id='det', user_id='u', session_id='s', messages=[
            schemas.Message(role='user', content='[system] User persona: {"age": 38}'),
            schemas.Message(role='user', content='Please forget my preference for jazz.'),
            schemas.Message(role='assistant', content='Understood.')])
        segments = [[0, 1, 2]]
        facts = [add._episode(req, segments[0])]
        st = Mock(get_summary=Mock(return_value=None))
        with patch.object(config, 'PERSONA_SOURCE_EXTRACTION', True), \
                patch.object(add, '_extract', AsyncMock(return_value=dict(facts=facts, segments=segments))), \
                patch.object(add, '_extract_segment_impl', AsyncMock()) as retry, \
                patch.object(add, 'embed', AsyncMock(side_effect=RuntimeError('stop before persistence'))):
            with self.assertRaisesRegex(RuntimeError, 'stop before persistence'):
                await add._run_add(st, req)
        retry.assert_not_awaited()
        self.assertTrue(any(f.get('_persona') and f['_sources'] == [0] for f in facts))
        self.assertTrue(any(f.get('type') == 'rule' and f['_sources'] == [1] for f in facts))

    async def test_uncovered_sources_still_retry_and_fact_order_is_preserved(self):
        facts = [dict(type='rule', _sources=[3], _segment=0, _deterministic=True),
                 add._episode(self.req, self.segments[0])]
        before = copy.deepcopy(facts)
        extra = dict(type='fact', content='The user collects blue ceramics.', _sources=[1], _segment=0)
        retry = AsyncMock(return_value=[extra])
        await coverage.supplement_facts(self.req, None, self.segments, facts, retry)
        self.assertEqual(retry.await_args.args[-1], [1, 2])
        self.assertEqual(facts, before + [extra])

    async def test_unparsed_persona_is_not_assumed_covered(self):
        self.req.messages[1].content = '[system] User persona: {broken json}'
        retry = AsyncMock(return_value=[])
        facts = [dict(type='fact', _sources=[3], _segment=0)]
        await coverage.supplement_facts(self.req, None, self.segments, facts, retry)
        self.assertEqual(retry.await_args.args[-1], [1, 2])

    async def test_only_supplementary_user_anchored_facts_are_added(self):
        before = copy.deepcopy(self.groups)
        user = dict(type='fact', content='User collects ceramics.', _sources=[1])
        joint = dict(type='fact', content='User asked about ceramics.', _sources=[1, 2])
        assistant = dict(type='fact', content='Assistant suggested ceramic shops.', _sources=[2])
        unrelated = dict(type='fact', content='Another user turn.', _sources=[3])
        events = []
        from app import memory_debug
        with patch.object(memory_debug, 'extraction_event',
                          lambda stage, **fields: events.append(dict(stage=stage, **fields))):
            await coverage.supplement(self.req, None, self.segments, self.groups,
                AsyncMock(return_value=[user, joint, assistant, unrelated]))
        self.assertEqual(self.groups[0], before[0] + [user, joint])
        ignored = next(e for e in events if e['stage'] == 'coverage_unanchored')
        self.assertEqual(ignored['facts'], [assistant, unrelated])

    async def test_assistant_only_retry_does_not_spend_semantic_calls(self):
        raw = dict(type='fact', content='The assistant said that sounds interesting.', triples=[],
                   evidence=[dict(message_index=1, quote=self.req.messages[2].content)])
        with patch.object(llm, 'complete_json', AsyncMock(return_value=dict(facts=[raw]))) as provider:
            result = await add._extract_coverage_segment(self.req, None, 0, [1, 2])
        self.assertEqual(provider.await_count, 1)
        self.assertEqual([f['type'] for f in result], ['episode'])

    async def test_user_fact_is_still_verified_and_primary_path_keeps_assistant_fact(self):
        user = dict(type='fact', content='The user collects blue ceramics.', triples=[],
                    evidence=[dict(message_index=0, quote=self.req.messages[1].content)])
        assistant = dict(type='fact', content='The assistant said that sounds interesting.', triples=[],
                         evidence=[dict(message_index=1, quote=self.req.messages[2].content)])
        results = []
        for extractor, expected_count in [(add._extract_segment_impl, 2), (add._extract_coverage_segment, 1)]:
            with patch.object(llm, 'complete_json', AsyncMock(return_value=dict(facts=[user, assistant]))), \
                    patch.object(add, '_verify_semantics', AsyncMock()) as verify:
                result = await extractor(self.req, None, 0, [1, 2])
            verify.assert_awaited_once()
            self.assertEqual(len(verify.await_args.args[0]), expected_count)
            results.append(next(f for f in result if f.get('_sources') == [1]))
        self.assertEqual(results[0], results[1])


if __name__ == '__main__':
    unittest.main()
