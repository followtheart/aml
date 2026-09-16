"""Offline tests for the persona layer (IMPROVEMENT_PLAN P0-P3)."""
import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ['AML_FAKE'] = '1'
os.environ['AML_MEMORY_DEBUG_LOG'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import add_pipeline as add, config, eval_scoring as scoring, profile, schemas
from app import search_pipeline as search, store
from app.embeddings import embed


def req(messages, request_id='r1', user_id='u', session_id='s'):
    return schemas.AddRequest(
        request_id=request_id, user_id=user_id, session_id=session_id,
        messages=[schemas.Message(role=r, content=c) for r, c in messages])


class ConsolidationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')

    def tearDown(self):
        self.st.conn.close()

    async def _persisted_pair(self, content, type_='fact'):
        vec = (await embed([content]))[0]
        aid = self.st.insert_amu(user_id='u', session_id='s', content=content,
                                 type=type_, embedding=vec)
        return aid, {'content': content, 'type': type_, '_sources': [0],
                     'evidence': [{'message_index': 0, 'quote': content}]}

    async def test_consolidation_creates_preference_then_merges(self):
        p1 = await self._persisted_pair('The user asked why bread baking feels satisfying')
        p2 = await self._persisted_pair('The user asked about sourdough hydration ratios')
        request = req([('user', 'tell me about bread')])
        answer = {'items': [{'content': 'The user bakes bread at home',
                             'kind': 'interest', 'basis': 'inferred',
                             'support_ids': [p1[0], p2[0]]}]}
        with patch.object(profile.llm, 'complete_json', AsyncMock(return_value=answer)):
            await profile.consolidate(self.st, request, [p1, p2])
        prefs = self.st.get_by_type('u', ['preference'])
        self.assertEqual(len(prefs), 1)
        self.assertIn('bakes bread', prefs[0]['content'])
        # session + request + two evidence ids -> stable beyond the threshold
        self.assertIn(prefs[0]['profile_status'], ('stable', 'static'))
        self.assertIsNone(prefs[0]['expires_at'])

        # A second consolidation with a near-identical assertion reinforces
        # the existing trait instead of duplicating it (A-Mem evolution).
        p3 = await self._persisted_pair('The user discussed proofing times for loaves')
        again = {'items': [{'content': 'The user bakes bread at home',
                            'kind': 'interest', 'basis': 'inferred',
                            'support_ids': [p1[0], p3[0]]}]}
        with patch.object(profile.llm, 'complete_json', AsyncMock(return_value=again)):
            await profile.consolidate(self.st, req([('user', 'more bread')], 'r2'),
                                      [p1, p3])
        prefs = self.st.get_by_type('u', ['preference'])
        self.assertEqual(len(prefs), 1)
        self.assertIn('r2', prefs[0]['support_sessions'])

    async def test_consolidation_rejects_noise(self):
        p1 = await self._persisted_pair('The weather was nice')
        bad = {'items': [{'content': 'Bread is baked in ovens',  # not about the user
                          'kind': 'interest', 'basis': 'stated', 'support_ids': [p1[0]]},
                         {'content': 'The user likes bread', 'kind': 'interest',
                          'basis': 'inferred', 'support_ids': [p1[0]]},  # 1 support < 2
                         {'content': 'The user likes bread', 'kind': 'interest',
                          'basis': 'stated', 'support_ids': ['amu_nonexistent']}]}
        with patch.object(profile.llm, 'complete_json', AsyncMock(return_value=bad)):
            await profile.consolidate(self.st, req([('user', 'bread?')]), [p1])
        self.assertEqual(self.st.get_by_type('u', ['preference']), [])

    async def test_core_profile_orders_rules_first_and_caps(self):
        async def put(content, type_):
            vec = (await embed([content]))[0]
            return self.st.insert_amu(user_id='u', session_id='s', content=content,
                                      type=type_, embedding=vec)
        pref = await put('The user bakes bread at home', 'preference')
        rule = await put('The user asked to forget vinyl', 'rule')
        fact = await put('Bread baking is satisfying', 'fact')
        core = self.st.core_profile('u', 10)
        self.assertEqual([c['id'] for c in core], [rule, pref])
        self.assertNotIn(fact, [c['id'] for c in core])


class ForgetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')

    def tearDown(self):
        self.st.conn.close()

    async def test_forget_invalidates_target_and_hides_it_everywhere(self):
        vec = (await embed(['The user enjoys modern electronic music festivals']))[0]
        target = self.st.insert_amu(
            user_id='u', session_id='s',
            content='The user enjoys modern electronic music festivals',
            type='preference', embedding=vec)
        self.st.insert_triple('u', 'user', 'interested_in', 'music festivals', target)
        rvec = (await embed(['The user requested to forget that they enjoy modern '
                             'electronic music festivals']))[0]
        rule = self.st.insert_amu(
            user_id='u', session_id='s',
            content='The user requested to forget that they enjoy modern '
                    'electronic music festivals',
            type='rule', embedding=rvec)
        request = req([('user', 'please forget the music festival thing')])
        fact = {'content': 'The user requested to forget that they enjoy modern '
                           'electronic music festivals', 'type': 'rule'}
        with patch.object(profile.llm, 'complete_json',
                          AsyncMock(return_value={'invalidate_ids': [target]})):
            await profile.apply_forget_rules(self.st, request, [(rule, fact)])
        gone = self.st.get_amus_by_ids([target])
        self.assertEqual(gone, [])  # suppressed: never returned, even by id
        self.assertEqual(self.st.core_profile('u', 10)[0]['id'], rule)
        self.assertEqual(self.st.triples_for_user('u'), [])
        with_history = self.st.get_amus_by_ids([target], include_history=True,
                                               include_sensitive=True)
        self.assertEqual(with_history, [])

    async def test_non_forget_text_does_not_trigger(self):
        vec = (await embed(['The user likes vinyl records']))[0]
        target = self.st.insert_amu(user_id='u', session_id='s',
                                    content='The user likes vinyl records',
                                    type='preference', embedding=vec)
        fact = {'content': 'The user likes vinyl records', 'type': 'preference'}
        await profile.apply_forget_rules(self.st, req([('user', 'hi')]), [(target, fact)])
        self.assertEqual(len(self.st.get_amus_by_ids([target])), 1)


class SyntheticTimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')

    def tearDown(self):
        self.st.conn.close()

    async def test_monotonic_continuation_across_chunks(self):
        await add.run_add(self.st, req([('user', 'first chunk about bread'),
                                        ('assistant', 'bread answer')], 'c1'))
        await add.run_add(self.st, req([('user', 'second chunk about vinyl')], 'c2'))
        rows = self.st.conn.execute(
            'SELECT request_id, message_index, timestamp FROM source_messages '
            'ORDER BY timestamp').fetchall()
        ts = [r['timestamp'] for r in rows]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(rows[0]['timestamp'],
                         config.SYNTHETIC_EPOCH_MS + config.SYNTHETIC_STEP_MS)
        by_req = {}
        for r in rows:
            by_req.setdefault(r['request_id'], []).append(r['timestamp'])
        self.assertGreater(min(by_req['c2']), max(by_req['c1']))
        # The search anchor follows the synthetic timeline, not the wall clock.
        latest = self.st.latest_time('u')
        self.assertTrue(latest.startswith('2020-'), latest)


class SearchInjectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')

    def tearDown(self):
        self.st.conn.close()

    async def memory(self, content, **kwargs):
        vec = (await embed([content]))[0]
        return self.st.insert_amu(user_id='u', session_id='s', content=content,
                                  embedding=vec, **kwargs)

    async def test_core_profile_prepended_outside_topk(self):
        rule = await self.memory('The user asked to forget vinyl', type='rule')
        pref = await self.memory('The user bakes bread at home', type='preference')
        for i in range(5):
            await self.memory(f'generic fact {i} about kitchens')
        req = schemas.SearchRequest(user_id='u', query='kitchen ideas', top_k=2)
        plan = {'intent': 'preference', 'entities': ['kitchen']}

        async def scorer(prompt, *a, **k):
            import re
            if 'relevance scoring module' in prompt:
                return {'scores': [{'id': i, 'relevance': 0.9, 'keep': True}
                                   for i in re.findall(r'^(amu_\w+|summary_\w+):',
                                                       prompt, re.M)]}
            return {'sufficient': True, 'confidence': 0.9, 'missing': '',
                    'follow_up_queries': []}

        with patch.object(search, '_understand', AsyncMock(return_value=plan)), \
             patch.object(search.llm, 'complete_json', side_effect=scorer):
            resp = await search.run_search(self.st, req)
        ids = [d.id for d in resp.data]
        # Persona view (P3) hides generic world-knowledge facts for
        # preference intents; the Core Profile is always prepended.
        self.assertEqual(ids[:2], [rule, pref])
        self.assertTrue(all(d.content.startswith('[core profile]') for d in resp.data[:2]))

        # Fact intents still see generic memories, ranked below the profile.
        req2 = schemas.SearchRequest(user_id='u', query='kitchen ideas', top_k=2)
        with patch.object(search, '_understand', AsyncMock(return_value={
                'intent': 'fact', 'entities': ['kitchen']})), \
             patch.object(search.llm, 'complete_json', side_effect=scorer):
            resp2 = await search.run_search(self.st, req2)
        ids2 = [d.id for d in resp2.data]
        self.assertEqual(ids2[:2], [rule, pref])
        self.assertEqual(len(ids2), 4)  # top_k=2 ranked + 2 injected profile

    async def test_rule_survives_rerank_rejection(self):
        rule = await self.memory('The user asked to forget jazz', type='rule')
        await self.memory('jazz festival lineups 2023')
        req = schemas.SearchRequest(user_id='u', query='jazz events', top_k=5,
                                    options=['A. x', 'B. y'])

        async def scorer(prompt, *a, **k):
            import re
            if 'relevance scoring module' in prompt:
                return {'scores': [{'id': i, 'relevance': 0.0, 'keep': False}
                                   for i in re.findall(r'^(amu_\w+):', prompt, re.M)]}
            return {'sufficient': True, 'confidence': 0.9, 'missing': '',
                    'follow_up_queries': []}

        with patch.object(search, '_understand', AsyncMock(return_value={
                'intent': 'preference', 'entities': ['jazz']})), \
             patch.object(search.llm, 'complete_json', side_effect=scorer):
            resp = await search.run_search(self.st, req)
        self.assertIn(rule, [d.id for d in resp.data])


class ChoiceAlignTests(unittest.IsolatedAsyncioTestCase):
    def test_alignment_reaches_answer_prompt(self):
        qa = {'question': 'Weekend kitchen project ideas?',
              'options': ['A. Bake bread at home', 'B. Order takeout'],
              'gold_labels': ['A'], 'qa_type': 'single_choice', 'scoring': 'choice'}
        memories = [{'memory_type': 'preference',
                     'content': 'The user bakes bread at home'}]
        align = {'options': [{'letter': 'A', 'kind': 'persona', 'supported': True,
                              'evidence': 'bakes bread at home'},
                             {'letter': 'B', 'kind': 'generic', 'supported': False,
                              'evidence': ''}]}
        captured = {}

        async def fake_complete(prompt, **kwargs):
            captured['prompt'] = prompt
            return 'A'

        with patch.object(scoring.llm, 'complete_json',
                          AsyncMock(return_value=align)), \
             patch.object(scoring.llm, 'complete', side_effect=fake_complete):
            pred, score, _ = asyncio.run(scoring.evaluate(qa, memories))
        self.assertEqual((pred, score), ('A', 1.0))
        self.assertIn('Persona evidence alignment', captured['prompt'])
        self.assertIn('A: persona, SUPPORTED', captured['prompt'])

    def test_alignment_failure_fails_open(self):
        qa = {'question': 'q', 'options': ['A. x', 'B. y'], 'gold_labels': ['B'],
              'qa_type': 'single_choice', 'scoring': 'choice'}
        with patch.object(scoring.llm, 'complete_json',
                          AsyncMock(side_effect=RuntimeError('offline'))), \
             patch.object(scoring.llm, 'complete', AsyncMock(return_value='B')):
            pred, score, _ = asyncio.run(scoring.evaluate(qa, []))
        self.assertEqual((pred, score), ('B', 1.0))


if __name__ == '__main__':
    unittest.main()
