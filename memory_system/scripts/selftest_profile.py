"""Offline tests for the persona layer (IMPROVEMENT_PLAN P0-P3)."""
import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ['AML_FAKE'] = '1'
os.environ['AML_MEMORY_DEBUG_LOG'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from listwise_fixture import from_scores
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
        request = req([('user', p1[1]['content']), ('user', p2[1]['content'])])
        p2[1]['_sources'] = [1]
        self.st.save_messages(request)
        answer = {'items': [{'content': 'The user bakes bread at home',
                             'kind': 'interest', 'basis': 'inferred',
                             'support_ids': [p1[0], p2[0]]}]}
        with patch.object(profile.llm, 'complete_json', AsyncMock(return_value=answer)):
            await profile.consolidate(self.st, request, [p1, p2])
        prefs = self.st.get_by_type('u', ['preference'])
        self.assertEqual(len(prefs), 1)
        self.assertIn('bakes bread', prefs[0]['content'])
        # Two original source observations, rather than request/AMU aliases.
        self.assertIn(prefs[0]['profile_status'], ('stable', 'static'))
        self.assertIsNone(prefs[0]['expires_at'])

        # A second consolidation with a near-identical assertion reinforces
        # the existing trait instead of duplicating it (A-Mem evolution).
        p3 = await self._persisted_pair('The user discussed proofing times for loaves')
        again = {'items': [{'content': 'The user bakes bread at home',
                            'kind': 'interest', 'basis': 'inferred',
                            'support_ids': [p1[0], p3[0]]}]}
        dict_pair = (p3[0], dict(p3[1], _sources=[1]))
        with patch.object(profile.llm, 'complete_json', AsyncMock(return_value=again)):
            second_request = req([('user', p1[1]['content']), ('user', p3[1]['content'])], 'r2')
            self.st.save_messages(second_request)
            await profile.consolidate(self.st, second_request,
                                      [p1, dict_pair])
        prefs = self.st.get_by_type('u', ['preference'])
        self.assertEqual(len(prefs), 1)
        self.assertTrue(any('r2' in key for key in prefs[0]['support_sessions']))

    async def test_existing_preference_does_not_drop_later_item(self):
        first = await self._persisted_pair('The user enjoys gardening', 'preference')
        second = await self._persisted_pair('The user enjoys painting', 'fact')
        second[1]['_sources'] = [1]
        request = req([('user', first[1]['content']), ('user', second[1]['content'])])
        self.st.save_messages(request)
        before = self.st.dependencies_for(first[0])
        answer = {'items': [
            {'content': first[1]['content'], 'basis': 'stated', 'support_ids': [first[0]]},
            {'content': second[1]['content'], 'basis': 'stated', 'support_ids': [second[0]]},
        ]}
        with patch.object(config, 'PROFILE_CONSOLIDATION_ENABLED', True), patch.object(
                profile.llm, 'complete_json', AsyncMock(return_value=answer)):
            await profile.consolidate(self.st, request, [first, second])
        preferences = self.st.get_by_type('u', ['preference'])
        self.assertEqual({p['content'] for p in preferences},
                         {first[1]['content'], second[1]['content']})
        self.assertEqual(self.st.dependencies_for(first[0]), before)
        painted = next(p for p in preferences if p['content'] == second[1]['content'])
        self.assertEqual([d['source_id'] for d in self.st.dependencies_for(painted['id'])],
                         [second[0]])


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
        rule = await put('Always answer the user in Chinese', 'rule')
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
        request = req([('user', 'please forget that I enjoy music festivals')])
        fact = {'content': 'The user requested to forget that they enjoy modern '
                           'electronic music festivals', 'type': 'rule', '_sources': [0]}
        with patch.object(profile.llm, 'complete_json',
                          AsyncMock(return_value={'invalidate_ids': [target]})):
            await profile.apply_forget_rules(self.st, request, [(rule, fact)])
        gone = self.st.get_amus_by_ids([target])
        self.assertEqual(gone, [])  # suppressed: never returned, even by id
        self.assertEqual(self.st.core_profile('u', 10), [])
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

    def test_forget_request_detection(self):
        self.assertTrue(profile.is_forget_request(
            'Please forget that I use TikTok for entertainment.'))
        self.assertTrue(profile.is_forget_request(
            'Please forget the detail about me losing a close friend.'))
        self.assertFalse(profile.is_forget_request("Don't forget the milk."))
        self.assertFalse(profile.is_forget_request('I keep forgetting about my keys.'))
        self.assertEqual(profile.forget_sentences(
            'Any recipe ideas? Please forget that I use meal kits. Thanks!'),
            ['Please forget that I use meal kits.'])

    def test_missing_forget_request_is_synthesized_as_rule(self):
        request = req([('user', 'What are good foods to add? Please forget that I take '
                                'photos during family trips.'),
                       ('assistant', 'Sure.')])
        facts = [{'content': 'The user asked for food ideas', 'type': 'fact',
                  '_sources': [0], '_segment': 0}]
        out = profile.ensure_forget_rules(request, facts, [[0, 1]])
        rules = [f for f in out if f['type'] == 'rule']
        self.assertEqual(len(rules), 1)
        self.assertIn('Please forget that I take photos during family trips.',
                      rules[0]['content'])
        self.assertEqual(rules[0]['_sources'], [0])
        self.assertEqual(rules[0]['evidence'][0]['message_index'], 0)
        self.assertTrue(profile.is_forget_rule(rules[0]))

    def test_forget_request_stored_as_fact_is_retyped_and_not_duplicated(self):
        request = req([('user', 'Please forget that I have a vitamin D deficiency.')])
        facts = [{'content': 'Daniel requested to forget that he has a vitamin D deficiency.',
                  'type': 'fact', '_sources': [0], '_segment': 0}]
        out = profile.ensure_forget_rules(request, facts, [[0]])
        self.assertEqual([f['type'] for f in out], ['rule'])

    def test_existing_rule_is_not_duplicated(self):
        request = req([('user', 'Please forget that I use TikTok.')])
        facts = [{'content': 'The user asked the assistant to forget that they use TikTok.',
                  'type': 'rule', '_sources': [0], '_segment': 0}]
        self.assertEqual(len(profile.ensure_forget_rules(request, facts, [[0]])), 1)

    async def test_run_add_persists_forget_rule_without_extractor_help(self):
        request = req([('user', 'Recipe ideas please. Please forget that I use '
                                'subscription meal kit services.')], 'c9')
        await add.run_add(self.st, request)
        rules = self.st.get_by_type('u', ['rule'])
        self.assertTrue(any('subscription meal kit' in r['content'] for r in rules))
        self.assertTrue(any(profile.is_forget_rule(r) for r in rules))


class SyntheticTimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')

    def tearDown(self):
        self.st.conn.close()

    async def test_undated_messages_remain_undated(self):
        await add.run_add(self.st, req([('user', 'first chunk about bread'),
                                        ('assistant', 'bread answer')], 'c1'))
        await add.run_add(self.st, req([('user', 'second chunk about vinyl')], 'c2'))
        rows = self.st.conn.execute(
            'SELECT request_id, message_index, timestamp FROM source_messages '
            'ORDER BY timestamp').fetchall()
        self.assertEqual([r['timestamp'] for r in rows], [None, None, None])
        self.assertIsNone(self.st.latest_time('u'))



class SearchInjectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')

    def tearDown(self):
        self.st.conn.close()

    async def memory(self, content, **kwargs):
        vec = (await embed([content]))[0]
        return self.st.insert_amu(user_id='u', session_id='s', content=content,
                                  embedding=vec, **kwargs)

    def assert_source_less_rejection_is_unreviewed(self, response, rejected_ids):
        manifest = response.coverage_manifest
        recovery = manifest['cascade']['listwise']['recovery']
        self.assertEqual(manifest['cascade']['listwise']['status'], 'ok')
        self.assertEqual(manifest['rerank_status'], 'partial')
        self.assertTrue(manifest['search_degraded'])
        self.assertEqual(recovery['status'], 'partial')
        self.assertEqual(recovery['reason'], 'no_admissible_review')
        self.assertEqual(set(recovery['unreviewed_ids']), set(rejected_ids))
        self.assertEqual(recovery['reviewed_ids'], [])
        self.assertEqual(recovery['restored_ids'], [])
        self.assertEqual({r['candidate_id'] for r in recovery['omitted']}, set(rejected_ids))
        self.assertEqual({r['reason'] for r in recovery['omitted']}, {'no_visible_source'})
        self.assertEqual(manifest['rerank_errors'], [
            {'stage': 'listwise_recovery', 'error': 'ReviewIncomplete'}])

    async def test_core_profile_shares_final_packet_budget(self):
        rule = await self.memory('Always answer the user in Chinese', type='rule')
        pref = await self.memory('The user bakes bread at home', type='preference')
        rejected = [await self.memory(f'generic fact {i} about kitchens') for i in range(5)]
        req = schemas.SearchRequest(user_id='u', query='kitchen ideas', top_k=2)
        plan = {'intent': 'preference', 'entities': ['kitchen']}

        async def scorer(prompt, *a, **k):
            import re
            if k.get('stage') == 'search.listwise':
                return from_scores([0.9 if 'bakes bread' in row else
                                   0.8 if 'Chinese' in row else 0.1
                                   for row in re.findall(r'^\d+: (.*)$', prompt, re.M)])
            self.fail(f'No original sources are available for another model stage: {k.get("stage")}')

        with patch.object(search, '_understand', AsyncMock(return_value=plan)), \
             patch.object(search.llm, 'complete_json', side_effect=scorer):
            resp = await search.run_search(self.st, req)
        ids = [d.id for d in resp.data]
        # Useful profile evidence and instructions share the final budget;
        # unrelated profiles receive no automatic lexical-route advantage.
        self.assertEqual(set(ids), {rule, pref})
        self.assertEqual(len(resp.data), 2)
        self.assertTrue(resp.packet_hash)
        # These synthetic rejected facts have no original source to review;
        # useful survivors retain their budget, but deletion is unverified.
        self.assert_source_less_rejection_is_unreviewed(resp, rejected)

        # Fact intents retain general memories without a fixed profile prefix.
        req2 = schemas.SearchRequest(user_id='u', query='kitchen ideas', top_k=2)
        with patch.object(search, '_understand', AsyncMock(return_value={
                'intent': 'fact', 'entities': ['kitchen']})), \
             patch.object(search.llm, 'complete_json', side_effect=scorer):
            resp2 = await search.run_search(self.st, req2)
        ids2 = [d.id for d in resp2.data]
        self.assertEqual(len(ids2), 2)  # Core profile shares top_k and token budget
        self.assertEqual(resp2.coverage_manifest['core_bytes'], 0)
        self.assert_source_less_rejection_is_unreviewed(resp2, rejected)

    async def test_rule_survives_rerank_rejection(self):
        rule = await self.memory('Always respond politely to the user', type='rule')
        rejected = await self.memory('jazz festival lineups 2023')
        req = schemas.SearchRequest(user_id='u', query='jazz events', top_k=5,
                                    options=['A. x', 'B. y'])

        async def scorer(prompt, *a, **k):
            import re
            if k.get('stage') == 'search.listwise':
                self.assertNotIn('Always respond politely', prompt)
                return from_scores([0.0
                                   for i in re.findall(r'^\d+:', prompt, re.M)])
            self.fail(f'No original sources are available for another model stage: {k.get("stage")}')

        with patch.object(search, '_understand', AsyncMock(return_value={
                'intent': 'preference', 'entities': ['jazz']})), \
             patch.object(search.llm, 'complete_json', side_effect=scorer):
            resp = await search.run_search(self.st, req)
        self.assertEqual([d.id for d in resp.data], [rule])
        self.assert_source_less_rejection_is_unreviewed(resp, [rejected])


class DirectChoiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_choice_routes_original_packet_to_verified_answer(self):
        qa = {'question': 'Summer event ideas?',
              'options': ['A. An electronic music festival', 'B. A local food fair'],
              'gold_labels': ['B'], 'qa_type': 'single_choice', 'scoring': 'choice'}
        memories = [{'memory_type': 'rule', 'content':
                     'The user requested to forget their electronic music festival interest'},
                    {'memory_type': 'episode', 'content':
                     'Claire shared a story about her garden; the user asked to edit it.'}]
        entries = [{'letter': 'A', 'kind': 'persona', 'status': 'forbidden',
                    'claims': [], 'validation_errors': []},
                   {'letter': 'B', 'kind': 'generic', 'status': 'eligible',
                    'claims': [], 'validation_errors': []}]

        async def verified_answer(actual_qa, packet, diagnostics):
            self.assertIs(actual_qa, qa)
            self.assertIs(packet, memories)
            diagnostics.update(answer_policy='verified-source-choice-v1',
                               choice_alignment=entries, answer_validation='ok')
            return 'B'

        answer = AsyncMock(side_effect=verified_answer)
        module = SimpleNamespace(VERSION='verified-source-choice-v1', answer=answer)
        unexpected = AsyncMock(side_effect=AssertionError('unexpected direct answer or judge'))
        with patch.object(scoring, 'answer_choice', module, create=True), \
             patch.object(scoring.llm, 'complete_json', unexpected), \
             patch.object(scoring.llm, 'complete', unexpected):
            pred, score, diag = await scoring.evaluate(qa, memories)
        self.assertEqual((pred, score), ('B', 1.0))
        self.assertEqual(diag['answer_policy'], module.VERSION)
        self.assertEqual(diag['choice_alignment'], entries)
        self.assertEqual(diag['answer_validation'], 'ok')
        unexpected.assert_not_awaited()
        answer.assert_awaited_once()

    async def test_answer_failure_keeps_explicit_diagnostics(self):
        qa = {'question': 'q', 'options': ['A. x', 'B. y'], 'gold_labels': ['B'],
              'qa_type': 'single_choice', 'scoring': 'choice'}

        async def failure(qa, memories, diagnostics):
            diagnostics['answer_validation_errors'] = ['missing_options']
            raise RuntimeError('offline')

        with patch.object(scoring.answer_choice, 'answer', AsyncMock(side_effect=failure)):
            pred, score, diag = await scoring.evaluate(qa, [])
        self.assertEqual((pred, score), ('', 0.0))
        self.assertEqual(diag['error_stage'], 'answer')
        self.assertEqual(diag['error_type'], 'RuntimeError')
        self.assertEqual(diag['answer_policy'], scoring.answer_choice.VERSION)
        self.assertEqual(diag['answer_validation_errors'], ['missing_options'])


if __name__ == '__main__':
    unittest.main()
