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

    async def test_core_profile_shares_final_packet_budget(self):
        rule = await self.memory('Always answer the user in Chinese', type='rule')
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
        # Personal evidence survives; core rows retain their retrieval ranks.
        self.assertEqual(set(ids), {rule, pref})
        self.assertEqual(len(resp.data), 2)
        self.assertTrue(resp.packet_hash)

        # Fact intents retain general memories without a fixed profile prefix.
        req2 = schemas.SearchRequest(user_id='u', query='kitchen ideas', top_k=2)
        with patch.object(search, '_understand', AsyncMock(return_value={
                'intent': 'fact', 'entities': ['kitchen']})), \
             patch.object(search.llm, 'complete_json', side_effect=scorer):
            resp2 = await search.run_search(self.st, req2)
        ids2 = [d.id for d in resp2.data]
        self.assertEqual(len(ids2), 2)  # Core profile shares top_k and token budget
        self.assertLessEqual(resp2.coverage_manifest['core_bytes'],
                             resp2.coverage_manifest['core_budget'])

    async def test_rule_survives_rerank_rejection(self):
        rule = await self.memory('Always respond politely to the user', type='rule')
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


def cited_response(align, qa, memories):
    """Fixtures follow the new explicit claim/source citation contract."""
    import copy
    align = copy.deepcopy(align)
    for entry, option in zip(align['options'], qa['options']):
        entry['option_claim'] = option[3:] if entry['kind'] == 'persona' else ''
        entry['evidence_id'] = next((m.get('id', f'memory_{i}') for i, m in enumerate(memories)
                                    if entry['evidence'] and entry['evidence'] in m['content']
                                    and m['memory_type'] != 'rule'), '')
        entry['unsupported_claims'] = []
        entry['constraint_id'], entry['constraint_span'] = '', ''
        if entry['forbidden']:
            i, rule = next((i, m) for i, m in enumerate(memories) if m['memory_type'] == 'rule')
            entry['constraint_id'] = rule.get('id', f'memory_{i}')
            entry['constraint_span'] = rule['content']
    return align


class ChoiceAlignTests(unittest.IsolatedAsyncioTestCase):
    def test_alignment_reaches_answer_prompt(self):
        qa = {'question': 'Weekend kitchen project ideas?',
              'options': ['A. Bake bread at home', 'B. Order takeout'],
              'gold_labels': ['A'], 'qa_type': 'single_choice', 'scoring': 'choice'}
        memories = [{'memory_type': 'preference',
                     'content': 'The user bakes bread at home'}]
        align = {'options': [{'letter': 'A', 'kind': 'persona', 'match': 'weak',
                              'evidence': 'bakes bread at home', 'forbidden': False},
                             {'letter': 'B', 'kind': 'generic', 'match': 'none',
                              'evidence': '', 'forbidden': False}]}
        captured = {}

        async def fake_complete(prompt, **kwargs):
            captured['prompt'] = prompt
            return 'A'

        with patch.object(scoring.llm, 'complete_json',
                          AsyncMock(return_value=cited_response(align, qa, memories))), \
             patch.object(scoring.llm, 'complete', side_effect=fake_complete):
            pred, score, diag = asyncio.run(scoring.evaluate(qa, memories))
        self.assertEqual((pred, score), ('A', 1.0))
        self.assertIn('Persona evidence alignment', captured['prompt'])
        self.assertIn('A: persona, match=weak', captured['prompt'])
        self.assertIn('Evidence tiers (ties are unordered; not an answer): A > B', captured['prompt'])
        # Alignment records the exact evidence IDs used for its judgements.
        self.assertEqual([e['letter'] for e in diag['choice_alignment']], ['A', 'B'])
        self.assertTrue(diag['choice_alignment'][0]['citation_valid'])
        self.assertEqual(diag['alignment_input_ids'], ['memory_0'])

    def test_alignment_preserves_demographics_with_scope_rules(self):
        # Relevant location/job evidence remains visible, with dependency rules.
        qa = {'question': 'Weekend plans?',
              'options': ['A. As a Kansan you might enjoy a local fair',
                          'B. Order takeout'],
              'gold_labels': ['B'], 'qa_type': 'single_choice', 'scoring': 'choice'}
        memories = [{'memory_type': 'profile',
                     'content': 'The user lives in Lawrence, Kansas'},
                    {'memory_type': 'preference',
                     'content': 'The user bakes bread at home'}]
        captured = {}

        async def fake_complete_json(prompt, *a, **k):
            captured['align_prompt'] = prompt
            return {'options': [{'letter': 'A', 'kind': 'persona', 'match': 'none',
                                 'evidence': '', 'forbidden': False},
                                {'letter': 'B', 'kind': 'generic', 'match': 'none',
                                 'evidence': '', 'forbidden': False}]}

        with patch.object(scoring.llm, 'complete_json', side_effect=fake_complete_json), \
             patch.object(scoring.llm, 'complete', AsyncMock(return_value='B')):
            pred, score, _ = asyncio.run(scoring.evaluate(qa, memories))
        self.assertEqual((pred, score), ('B', 1.0))
        # Relevant location/job evidence remains visible, with dependency rules.
        self.assertIn('The user lives in Lawrence, Kansas', captured['align_prompt'])
        self.assertIn('directly dependent advice', captured['align_prompt'])

    def test_autopick_answers_from_ranked_alignment(self):
        qa = {'question': 'Weekend kitchen project ideas?',
              'options': ['A. Order takeout', 'B. Bake bread at home'],
              'gold_labels': ['B'], 'qa_type': 'single_choice', 'scoring': 'choice'}
        memories = [{'memory_type': 'preference',
                     'content': 'The user bakes bread at home'}]
        align = {'options': [{'letter': 'A', 'kind': 'generic', 'match': 'none',
                              'evidence': '', 'forbidden': False},
                             {'letter': 'B', 'kind': 'persona', 'match': 'strong',
                              'evidence': 'bakes bread at home', 'forbidden': False}]}
        with patch.object(scoring.config, 'CHOICE_AUTOPICK', True), \
             patch.object(scoring.llm, 'complete_json', AsyncMock(return_value=cited_response(align, qa, memories))), \
             patch.object(scoring.llm, 'complete',
                          AsyncMock(side_effect=AssertionError('answer model called'))):
            pred, score, diag = asyncio.run(scoring.evaluate(qa, memories))
        self.assertEqual((pred, score), ('B', 1.0))
        self.assertEqual(diag.get('choice_autopick'), 'B')

    def test_forgotten_trait_marks_option_forbidden(self):
        qa = {'question': 'Summer event ideas?',
              'options': ['A. Since you love electronic festivals, try Tomorrowland',
                          'B. A local food fair'],
              'gold_labels': ['B'], 'qa_type': 'single_choice', 'scoring': 'choice'}
        memories = [{'memory_type': 'rule',
                     'content': 'The user requested to forget that they enjoy modern '
                                'electronic music festivals'},
                    {'memory_type': 'preference',
                     'content': 'The user enjoys modern electronic music festivals'}]
        align = {'options': [{'letter': 'A', 'kind': 'persona', 'match': 'strong',
                              'evidence': 'electronic music festivals', 'forbidden': True},
                             {'letter': 'B', 'kind': 'generic', 'match': 'none',
                              'evidence': '', 'forbidden': False}]}
        captured = {}

        async def fake_complete_json(prompt, *a, **k):
            captured['align_prompt'] = prompt
            return cited_response(align, qa, memories)

        async def fake_complete(prompt, **kwargs):
            captured['prompt'] = prompt
            return 'B'

        with patch.object(scoring.llm, 'complete_json', side_effect=fake_complete_json), \
             patch.object(scoring.llm, 'complete', side_effect=fake_complete):
            pred, score, _ = asyncio.run(scoring.evaluate(qa, memories))
        self.assertEqual((pred, score), ('B', 1.0))
        # The forget rule is handed to the aligner as a forgotten trait, not
        # as ordinary persona evidence.
        forgotten = captured['align_prompt'].split('asked the assistant to FORGET')[1]
        self.assertIn('electronic music festivals', forgotten)
        self.assertIn('A: persona, match=strong, FORBIDDEN', captured['prompt'])
        self.assertIn('Evidence tiers (ties are unordered; not an answer): B > A', captured['prompt'])

    def test_forbidden_autopick_skips_banned_top_option(self):
        qa = {'question': 'Summer event ideas?',
              'options': ['A. Since you love electronic festivals, try Tomorrowland',
                          'B. A local food fair'],
              'gold_labels': ['B'], 'qa_type': 'single_choice', 'scoring': 'choice'}
        memories = [{'memory_type': 'rule',
                     'content': 'The user requested to forget that they enjoy modern '
                                'electronic music festivals'},
                    {'memory_type': 'preference',
                     'content': 'The user enjoys modern electronic music festivals'}]
        align = {'options': [{'letter': 'A', 'kind': 'persona', 'match': 'strong',
                              'evidence': 'electronic music festivals', 'forbidden': True},
                             {'letter': 'B', 'kind': 'generic', 'match': 'none',
                              'evidence': '', 'forbidden': False}]}
        with patch.object(scoring.config, 'CHOICE_AUTOPICK', True), \
             patch.object(scoring.llm, 'complete_json', AsyncMock(return_value=cited_response(align, qa, memories))), \
             patch.object(scoring.llm, 'complete',
                          AsyncMock(side_effect=AssertionError('answer model called'))):
            pred, score, diag = asyncio.run(scoring.evaluate(qa, memories))
        # Autopick requires a non-forbidden persona match; B has none here, so
        # the answer model must be consulted... it is mocked to fail, which
        # proves autopick did NOT fire on a banned-only ranking.
        self.assertEqual((pred, score), ('', 0.0))

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
