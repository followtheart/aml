"""Offline regressions for bounded recall, premise queries and source identity."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, personal_evidence as pe, retrieval_queries as rq
from app import schemas, search_pipeline as search, store
from app.embeddings import embed


def source(text, index=0, role='user', timestamp=None):
    return dict(request_id='r', message_index=index, role=role, content=text, timestamp=timestamp)


class QueryTests(unittest.TestCase):
    def test_short_premises_replace_advice_and_generic_slot_can_be_empty(self):
        options = ['A. Since you bake bread, try a recipe.', 'B. Try a relaxing recipe.']
        specs = rq.build('Kitchen ideas?', options, {'option_queries': ['bake bread', '']})
        self.assertEqual([s['text'] for s in specs], ['Kitchen ideas?', 'bake bread'])
        self.assertEqual(specs[1]['option_index'], 0)

    def test_missing_malformed_or_unrelated_entries_preserve_option_fallback(self):
        options = ['A. Since you collect rare vinyl, try a marketplace',
                   'B. If you do not own a dog, consider visiting friends']
        for value in [None, {}, ['vinyl'], [None, 4], ['Kansas basketball dog ownership', '']]:
            specs = rq.build('Weekend ideas?', options, {'option_queries': value})
            self.assertIn('Since you collect rare vinyl', [s['text'] for s in specs])
            if value != ['Kansas basketball dog ownership', '']:
                self.assertIn('If you do not own a dog', [s['text'] for s in specs])

    def test_unrelated_profile_expansion_cannot_contaminate_choice_queries(self):
        specs = rq.build('Kitchen projects?', ['A. Baking bread'], {
            'expanded_queries': ['Kitchen projects for a teacher who owns a calm dog']})
        self.assertNotIn('dog', ' '.join(s['text'] for s in specs))

    def test_long_fallback_is_bounded_and_options_precede_expansions(self):
        options = [f'{chr(65+i)}. Distinct premise {i} ' + 'word ' * 100 for i in range(4)]
        specs = rq.build('Question?', options, {'expanded_queries': ['Question?'] * 10})
        self.assertLessEqual(len(specs), 6)
        self.assertEqual([s['option_index'] for s in specs[1:]], [0, 1, 2, 3])
        self.assertTrue(all(len(s['text']) <= 240 for s in specs[1:]))


class SourceTests(unittest.TestCase):
    def test_sentence_evidence_beats_long_scattered_keyword_overlap(self):
        sources = [source('My work involves effort. My school is local. My student likes books. '
                          'My graduation was years ago. I support steady influence.'),
                   source('Marcus wrote: I mentored a student who left school before graduation. '
                          'I wondered whether my efforts made a difference.', 1, 'assistant'),
                   source('Support has many meanings. School policy varies. Graduation is a milestone. '
                          'Effort matters. Students learn. Influence changes.', 2, 'assistant')]
        selected = pe.compact_sources({}, sources, 'mentored student left school before graduation efforts difference', 128, 1)
        visible = [s for s in selected if s.get('content')]
        self.assertEqual([s['message_index'] for s in visible], [1])
        self.assertIn('Marcus wrote:', visible[0]['content'])

    def test_later_matched_sentence_keeps_attribution_and_negation(self):
        text = 'Marcus wrote: ' + 'We met regularly and discussed assignments. ' * 10
        text += 'I did not cause the student to leave school. That decision was theirs.'
        shown, lo, hi = pe.excerpt(text, 'student leave school', 128)
        self.assertIn('Marcus wrote:', shown)
        self.assertIn('I did not cause', shown)
        self.assertEqual(text[lo:hi], shown)

    def test_duplicate_source_cannot_use_up_second_evidence_slot(self):
        text = 'I enjoy bread baking at home.'
        sources = [source(text), source(text, 1), source('Claire wrote: I do not bake bread.', 2, 'assistant')]
        rows = pe.compact_sources({}, sources, 'bread baking', 128, 2)
        self.assertEqual(sum(bool(s.get('content')) for s in rows), 2)
        self.assertEqual(sum(s.get('content_omitted') == 'duplicate_source' for s in rows), 1)
        self.assertTrue(rows[2].get('content'))

    def test_chinese_attribution_and_negation_survive_late_excerpt(self):
        text = '張三說：' + '我們每週談談功課。' * 30 + '我沒有要求學生退學。這是學生自己的決定。'
        shown, lo, hi = pe.excerpt(text, '學生退學', 128)
        self.assertIn('張三說：', shown)
        self.assertIn('我沒有要求學生退學', shown)
        self.assertEqual(text[lo:hi], shown)

    def test_different_role_or_time_is_not_deduplicated(self):
        for other in [source('I enjoy bread.', 1, 'assistant'), source('I enjoy bread.', 1, timestamp=42)]:
            rows = pe.compact_sources({}, [source('I enjoy bread.'), other], 'bread', 128, 2)
            self.assertTrue(all(r.get('content') for r in rows))


class RecallTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')

    def tearDown(self):
        self.st.conn.close()

    async def memory(self, text, kind='fact'):
        aid = self.st.insert_amu(user_id='u', session_id='s', content=text, type=kind,
                                embedding=(await embed([text]))[0])
        req = schemas.AddRequest(request_id=aid, user_id='u', session_id='s',
                                 messages=[schemas.Message(role='user', content=text)])
        self.st.save_messages(req)
        self.st.link_sources(aid, aid, [0])
        return aid

    async def test_recall_limits_do_not_grow_with_final_topk(self):
        for i in range(8):
            await self.memory(f'I enjoy garden planting project {i}.', 'preference')
        plan = {'intent': 'preference'}
        with patch.multiple(config, RECALL_VECTOR_LIMIT=4, RECALL_SOURCE_LIMIT=3,
                            RECALL_FTS_LIMIT=2, RECALL_PROFILE_LIMIT=1, RECALL_EXPANSION_LIMIT=0):
            await search._recall(self.st, schemas.SearchRequest(user_id='u', query='garden planting', top_k=100), plan)
        counts = {r['channel']: len(r['candidates']) for r in plan['_routes']}
        self.assertEqual(counts, dict(vector=4, source_text=3, full_text=2, profile_rule=1))

    async def test_direct_coverage_skips_expansions_even_with_legacy_expand_flag(self):
        for i in range(4):
            await self.memory(f'I have garden plants {i}.')
        plan = {'intent': 'fact', '_expand': True}
        with patch.object(self.st, 'triples_for_user', side_effect=AssertionError('unneeded graph')):
            await search._recall(self.st, schemas.SearchRequest(user_id='u', query='garden plants'), plan)
        self.assertFalse(plan['_expansion']['enabled'])
        self.assertFalse({'graph', 'scene'} & {r['channel'] for r in plan['_routes']})

    async def test_multi_hop_and_sparse_recall_keep_bounded_expansion(self):
        # Unsourced fillers own the direct top-3 seats, so the bridge is never a seed.
        for i in range(3):
            self.st.insert_amu(user_id='u', session_id='s', content=f'Nora visited Kyoto in spring {i}.',
                               embedding=(await embed([f'Nora visited Kyoto in spring {i}.']))[0])
        for intent, reason in [('multi_hop', 'intent'), ('fact', 'sparse_direct')]:
            aid = await self.memory('Nora teaches Japanese in Kyoto.')
            bridge = await self.memory('The Gion festival runs every July.')
            self.st.insert_triple('u', 'Nora', 'lives_in', 'Kyoto', aid)
            self.st.insert_triple('u', 'Kyoto', 'hosts', 'Gion festival', bridge)
            plan = {'intent': intent, 'entities': ['Nora']}
            with patch.object(config, 'RECALL_EXPANSION_LIMIT', 4):
                await search._recall(self.st, schemas.SearchRequest(user_id='u', query='Nora Kyoto'), plan)
            expanded = [r for r in plan['_routes'] if r['channel'] in ('graph', 'scene')]
            self.assertEqual(plan['_expansion']['reason'], reason)
            self.assertLessEqual(sum(len(r['candidates']) for r in expanded), 4)
            graph_ids = {c['id'] for r in plan['_routes'] if r['channel'] == 'graph' for c in r['candidates']}
            # The graph lane brings what the seeds lead to, never a seed itself.
            self.assertIn(bridge, graph_ids)
            self.assertFalse(graph_ids & set(plan['_graph_seed_ids']))
            self.assertIn(aid, {c['id'] for r in plan['_routes'] for c in r['candidates']})

    async def test_choice_planner_does_not_read_unverified_profile(self):
        with patch.object(config, 'QUERY_PROFILE_DIGEST', True), \
                patch.object(self.st, 'core_profile', side_effect=AssertionError('profile leak')), \
                patch.object(search.llm, 'complete_json', AsyncMock(return_value={'intent': 'preference'})):
            await search._understand(self.st, schemas.SearchRequest(user_id='u', query='kitchen', options=['A. Bread']))

    async def test_profile_quota_does_not_remove_user_instructions(self):
        rules = {await self.memory(f'Please use concise replies in context {i}.', 'rule') for i in range(5)}
        for i in range(4):
            await self.memory(f'I enjoy garden planting {i}.', 'preference')
        plan = {'intent': 'preference'}
        with patch.object(config, 'RECALL_PROFILE_LIMIT', 1):
            await search._recall(self.st, schemas.SearchRequest(user_id='u', query='garden planting'), plan)
        rows = next(r['candidates'] for r in plan['_routes'] if r['channel'] == 'profile_rule')
        self.assertTrue(rules <= {r['id'] for r in rows})
        self.assertEqual(sum(r['type'] != 'rule' for r in rows), 1)

    async def test_same_source_episode_duplicates_merge_but_distinct_status_survives(self):
        first = await self.memory('Marcus wrote: I mentored a student.', 'episode')
        second = self.st.insert_amu(user_id='u', session_id='s', content='Duplicate episode', type='episode')
        self.st.link_sources(second, first, [0])
        third = self.st.insert_amu(user_id='u', session_id='s', content='Disputed episode',
                                  type='episode', resolution_status='disputed')
        self.st.link_sources(third, first, [0])
        plan = {}
        rows = search._prepare_candidates(self.st, schemas.SearchRequest(user_id='u', query='student'),
                                          plan, self.st.get_amus_by_ids([first, second, third]))
        self.assertEqual(len(rows), 2)
        merged = next(r for r in rows if r['id'] != third)
        self.assertEqual({merged['id'], *merged['_equivalent_ids']}, {first, second})
        packet, _, _ = search._pack_evidence(self.st, schemas.SearchRequest(user_id='u', query='student'),
                                             plan, rows, '2026-09-18T00:00:00Z')
        self.assertTrue(any('Marcus' in p['content'] for p in packet))


if __name__ == '__main__':
    unittest.main()
