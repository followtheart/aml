"""Evidence recovery regressions from the 8/26 audit; isolated, no providers."""
import copy
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from listwise_fixture import from_scores
from app import (add_pipeline as add, answer_context, config, cross_encoder, evidence_packet,
                 graph, personal_evidence, schemas, search_pipeline as search, store)


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')

    def tearDown(self):
        self.st.conn.close()

    def memory(self, content='Asked for editing help.', sources=(), user='u', **kwargs):
        aid = self.st.insert_amu(user_id=user, session_id='s', content=content, **kwargs)
        if sources:
            messages = [s if isinstance(s, schemas.Message) else schemas.Message(role=r, content=s)
                        for r, s in sources]
            self.st.save_messages(schemas.AddRequest(request_id=aid, user_id=user,
                                  session_id='s', messages=messages))
            self.st.link_sources(aid, aid, list(range(len(messages))))
        return aid

    def req(self, query='fashion Milan runway', **kwargs):
        return schemas.SearchRequest(user_id='u', query=query, **kwargs)

    async def test_source_only_match_reaches_reranker_and_answer(self):
        aid = self.memory(sources=[('assistant', 'I follow runway fashion in Milan.')], type='episode')
        self.assertEqual(self.st.fts_search('u', 'Milan', 10), [])
        async def rank(prompt, *args, **kwargs):
            self.assertIn('runway fashion in Milan', prompt)
            self.assertIn('role: assistant', prompt)
            return from_scores([0.9])
        with patch.object(search, '_understand', AsyncMock(return_value={'intent': 'fact'})), \
                patch.object(search.llm, 'complete_json', side_effect=rank) as call:
            response = await search.run_search(self.st, self.req(top_k=1))
        self.assertEqual(call.call_count, 1)
        self.assertIn('runway fashion in Milan', call.call_args.args[0])
        self.assertEqual([m.id for m in response.data], [aid])
        self.assertIn('runway fashion in Milan', answer_context.build([m.model_dump() for m in response.data]))

    def test_source_search_privacy_user_cold_and_current_gates(self):
        source = [('user', 'Facebook family updates')]
        current = self.memory(sources=source)
        old = self.memory(sources=source, valid_from='2020-01-01', valid_to='2024-01-01')
        secret = self.memory(sources=[('user', schemas.Message(role='user',
                             content='Facebook family updates', sensitivity='sensitive'))])
        suppressed = self.memory(sources=source, sensitivity='suppressed')
        stale = self.memory(sources=source)
        retracted = self.memory(sources=source)
        cold = self.memory(sources=source)
        self.memory(sources=source, user='other')
        self.st.conn.execute("UPDATE amu SET view_status='stale' WHERE id=?", (stale,))
        self.st.conn.execute("UPDATE amu SET resolution_status='retracted' WHERE id=?", (retracted,))
        self.st.conn.execute("UPDATE amu SET tier='cold' WHERE id=?", (cold,))
        self.st.conn.commit()
        self.assertEqual([m['id'] for m in self.st.source_search('u', 'Facebook', 1)], [current])
        ids = {m['id'] for m in self.st.source_search('u', 'Facebook', 20,
               include_history=True, include_sensitive=True, include_cold=True)}
        self.assertEqual(ids, {current, old, secret, cold})
        self.assertNotIn(suppressed, ids)
        # Even a corrupted legacy AMU label cannot expose a sensitive source hit.
        self.st.conn.execute("UPDATE amu SET sensitivity='normal' WHERE id=?", (secret,))
        self.assertNotIn(secret, {m['id'] for m in self.st.source_search('u', 'Facebook', 20)})

    def test_source_search_cjk(self):
        aid = self.memory(sources=[('user', '我每周练习冲浪，也喜欢看漫画。')])
        self.assertEqual([m['id'] for m in self.st.source_search('u', '冲浪', 10)], [aid])

    def test_source_index_staged_rollback_snapshot_and_purge(self):
        with patch.object(store, '_now', return_value='2024-06-01T00:00:00+00:00'):
            aid = self.memory(sources=[('user', 'Facebook updates')], valid_from='2024-01-01')
        with self.assertRaises(RuntimeError):
            with self.st.staged('u') as staged:
                staged.save_messages(schemas.AddRequest(request_id='rollback', user_id='u', session_id='s',
                                     messages=[schemas.Message(role='user', content='discard me')]))
                raise RuntimeError('rollback')
        self.assertEqual(self.st.conn.execute("SELECT COUNT(*) FROM source_fts WHERE request_id='rollback'").fetchone()[0], 0)
        with patch.object(store, '_now', return_value='2024-06-10T00:00:00+00:00'):
            self.st.close_validity(aid, '2024-06-10T00:00:00+00:00')
            self.memory(sources=[('user', 'Facebook future source')])
        with self.st.snapshot('u', as_of='2024-06-05T00:00:00+00:00') as snap:
            self.assertEqual([m['id'] for m in snap.source_search('u', 'Facebook', 10)], [aid])
        self.st.purge_user('u')
        self.assertEqual(self.st.source_search('u', 'Facebook', 10, include_history=True), [])
        self.assertEqual(self.st.conn.execute('SELECT COUNT(*) FROM source_fts').fetchone()[0], 0)

    def test_legacy_source_index_backfill_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'old.db')
            st = store.Store(path)
            st.save_messages(schemas.AddRequest(request_id='r', user_id='u', session_id='s',
                             messages=[schemas.Message(role='user', content='Facebook updates')]))
            aid = st.insert_amu(user_id='u', session_id='s', content='Meta summary')
            st.link_sources(aid, 'r', [0])
            state = st.user_state('u')
            st.conn.execute('DROP TABLE source_fts')
            st.conn.commit()
            st.conn.close()
            for _ in range(2):
                st = store.Store(path)
                try:
                    self.assertEqual([m['id'] for m in st.source_search('u', 'Facebook', 10)], [aid])
                    self.assertEqual(st.conn.execute('SELECT COUNT(*) FROM source_fts').fetchone()[0], 1)
                    self.assertEqual(st.user_state('u'), state)
                finally:
                    st.conn.close()

    async def test_rejected_claim_retains_original_not_meta_summary(self):
        req = schemas.AddRequest(request_id='r', user_id='u', session_id='s', messages=[
            schemas.Message(role='user', content='I tried surfing with my family.')])
        data = {'facts': [{'content': 'I own a yacht.', 'type': 'fact', 'evidence': [
                            {'message_index': 0, 'quote': 'I own a yacht.'}]}],
                'episode': {'narrative': 'Asked for advice.'}}
        with patch.object(add.llm, 'complete_json', AsyncMock(return_value=data)):
            facts = await add._extract_segment(req, None, 0, [0])
        self.assertEqual(len(facts), 1)
        self.assertIn('I tried surfing with my family.', facts[0]['content'])
        self.assertNotIn('yacht', facts[0]['content'])
        self.assertEqual((facts[0]['type'], facts[0]['sensitivity'], facts[0]['epistemic_status']),
                         ('episode', 'normal', 'observed'))
        # Rejection never clears an explicit model or source privacy annotation.
        for source_sensitive in (False, True):
            req.messages[0].sensitivity = 'sensitive' if source_sensitive else 'normal'
            data['facts'][0]['sensitivity'] = 'normal' if source_sensitive else 'sensitive'
            with patch.object(add.llm, 'complete_json', AsyncMock(return_value=data)):
                facts = await add._extract_segment(req, None, 0, [0])
            self.assertEqual(facts[0]['sensitivity'], 'sensitive')

    def test_assistant_only_advice_is_not_a_user_rule(self):
        text = 'Try playing fetch with your dog.'
        req = schemas.AddRequest(request_id='r', user_id='u', session_id='s',
                                 messages=[schemas.Message(role='assistant', content=text)])
        fact = add._validate_fact({'content': text, 'type': 'rule',
                   'evidence': [{'message_index': 0, 'quote': text}]}, req.messages, 0, req)
        self.assertEqual((fact['type'], fact['epistemic_status']), ('fact', 'inferred'))

    async def test_coverage_reservation_and_ce_promote_candidate_beyond_coarse_head(self):
        rows = [{'id': f'm{i}', 'content': f'record {i}', '_fused': 1-i/100}
                for i in range(90)]
        rows[60]['_coverage_ids'] = ['rare']
        rows[60]['_supported_coverage_ids'] = ['rare']
        plan = {'_coverage_requirements': [{'id': 'rare', 'text': 'record 60'}]}
        submitted = []
        async def ce(query, documents, **kwargs):
            submitted.extend(documents)
            return [12.5 if text == 'record 60' else -3.0 for text in documents]
        async def rank(prompt, *args, **kwargs):
            lines = re.findall(r'^\d+: (.*)$', prompt, re.M)
            self.assertLessEqual(len(lines), 10)
            return from_scores([.99 if 'record 60' in line else .1 for line in lines])
        with patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(search.llm, 'complete_json', side_effect=rank) as call:
            ranked = await search._filter_rerank(self.req(top_k=50), plan, rows)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(len(set(submitted)), 50)
        for stage in ('coarse', 'fine', 'listwise'):
            self.assertIn('m60', plan['_cascade'][stage]['selected_ids'])
        self.assertEqual(ranked[0]['id'], 'm60')
        self.assertEqual(ranked[0]['_final'], 12.5)
        self.assertEqual([m['id'] for m in ranked if m['_cascade_selected']], ['m60'])

    async def test_invalid_listwise_permutations_fall_back_to_ce_order(self):
        for ranking in ([0], [0, 0], [0, True], [0, 9]):
            with self.subTest(ranking=ranking), patch.object(search.llm, 'complete_json',
                    AsyncMock(return_value=dict(ranking=ranking, irrelevant=[], groups=[]))) as call, \
                    patch.object(cross_encoder, 'rerank', AsyncMock(return_value=[.2, .8])):
                plan = {}
                ranked = await search._filter_rerank(self.req(), plan,
                          [dict(id='a', content='one'), dict(id='b', content='two')])
                self.assertEqual([m['id'] for m in ranked], ['b', 'a'])
                self.assertEqual([m['_final'] for m in ranked], [.8, .2])
                self.assertTrue(all(m['_cascade_selected'] for m in ranked))
                self.assertEqual(call.call_count, 2)
                self.assertEqual(plan['_cascade']['listwise']['status'], 'fallback')

    def test_fashion_draft_beats_editorial_messages(self):
        aid = self.memory(type='episode', sources=[('user', 'Please refine this note.'),
             ('assistant', 'I follow Milan runway fashion and designers.'),
             ('user', 'Please improve this biography.'), ('assistant', 'Mark is an English teacher.')])
        plan = {}
        rows = search._prepare_candidates(self.st, self.req(), plan, self.st.get_amus_by_ids([aid]))
        self.assertIn('Milan runway', rows[0]['_rank_text'])
        packet, _, _ = search._pack_evidence(self.st, self.req(), plan, rows, '2026-09-17T00:00:00Z')
        self.assertIn('Milan runway', packet[0]['content'])
        self.assertIn('role: assistant', packet[0]['content'])

    def test_excerpt_keeps_nearby_subject_and_negation(self):
        text = 'Unrelated preface. ' * 30 + 'Claire described her garden. I do not have a garden. ' + 'Closing note. ' * 30
        shown, start, end = personal_evidence.excerpt(text, 'garden', 160)
        self.assertEqual(shown, text[start:end])
        self.assertIn('Claire', shown)
        self.assertIn('I do not have a garden.', shown)

    def test_document_episode_keeps_full_original_once(self):
        text = 'Claire described her garden. ' * 40
        aid = self.memory('user: ' + text, type='episode', sources=[('user', text)])
        packet, _, _ = search._pack_evidence(self.st, self.req(), {'intent': 'document'},
                         self.st.get_amus_by_ids([aid]), '2026-09-17T00:00:00Z')
        self.assertEqual(packet[0]['content'].count(text), 1)
        self.assertEqual(packet[0]['sources'][0]['content'], text)

    def test_coalesced_preferences_keep_all_source_provenance(self):
        a = self.memory('The user enjoys bread.', type='preference', sources=[('user', 'I bake bread.')])
        b = self.memory('The user enjoys bread', type='preference', sources=[('user', 'I make sourdough.')],
                        temporal=dict(raw=None, start=None, end=None, precision='unknown', reference_time=None))
        rows = self.st.get_amus_by_ids([a, b])
        plan = {}
        candidates = search._prepare_candidates(self.st, self.req('bread'), plan, rows)
        self.assertEqual(len(candidates), 1)
        packet, _, _ = search._pack_evidence(self.st, self.req('bread', top_k=1), plan, candidates, '2026-09-17T00:00:00Z')
        self.assertEqual({packet[0]['id'], *packet[0]['equivalent_ids']}, {a, b})
        self.assertEqual({s['request_id'] for s in packet[0]['sources']}, {a, b})
        self.assertIn('I bake bread.', packet[0]['content'])
        self.assertIn('I make sourdough.', packet[0]['content'])
        changed = copy.deepcopy(packet)
        changed[0]['equivalent_ids'] = []
        with self.assertRaises(ValueError):
            answer_context.build(changed)

    def test_same_wording_with_different_known_times_is_not_coalesced(self):
        a = self.memory('The user enjoys bread.', type='preference', valid_from='2024-01-01')
        b = self.memory('The user enjoys bread', type='preference', valid_from='2025-01-01')
        rows = search._coalesce_preferences(self.st.get_amus_by_ids([a, b]), {})
        self.assertEqual(len(rows), 2)

    def test_constraints_do_not_consume_evidence_slots_but_obey_byte_budget(self):
        rules = [dict(id=f'r{i}', content=f'Forget trait {i}.', is_constraint=True) for i in range(6)]
        facts = [dict(id=f'f{i}', content=f'evidence {i}') for i in range(50)]
        packet, _, manifest = evidence_packet.pack(rules + facts, 50, 32000)
        self.assertEqual((len(packet), manifest['evidence_count'], manifest['constraint_count']), (56, 50, 6))
        changed = copy.deepcopy(packet)
        changed[0]['is_constraint'] = False
        with self.assertRaises(ValueError):
            answer_context.build(changed)
        packed, _, manifest = evidence_packet.pack(rules + facts, 50, 70)
        self.assertLessEqual(len(answer_context.build(packed).encode('utf-8')), 70)
        self.assertTrue(any(m['reason'] == 'token_budget' for m in manifest['omitted']))

    def test_graph_actual_seed_recovers_neighbor_without_literal_query_match(self):
        triples = [dict(subject='Daniel', relation='visits', object='Comic-Con', amu_id='a'),
                   dict(subject='Comic-Con', relation='features', object='Comics', amu_id='b'),
                   dict(subject='Other', relation='owns', object='Car', amu_id='c')]
        selected = graph.filter_triples(triples, 'family entertainment', ['hobbies'], seed_amu_ids=['a'])
        recalled = graph.ppr_recall(selected, ['hobbies'], seed_amu_ids=['a'])
        self.assertEqual(set(recalled), {'a', 'b'})


if __name__ == '__main__':
    unittest.main()
