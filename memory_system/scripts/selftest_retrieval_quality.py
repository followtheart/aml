"""Retrieval precision with controlled CE logits and listwise decisions.

No provider calls or live database writes. These test policy behavior, not
benchmark accuracy or the relevance calibration of a real model.
"""
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from listwise_fixture import from_scores
from app import (answer_context, config, cross_encoder, evidence_packet, graph_fusion, personal_evidence,
                 schemas, search_pipeline as search, store)
from app.embeddings import embed


class QualityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')
        async def fixed_ce(query, documents, **kwargs):
            return [2.0] * len(documents)
        ce = patch.object(cross_encoder, 'rerank', side_effect=fixed_ce)
        ce.start()
        self.addCleanup(ce.stop)

    def tearDown(self):
        self.st.conn.close()

    async def memory(self, content, source=None, **kwargs):
        aid = self.st.insert_amu(user_id='u', session_id='s', content=content,
                                embedding=(await embed([content]))[0], **kwargs)
        if source:
            self.st.save_messages(schemas.AddRequest(request_id=aid, user_id='u',
                session_id='s', messages=[schemas.Message(role='user', content=source)]))
            self.st.link_sources(aid, aid, [0])
        return aid

    async def retrieve(self, query, scores, **kwargs):
        async def ce(query, documents, **kw):
            self.assertTrue(kw['stage'].startswith('search.cross_encoder.batch_'))
            return [8 * scores(document) - 3 for document in documents]
        async def rank(prompt, *args, **kw):
            self.assertTrue(kw['stage'].startswith('search.listwise'))
            rows = re.findall(r'^\d+: (.*)$', prompt, re.M)
            return from_scores([scores(row) for row in rows])
        with patch.object(search, '_understand', AsyncMock(return_value={'intent': 'fact'})), \
                patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(search.llm, 'complete_json', side_effect=rank):
            return await search.run_search(self.st, schemas.SearchRequest(
                user_id='u', query=query, **kwargs))

    async def test_english_question_words_do_not_recall_unrelated_memories(self):
        await self.memory('The user enjoys pottery and has a kiln.')
        gold = await self.memory('The user works in Kyoto.')
        rows = self.st.fts_search('u', 'Where does the user work in Kyoto?', 10)
        self.assertEqual([m['id'] for m in rows], [gold])

    async def test_cjk_ranks_before_limit_and_preserves_mixed_language(self):
        for i in range(12):
            await self.memory(f'喜歡咖啡。編號 {i}')
        gold = await self.memory('每週在台北的 Kyoto 咖啡館練習日語。')
        self.assertEqual(self.st.fts_search('u', '台北 Kyoto 咖啡館 日語', 1)[0]['id'], gold)
        self.assertIn('日語', personal_evidence.terms('日語'))
        self.assertNotIn('浪台', personal_evidence.terms('衝浪，台北'))

    async def test_cjk_source_route_ranks_by_overlap_not_random_id(self):
        noise = await self.memory('unrelated synopsis', source='咖啡')
        gold = await self.memory('another synopsis', source='台北咖啡館日語交流')
        # Stable adverse ordering: the stronger match is last by ID.
        self.st.conn.execute('UPDATE amu SET id=? WHERE id=?', ('amu_000noise', noise))
        self.st.conn.execute('UPDATE amu_sources SET amu_id=? WHERE amu_id=?', ('amu_000noise', noise))
        self.st.conn.execute('UPDATE amu SET id=? WHERE id=?', ('amu_zzzgold', gold))
        self.st.conn.execute('UPDATE amu_sources SET amu_id=? WHERE amu_id=?', ('amu_zzzgold', gold))
        self.st.conn.commit()
        rows = self.st.source_search('u', '台北咖啡館日語', 1)
        self.assertEqual(rows[0]['id'], 'amu_zzzgold')

    def test_query_variants_use_ranks_not_incomparable_bm25_scores(self):
        a = [dict(id=f'a{i}', _score=100-i) for i in range(5)]
        b = [dict(id='rare-option', _score=0.001)]
        result = search._merge_query_results([a, b], 2)
        self.assertEqual({m['id'] for m in result}, {'a0', 'rare-option'})

    async def test_many_source_links_do_not_starve_distinct_memories(self):
        a = await self.memory('First synopsis')
        b = await self.memory('Second synopsis')
        crowded, other = sorted([a, b])
        for aid, count in ((crowded, 40), (other, 1)):
            self.st.save_messages(schemas.AddRequest(request_id=aid, user_id='u', session_id='s',
                messages=[schemas.Message(role='user', content='Kyoto') for _ in range(count)]))
            self.st.link_sources(aid, aid, list(range(count)))
        self.assertEqual({m['id'] for m in self.st.source_search('u', 'Kyoto', 2)}, {a, b})

    def test_duplicate_ids_within_routes_do_not_amplify_graph_prior(self):
        a = dict(id='a', content='Unrelated car engines', _score=.1)
        b = dict(id='b', content='Maya works in Kyoto', _score=.9)
        def plan():
            return dict(query='Maya Kyoto', _routes=[{'channel': 'vector'}, {'channel': 'full_text'}])
        baseline = graph_fusion.fuse([[a, b], [b]], plan())
        result = graph_fusion.fuse([[a, a, b], [b, b]], plan())
        self.assertEqual(result[0]['id'], 'b')
        self.assertEqual({c['id']: c['_fused'] for c in result},
                         {c['id']: c['_fused'] for c in baseline})

    def test_expansion_routes_do_not_outvote_two_direct_channels(self):
        gold = dict(id='gold', content='Maya works in Kyoto', _score=.9)
        noise = dict(id='noise', content='Maya discusses car engines', _score=.05)
        routes = [[gold, noise], [gold], [noise], [noise]]
        plan = dict(query='Maya Kyoto', intent='fact', _routes=[dict(channel=name)
                    for name in ('vector', 'full_text', 'graph', 'scene')])
        self.assertEqual(graph_fusion.fuse(routes, plan)[0]['id'], 'gold')

    async def test_rare_option_outside_graph_head_survives_each_bounded_rerank_stage(self):
        rows = [dict(id=f'noise{i}', content=f'common indoor topic {i}', _fused=1-i/100,
                     _coverage_ids=['option:0']) for i in range(90)]
        rows.append(dict(id='rare', content='Only option B matches the surfing evidence.',
                         _fused=.001, _coverage_ids=['option:1']))
        plan = {'_coverage_requirements': [dict(id='option:0', text='indoor'),
                                           dict(id='option:1', text='surfing')]}
        self.assertEqual(sorted(rows, key=lambda c: c['_fused'], reverse=True)[-1]['id'], 'rare')
        ce_batches = []
        async def ce(query, documents, **kwargs):
            self.assertLessEqual(len(documents), 4)
            ce_batches.append(list(documents))
            return [4.5 if 'surfing evidence' in text else -.5 for text in documents]
        async def rank(prompt, *args, **kwargs):
            self.assertEqual(kwargs['stage'], 'search.listwise')
            candidates = re.findall(r'^\d+: (.*)$', prompt, re.M)
            self.assertLessEqual(len(candidates), 6)
            self.assertTrue(any('surfing evidence' in row for row in candidates))
            return from_scores([.95 if 'surfing evidence' in row else 0 for row in candidates])
        with patch.multiple(config, CASCADE_COARSE_LIMIT=10, CASCADE_FINE_LIMIT=8,
                            CASCADE_LLM_LIMIT=6, CE_BATCH_SIZE=4), \
                patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(search.llm, 'complete_json', side_effect=rank) as call:
            ranked = await search._filter_rerank(schemas.SearchRequest(user_id='u', query='weekend',
                options=['A. indoor activities', 'B. surfing'], top_k=10), plan, rows)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(len(ce_batches), 3)
        self.assertEqual(sum(map(len, ce_batches)), 10)
        self.assertEqual(plan['_cascade']['coarse']['output_count'], 10)
        self.assertIn('rare', plan['_cascade']['coarse']['selected_ids'])
        self.assertIn('rare', plan['_cascade']['fine']['selected_ids'])
        self.assertIn('rare', plan['_cascade']['listwise']['candidate_ids'])
        self.assertEqual(ranked[0]['id'], 'rare')
        self.assertEqual([c['id'] for c in ranked if c['_cascade_selected']], ['rare'])

    async def test_forget_constraints_survive_route_capacity(self):
        for i in range(7):
            await self.memory(f'The user asked to forget their address {i}.', type='rule')
        for i in range(4):
            await self.memory(f'The user enjoys surfing activity {i}.', type='preference')
        plan = {'intent': 'preference'}
        with patch.object(config, 'RECALL_PER_ROUTE', 2):
            await search._recall(self.st, schemas.SearchRequest(user_id='u', query='surfing', top_k=1), plan)
        rows = next(r['candidates'] for r in plan['_routes'] if r['channel'] == 'profile_rule')
        self.assertEqual(sum(row['type'] == 'rule' for row in rows), 7)

    async def test_unrelated_profile_has_no_lexical_route_boost(self):
        noise = await self.memory('The user collects stamps.', type='preference')
        gold = await self.memory('The user enjoys surfing.', type='preference')
        plan = {'intent': 'preference'}
        await search._recall(self.st, schemas.SearchRequest(user_id='u', query='surfing'), plan)
        rows = next(r['candidates'] for r in plan['_routes'] if r['channel'] == 'profile_rule')
        self.assertIn(gold, {m['id'] for m in rows})
        self.assertNotIn(noise, {m['id'] for m in rows})

    async def test_asserted_and_inferred_preferences_are_not_coalesced(self):
        a = await self.memory('The user enjoys tea.', type='preference', epistemic_status='asserted')
        b = await self.memory('The user enjoys tea.', type='preference', epistemic_status='inferred')
        candidates = search._prepare_candidates(self.st, schemas.SearchRequest(user_id='u', query='tea'),
                                                {}, self.st.get_amus_by_ids([a, b]))
        self.assertEqual(len(candidates), 2)

    async def test_cold_scene_can_supply_evidence_in_same_pass(self):
        gold = await self.memory('Nora teaches Japanese in Kyoto.')
        vec = (await embed(['Kyoto Japanese']))[0]
        sid = self.st.insert_scene('u', vec, ['Kyoto', 'Japanese'])
        self.st.assign_scene([gold], sid, 'cell')
        self.st.set_tier([gold], 'cold')
        self.st.register_dependencies('scene', sid, [gold])
        self.st.reset_scene_interactions(sid, tier='cold')
        plan = {'intent': 'multi_hop', '_cold': True}
        await search._recall(self.st, schemas.SearchRequest(user_id='u', query='Kyoto Japanese'), plan)
        rows = next(r['candidates'] for r in plan['_routes'] if r['channel'] == 'scene')
        self.assertIn(gold, {row['id'] for row in rows})

    async def test_mixed_scores_keep_useful_bridge_and_drop_topic_only_noise(self):
        gold = await self.memory('Maya has a sister called Nora.')
        bridge = await self.memory('Nora lives in Kyoto.')
        for i in range(12):
            await self.memory(f'Maya once discussed unrelated topic {i}.')
        result = await self.retrieve('Where does Maya sister live?',
                                     lambda text: .9 if 'Nora' in text else .05)
        self.assertEqual({x.id for x in result.data}, {gold, bridge})
        self.assertEqual(len(result.coverage_manifest['selection_omitted']), 12)

    async def test_negative_ce_logits_are_not_treated_as_probability_thresholds(self):
        for i in range(20):
            await self.memory(f'Weak possible clue about Maya {i}.')
        result = await self.retrieve('Maya', lambda _: .1)
        self.assertEqual(len(result.data), config.CASCADE_LLM_LIMIT)
        self.assertTrue(all(x.score_kind == 'cross_encoder' for x in result.data))
        self.assertTrue(all(abs(x.score - (-2.2)) < 1e-9 for x in result.data))
        self.assertEqual(result.coverage_manifest['selection_mode'], 'graph_cascade')
        self.assertEqual(result.coverage_manifest['cascade']['listwise']['status'], 'ok')

    async def test_all_irrelevant_returns_no_evidence_but_keeps_forget_constraint(self):
        await self.memory('A discussion of car engines.')
        rule = await self.memory('The user asked to forget their address.', type='rule')
        result = await self.retrieve('What music do I like?', lambda _: 0)
        self.assertEqual([x.id for x in result.data], [rule])
        self.assertEqual(result.coverage_manifest['evidence_count'], 0)

    async def test_partial_batch_failure_is_visible_in_response_and_packet(self):
        memory_ids = set()
        for i in range(6):
            memory_ids.add(await self.memory(f'The user discussed Kyoto trip detail {i}.'))
        async def ce(query, documents, **kw):
            if kw['stage'].endswith('batch_1'):
                raise ConnectionError('test provider unavailable')
            return [6.] * len(documents)
        async def rank(prompt, *args, **kw):
            self.assertEqual(kw['stage'], 'search.listwise')
            return from_scores([.6] * len(re.findall(r'^\d+:', prompt, re.M)))
        with patch.object(config, 'CE_BATCH_SIZE', 2), \
                patch.object(search, '_understand', AsyncMock(return_value={'intent': 'fact'})), \
                patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(search.llm, 'complete_json', side_effect=rank):
            result = await search.run_search(self.st, schemas.SearchRequest(user_id='u', query='Kyoto', top_k=50))
        self.assertEqual(result.coverage_manifest['rerank_status'], 'partial')
        self.assertEqual(result.coverage_manifest['selection_mode'], 'graph_cascade')
        self.assertEqual({item.id for item in result.data}, memory_ids)
        self.assertEqual(sum(x.score is None for x in result.data), 2)
        self.assertTrue(all(c.score in (None, 6.) for c in result.data))
        self.assertTrue(all(c.score_kind == 'listwise' for c in result.data if c.score is None))
        self.assertTrue(all(c.score_kind == 'cross_encoder' for c in result.data if c.score is not None))
        self.assertEqual(result.coverage_manifest['cascade']['cross_encoder']['status'], 'partial')
        self.assertEqual(result.coverage_manifest['cascade']['listwise']['status'], 'ok')
        self.assertTrue(result.coverage_manifest['rerank_errors'])
        self.assertTrue(answer_context.build([c.model_dump() for c in result.data]))

    async def test_recovered_listwise_is_distinguished_from_degraded_search(self):
        await self.memory('The user lives in Kyoto.')
        stages = []
        async def rank(prompt, *args, **kw):
            stages.append(kw['stage'])
            return from_scores([.6]) if kw['stage'].endswith('.repair') else from_scores([.1] * 4)
        async def ce(query, documents, **kw):
            return [3.7] * len(documents)
        with patch.object(search, '_understand', AsyncMock(return_value={'intent': 'fact'})), \
                patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(search.llm, 'complete_json', side_effect=rank):
            result = await search.run_search(self.st, schemas.SearchRequest(user_id='u', query='Kyoto'))
        self.assertEqual(result.coverage_manifest['rerank_status'], 'recovered')
        self.assertEqual(result.coverage_manifest['rerank_errors'], [])
        self.assertEqual(stages, ['search.listwise', 'search.listwise.repair'])
        self.assertEqual(result.data[0].score, 3.7)
        self.assertEqual(result.coverage_manifest['cascade']['listwise']['status'], 'recovered')

    async def test_unscored_tail_cannot_refill_packet_after_noise_filter(self):
        for i in range(85):
            await self.memory(f'Alice clue {i}')
        with patch.object(config, 'CASCADE_COARSE_LIMIT', 10):
            result = await self.retrieve('Alice clue', lambda _: 0, top_k=5)
        self.assertEqual(result.data, [])
        self.assertEqual(result.coverage_manifest['cascade']['coarse']['output_count'], 10)
        self.assertEqual(result.coverage_manifest['cascade']['listwise']['output_count'], 0)

    async def test_invalid_rerank_has_bounded_coherent_fallback(self):
        for i in range(15):
            await self.memory(f'Alice clue {i}')
        with patch.object(search, '_understand', AsyncMock(return_value={'intent': 'fact'})), \
                patch.object(cross_encoder, 'rerank', AsyncMock(side_effect=ConnectionError('CE unavailable'))), \
                patch.object(search.llm, 'complete_json', AsyncMock(return_value=from_scores([.9]))):
            result = await search.run_search(self.st, schemas.SearchRequest(user_id='u', query='Alice clue'))
        self.assertEqual(len(result.data), config.EVIDENCE_FALLBACK_ITEMS)
        self.assertEqual(result.coverage_manifest['selection_mode'], 'graph_cascade')
        self.assertEqual(result.coverage_manifest['cascade']['cross_encoder']['status'], 'fallback')
        self.assertEqual(result.coverage_manifest['cascade']['listwise']['status'], 'fallback')
        self.assertTrue(result.coverage_manifest['search_degraded'])
        self.assertTrue(all(c.score is None and c.score_kind == 'graph_fallback' for c in result.data))

    async def test_disputed_vector_hit_keeps_conflict_and_attribution_metadata(self):
        gold = await self.memory('Maya lives in Kyoto.', resolution_status='disputed',
                                 epistemic_status='inferred')
        result = await self.retrieve('Maya Kyoto', lambda _: .9)
        self.assertEqual(result.evidence_status, 'conflicting')
        self.assertEqual(result.coverage_manifest['conflicting_ids'], [gold])
        self.assertIn('disputed', result.data[0].content)
        self.assertIn('inferred', result.data[0].content)

    def test_redundant_episode_does_not_consume_an_evidence_slot(self):
        source = dict(request_id='r', message_index=0, role='user', content='I live in Kyoto.')
        items = [dict(id='fact', content='The user lives in Kyoto.', sources=[source]),
                 dict(id='episode', content='Conversation excerpts.', memory_type='episode', sources=[source]),
                 dict(id='next', content='The user works in Osaka.')]
        packet, _, manifest = evidence_packet.pack(items, 2, 32000)
        self.assertEqual([m['id'] for m in packet], ['fact', 'next'])
        self.assertEqual(manifest['omitted'][0]['reason'], 'redundant_episode')
        self.assertIn('I live in Kyoto.', answer_context.build(packet))

    def test_episode_with_additional_context_survives(self):
        text = 'Claire lives in Kyoto. I live in Osaka.'
        partial = dict(request_id='r', message_index=0, role='user', content=text[:22],
                       content_span=dict(start=0, end=22, original_length=len(text)))
        full = dict(request_id='r', message_index=0, role='user', content=text)
        packet, _, _ = evidence_packet.pack([
            dict(id='fact', content='Claire lives in Kyoto.', sources=[partial]),
            dict(id='episode', content='Conversation excerpts.', memory_type='episode', sources=[full])], 2, 32000)
        self.assertEqual(len(packet), 2)
        self.assertIn('I live in Osaka.', answer_context.build(packet))


if __name__ == '__main__':
    unittest.main()
