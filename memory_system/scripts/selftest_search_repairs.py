"""Offline behavior regressions for complete, degraded, and bounded recall.

Run with ``python memory_system/scripts/selftest_search_repairs.py``.
All provider behavior is deterministic; no credentials or network are used.
"""
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

os.environ['AML_FAKE'] = '1'
os.environ['AML_MEMORY_DEBUG_LOG'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from listwise_fixture import from_scores
from app import config, cross_encoder, graph, graph_fusion, retrieval_queries, schemas, search_pipeline as search, store
from app.embeddings import embed


async def score_all(prompt, *args, **kwargs):
    return from_scores([.9] * len(re.findall(r'^\d+:', prompt, re.M)))


async def score_documents(query, documents, **kwargs):
    """A stable raw CE logit isolates policy tests from fake lexical matching."""
    return [2.0] * len(documents)


class PlanningAndGraphRepairs(unittest.TestCase):
    def test_six_options_each_receive_a_retrieval_query(self):
        options = ['A. I enjoy painting landscapes.', 'B. I cycle to work.',
                   'C. I play the violin.', 'D. I cook vegetarian meals.',
                   'E. I collect vintage cameras.', 'F. I study astronomy.']
        specs = retrieval_queries.build('Which activity fits me?', options, {})
        self.assertEqual({entry['option_index'] for entry in specs
                          if entry.get('option_index') is not None}, set(range(6)))
        self.assertTrue(any(entry['text'] == 'Which activity fits me?' for entry in specs))

    def test_graph_retains_reachable_bridge_beyond_seed_neighbors(self):
        triples = [dict(subject='Alpha', relation='knows', object='Beta', amu_id='m1'),
                   dict(subject='Beta', relation='founded', object='Gamma', amu_id='m2'),
                   dict(subject='Gamma', relation='located_in', object='Delta', amu_id='m3')]
        retained = graph.filter_triples(triples, 'Alpha', ['Alpha'],
                                        limit=120, seed_amu_ids=['m1'])
        recalled = graph.ppr_recall(retained, ['Alpha'], seed_amu_ids=['m1'])
        self.assertIn('m3', recalled)

    def test_repeated_routes_in_one_family_do_not_multiply_fusion_score(self):
        route = [dict(id='bridge', content='The required bridge fact')]
        single = graph_fusion.fuse([route], {'query': 'required bridge',
            '_routes': [{'channel': 'full_text'}]})[0]['_fused']
        repeated = graph_fusion.fuse([route] * 20, {'query': 'required bridge',
            '_routes': [{'channel': 'full_text'}] * 20})[0]['_fused']
        self.assertAlmostEqual(single, repeated)

    def test_repeated_weak_channel_hits_do_not_displace_query_relevant_bridge(self):
        def low_route(prefix):
            return [dict(id=f'{prefix}{i}', content=f'Distractor {i}') for i in range(39)] + [
                dict(id='common', content='A common weak match', _score=.2)]
        routes = [[dict(id='bridge', content='HelioWorks headquarters are in Tallinn.')],
                  low_route('dense'), low_route('lexical')]
        ranked = graph_fusion.fuse(routes, {'query': 'HelioWorks Tallinn', '_routes': [
            {'channel': name} for name in ('graph', 'vector', 'full_text')]})
        positions = {item['id']: index for index, item in enumerate(ranked)}
        self.assertLess(positions['bridge'], positions['common'])


class SearchAvailabilityRepairs(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')
        self.debug = patch.object(config, 'SEARCH_DEBUG_LOG', '')
        self.debug.start()
        self.ce = patch.object(cross_encoder, 'rerank', side_effect=score_documents)
        self.ce.start()
        self.addCleanup(self.ce.stop)

    def tearDown(self):
        self.debug.stop()
        self.st.conn.close()

    async def memory(self, content, **kwargs):
        vector = (await embed([content]))[0]
        return self.st.insert_amu(user_id='u', session_id='s', content=content,
                                  embedding=vector, **kwargs)

    async def test_embedding_outage_still_returns_exact_lexical_evidence(self):
        memory_id = await self.memory('OrchidLedger was founded in Hangzhou.')
        req = schemas.SearchRequest(user_id='u', query='OrchidLedger')
        with patch.object(search, '_understand', AsyncMock(return_value={
                'intent': 'fact', 'entities': ['OrchidLedger']})), \
                patch.object(search, 'embed', AsyncMock(side_effect=ConnectionError('offline'))), \
                patch.object(search.llm, 'complete_json', side_effect=score_all):
            response = await search.run_search(self.st, req)
        self.assertIn(memory_id, {item.id for item in response.data})

    async def test_scene_recall_preserves_secondary_query_evidence(self):
        first, second = np.array([1., 0.]), np.array([0., 1.])
        scene_id = self.st.insert_scene('u', np.array([.7, .7]), ['firsttopic', 'secondtopic'])
        for index in range(4):
            self.st.insert_amu(user_id='u', session_id='s', content=f'Firsttopic detail {index}',
                               embedding=first, scene_id=scene_id)
        target = self.st.insert_amu(user_id='u', session_id='s', content='Secondtopic destination',
                                    embedding=second, scene_id=scene_id)
        self.st.register_dependencies('scene', scene_id, self.st.scene_cell_ids(scene_id))
        req = schemas.SearchRequest(user_id='u', query='Firsttopic')
        plan = {'intent': 'multi_hop', 'sub_queries': ['Secondtopic']}
        with patch.object(search, 'embed', AsyncMock(return_value=np.stack([first, second]))), \
                patch.multiple(config, RECALL_EXPANSION_LIMIT=2, SCENE_TOP_M=1,
                               SCENE_CELLS_PER_SCENE=2):
            await search._recall(self.st, req, plan)
        scene_candidates = {item['id'] for route in plan['_routes'] if route['channel'] == 'scene'
                            for item in route['candidates']}
        self.assertIn(target, scene_candidates)

    async def test_long_document_relevant_source_span_is_ranked_and_returned(self):
        fact = 'The ZephyrLaunch authorization code is KESTREL-2049.'
        original = ('Archived planning detail unrelated to the launch.\n' * 900) + fact
        memory_id = await self.memory('ZephyrLaunch project archive', type='episode')
        self.st.save_messages(schemas.AddRequest(request_id='long-source', user_id='u', session_id='s',
            messages=[schemas.Message(role='user', content=original)]))
        self.st.link_sources(memory_id, 'long-source', [0])
        prompts = []

        async def record_ranking(prompt, *args, **kwargs):
            prompts.append(prompt)
            return await score_all(prompt)

        req = schemas.SearchRequest(user_id='u', query='ZephyrLaunch authorization code',
                                     evidence_token_budget=32000)
        with patch.object(search, '_understand', AsyncMock(return_value={
                'intent': 'document', 'entities': ['ZephyrLaunch']})), \
                patch.object(search.llm, 'complete_json', side_effect=record_ranking):
            response = await search.run_search(self.st, req)
        self.assertTrue(any(fact in prompt for prompt in prompts))
        self.assertIn(memory_id, {item.id for item in response.data})
        self.assertIn(fact, '\n'.join(item.content for item in response.data))
        visible = [source for item in response.data for source in item.sources if source.get('content')]
        self.assertTrue(visible)
        for source in visible:
            span = source.get('content_span')
            self.assertIsNotNone(span)
            self.assertEqual(source['content'], original[span['start']:span['end']])

    async def test_manifest_identifies_option_evidence_present_and_missing(self):
        memory_id = await self.memory('Redwood observatory is in Hualien.')
        req = schemas.SearchRequest(user_id='u', query='Which site fits?',
            options=['A. Redwood observatory.', 'B. Cobalt laboratory.'])
        with patch.object(search, '_understand', AsyncMock(return_value={'intent': 'fact'})), \
                patch.object(search.llm, 'complete_json', side_effect=score_all):
            response = await search.run_search(self.st, req)
        coverage = response.coverage_manifest['coverage']
        requirements = {entry['id']: entry for entry in coverage['requirements']}
        self.assertEqual(requirements['option:0']['status'], 'packed')
        self.assertIn(memory_id, requirements['option:0']['candidate_ids'])
        self.assertIn(memory_id, requirements['option:0']['included_ids'])
        self.assertEqual(requirements['option:1']['status'], 'missing')
        self.assertEqual(requirements['option:1']['included_ids'], [])
        self.assertIn(requirements['option:1']['text'], response.missing_evidence)
        self.assertNotIn(requirements['option:0']['text'], response.missing_evidence)
        self.assertIn('not semantic sufficiency', coverage['basis'])

    async def test_one_grounded_followup_finds_company_discovered_in_first_hop(self):
        seed = self.st.insert_amu(user_id='u', session_id='s',
            content='My former colleague started the HelioWorks company.')
        destination = self.st.insert_amu(user_id='u', session_id='s',
            content='HelioWorks headquarters are in Tallinn.')
        stages = []

        async def provider(prompt, *args, **kwargs):
            stages.append(kwargs['stage'])
            if kwargs['stage'] == 'search.followup':
                self.assertIn(seed, prompt)
                self.assertIn('HelioWorks', prompt)
                self.assertNotIn('Tallinn', prompt)
                return {'queries': [{'text': 'HelioWorks', 'seed_ids': [seed],
                                     'coverage_ids': ['question']}]}
            return await score_all(prompt)

        req = schemas.SearchRequest(user_id='u', query="Where is my former colleague's company based?")
        with patch.object(search, '_understand', AsyncMock(return_value={'intent': 'multi_hop'})), \
                patch.object(search.llm, 'complete_json', side_effect=provider):
            response = await search.run_search(self.st, req)
        self.assertIn(destination, {item.id for item in response.data})
        self.assertIn('Tallinn', '\n'.join(item.content for item in response.data))
        self.assertEqual(stages.count('search.followup'), 1)
        self.assertEqual(response.coverage_manifest['followup']['status'], 'queried')

    async def test_followup_rejects_unobserved_entity_instead_of_searching_it(self):
        seed = self.st.insert_amu(user_id='u', session_id='s',
            content='My former colleague started the HelioWorks company.')
        unrelated = self.st.insert_amu(user_id='u', session_id='s',
            content='Atlantis headquarters are in Tallinn.')
        stages = []

        async def provider(prompt, *args, **kwargs):
            stages.append(kwargs['stage'])
            if kwargs['stage'] == 'search.followup':
                return {'queries': [{'text': 'Atlantis', 'seed_ids': [seed],
                                     'coverage_ids': ['question']}]}
            return await score_all(prompt)

        req = schemas.SearchRequest(user_id='u', query="Where is my former colleague's company based?")
        with patch.object(search, '_understand', AsyncMock(return_value={'intent': 'multi_hop'})), \
                patch.object(search.llm, 'complete_json', side_effect=provider):
            response = await search.run_search(self.st, req)
        self.assertNotIn(unrelated, {item.id for item in response.data})
        self.assertEqual(stages.count('search.followup'), 1)
        self.assertEqual(response.coverage_manifest['followup']['status'], 'no_grounded_bridge')

    async def test_one_slot_prefers_scored_answer_with_its_bridge(self):
        bridge = 'My former colleague started the HelioWorks company.'
        seed = self.st.insert_amu(user_id='u', session_id='s', content=bridge)
        destination = self.st.insert_amu(user_id='u', session_id='s',
            content='HelioWorks headquarters are in Tallinn.')

        async def provider(prompt, *args, **kwargs):
            if kwargs['stage'] == 'search.followup':
                return {'queries': [{'text': 'HelioWorks', 'seed_ids': [seed],
                                     'coverage_ids': ['question']}]}
            rows = re.findall(r'^\d+: (.*)$', prompt, re.M)
            return from_scores([.9 if 'Tallinn' in row else .3 for row in rows])

        req = schemas.SearchRequest(user_id='u', query="Where is my former colleague's company based?", top_k=1)
        with patch.object(search, '_understand', AsyncMock(return_value={'intent': 'multi_hop'})), \
                patch.object(search.llm, 'complete_json', side_effect=provider):
            response = await search.run_search(self.st, req)
        self.assertEqual([item.id for item in response.data], [destination])
        self.assertIn(bridge, response.data[0].content)
        self.assertIn('Tallinn', response.data[0].content)

    async def test_returned_unit_preserves_all_source_text_seen_by_ranker(self):
        memory_id = await self.memory('OrchidUnit launch summary.')
        messages = [schemas.Message(role='user', content='OrchidUnit launch date is May 6.\nApproval is still pending.'),
                    schemas.Message(role='assistant', content='The planned date is tentative until approval arrives.'),
                    schemas.Message(role='user', content='OrchidUnit ships only after approval; do not treat it as completed.')]
        self.st.save_messages(schemas.AddRequest(request_id='unit-source', user_id='u', session_id='s',
                                                messages=messages))
        self.st.link_sources(memory_id, 'unit-source', [0, 1, 2])
        ranked_texts, ce_texts = [], []

        async def record_rank(prompt, *args, **kwargs):
            self.assertEqual(kwargs['stage'], 'search.listwise')
            ranked_texts.extend(re.findall(r'^\d+: \[id: [^\]]+\] (.*)$', prompt, re.M))
            return await score_all(prompt)

        async def record_ce(query, documents, **kwargs):
            ce_texts.extend(documents)
            return await score_documents(query, documents, **kwargs)

        req = schemas.SearchRequest(user_id='u', query='OrchidUnit launch approval')
        with patch.object(search, '_understand', AsyncMock(return_value={'intent': 'fact'})), \
                patch.object(search.llm, 'complete_json', side_effect=record_rank), \
                patch.object(cross_encoder, 'rerank', side_effect=record_ce):
            response = await search.run_search(self.st, req)
        self.assertEqual(len(response.data), 1)
        item = response.data[0]
        self.assertEqual(ce_texts, [item.content])
        flattened = re.sub(r'\s+', ' ', item.content).strip()
        self.assertEqual(ranked_texts, [flattened])
        self.assertEqual({source['content'] for source in item.sources if source.get('content')},
                         {message.content for message in messages})

    async def test_three_memories_of_one_source_do_not_disable_expansion(self):
        ids = [await self.memory(f'OrchidGate claim {i}') for i in range(3)]
        self.st.save_messages(schemas.AddRequest(request_id='shared-source', user_id='u', session_id='s',
            messages=[schemas.Message(role='user', content='OrchidGate has three recorded claims.')]))
        for memory_id in ids:
            self.st.link_sources(memory_id, 'shared-source', [0])
        req = schemas.SearchRequest(user_id='u', query='OrchidGate')
        plan = {'intent': 'fact'}
        with patch.object(config, 'RECALL_EXPANSION_MIN_DIRECT', 3):
            await search._recall(self.st, req, plan)
        self.assertTrue(plan['_expansion']['enabled'])
        self.assertEqual(plan['_expansion']['coverage'][0]['independent_sources'], 1)

    async def test_unrelated_linked_messages_do_not_inflate_query_coverage(self):
        memory_id = await self.memory('OrchidGate opened in Hualien.')
        self.st.save_messages(schemas.AddRequest(request_id='mixed-source', user_id='u', session_id='s',
            messages=[schemas.Message(role='user', content='OrchidGate opened in Hualien.'),
                      schemas.Message(role='user', content='The blue bicycle needs a new chain.'),
                      schemas.Message(role='assistant', content='Tomorrow should be sunny.')]))
        self.st.link_sources(memory_id, 'mixed-source', [0, 1, 2])
        req = schemas.SearchRequest(user_id='u', query='OrchidGate')
        plan = {'intent': 'fact'}
        with patch.object(config, 'RECALL_EXPANSION_MIN_DIRECT', 3):
            await search._recall(self.st, req, plan)
        self.assertTrue(plan['_expansion']['enabled'])
        self.assertEqual(plan['_expansion']['coverage'][0]['independent_sources'], 1)

    async def test_malformed_optional_followup_preserves_first_hop_evidence(self):
        seed = self.st.insert_amu(user_id='u', session_id='s',
            content='My former colleague started the HelioWorks company.')

        async def provider(prompt, *args, **kwargs):
            if kwargs['stage'] == 'search.followup':
                return {'queries': None}
            return await score_all(prompt)

        req = schemas.SearchRequest(user_id='u', query="Where is my former colleague's company based?")
        with patch.object(search, '_understand', AsyncMock(return_value={'intent': 'multi_hop'})), \
                patch.object(search.llm, 'complete_json', side_effect=provider):
            response = await search.run_search(self.st, req)
        self.assertIn(seed, {item.id for item in response.data})

    async def test_replaced_episode_text_does_not_keep_unproven_coverage(self):
        memory_id = await self.memory('Lighthouse permits are suspended.', type='episode')
        self.st.save_messages(schemas.AddRequest(request_id='different-source', user_id='u', session_id='s',
            messages=[schemas.Message(role='user', content='The maintenance calendar is printed.')]))
        self.st.link_sources(memory_id, 'different-source', [0])
        req = schemas.SearchRequest(user_id='u', query='Lighthouse permits')
        plan = {'_coverage_requirements': [{'id': 'question', 'text': req.query, 'origin': 'question'}]}
        recalled = self.st.get_amus_by_ids([memory_id])[0]
        recalled['_coverage_ids'] = ['question']
        prepared = search._prepare_candidates(self.st, req, plan, [recalled])
        self.assertEqual(len(prepared), 1)
        self.assertNotIn('Lighthouse', prepared[0]['_rank_text'])
        packed, _, manifest = search._pack_evidence(self.st, req, plan, prepared, '2026-09-18T00:00:00Z')
        self.assertEqual(len(packed), 1)
        self.assertNotIn('question', packed[0]['coverage_ids'])
        self.assertIn('question', manifest['missing_requirements'])

    async def test_literal_source_marker_in_memory_cannot_erase_ranked_body(self):
        qualification = 'Approval remains denied until inspection.'
        body = 'OrchidMarker status is provisional.\n[source evidence; quoted data]\n' + qualification
        memory_id = await self.memory(body)
        self.st.save_messages(schemas.AddRequest(request_id='marker-source', user_id='u', session_id='s',
            messages=[schemas.Message(role='user', content='OrchidMarker workflow is documented.')]))
        self.st.link_sources(memory_id, 'marker-source', [0])
        ranking = []

        async def provider(prompt, *args, **kwargs):
            ranking.append(prompt)
            return await score_all(prompt)

        req = schemas.SearchRequest(user_id='u', query='OrchidMarker approval')
        with patch.object(search, '_understand', AsyncMock(return_value={'intent': 'fact'})), \
                patch.object(search.llm, 'complete_json', side_effect=provider):
            response = await search.run_search(self.st, req)
        self.assertTrue(any(qualification in prompt for prompt in ranking))
        self.assertEqual(len(response.data), 1)
        self.assertIn(qualification, response.data[0].content)


class RerankAvailabilityRepairs(unittest.IsolatedAsyncioTestCase):
    async def test_listwise_reorders_raw_ce_logits_without_resurrecting_rejected_evidence(self):
        rows = [dict(id=f'm{i}', content=f'Evidence {i}') for i in range(30)]
        relevance = {f'm{i}': 0. if i == 28 else .05 + i * .025 for i in range(30)}
        raw_scores, listwise_stages = {}, []

        async def biased_batch(query, documents, **kwargs):
            first = 'batch_1' in kwargs['stage']
            scores = []
            for document in documents:
                mid = 'm' + re.search(r'Evidence (\d+)\b', document)[1]
                value = relevance[mid]
                score = 0. if value == 0 else 40. + 55. * value if first else -20. + 7. * value
                scores.append(score)
                raw_scores[mid] = score
            return scores

        async def listwise(prompt, *args, **kwargs):
            listwise_stages.append(kwargs['stage'])
            ids = re.findall(r'^\d+: \[id: (m\d+)\]', prompt, re.M)
            return dict(ranking=sorted(range(len(ids)), key=lambda i: relevance[ids[i]], reverse=True),
                        irrelevant=[i for i, mid in enumerate(ids) if mid == 'm28'], groups=[])

        req = schemas.SearchRequest(user_id='u', query='Evidence', top_k=30)
        plan = {}
        with patch.multiple(config, CE_BATCH_SIZE=8, CASCADE_COARSE_LIMIT=30,
                            CASCADE_FINE_LIMIT=30, CASCADE_LLM_LIMIT=30), \
                patch.object(cross_encoder, 'rerank', side_effect=biased_batch), \
                patch.object(search.llm, 'complete_json', side_effect=listwise):
            ranked = await search._filter_rerank(req, plan, rows)
        self.assertEqual([item['id'] for item in ranked],
                         sorted(relevance, key=relevance.get, reverse=True))
        self.assertEqual(next(item['_final'] for item in ranked if item['id'] == 'm28'), 0.)
        self.assertNotIn('m28', {item['id'] for item in search._select_evidence(req, plan, ranked)})
        self.assertEqual({item['id']: item['_final'] for item in ranked}, raw_scores)
        self.assertEqual(listwise_stages, ['search.listwise'])
        self.assertEqual(sum(b['candidate_count'] for b in plan['_cascade']['cross_encoder']['batches']), 30)

    async def test_failed_batch_keeps_unknown_evidence_when_other_batch_is_zero(self):
        rows = [dict(id=f'm{i}', content=f'Evidence {i}') for i in range(25)]
        failed_ids, rejected_ids = set(), set()

        async def batch_result(query, documents, **kwargs):
            ids = {'m' + re.search(r'Evidence (\d+)\b', text)[1] for text in documents}
            if 'batch_1' in kwargs['stage']:
                failed_ids.update(ids)
                raise ConnectionError('offline')
            rejected_ids.update(ids)
            return [0.] * len(documents)

        async def listwise(prompt, *args, **kwargs):
            self.assertEqual(kwargs['stage'], 'search.listwise')
            ids = re.findall(r'^\d+: \[id: (m\d+)\]', prompt, re.M)
            return dict(ranking=list(range(len(ids))),
                        irrelevant=[i for i, mid in enumerate(ids) if mid in rejected_ids], groups=[])

        req = schemas.SearchRequest(user_id='u', query='Evidence', top_k=25)
        plan = {}
        with patch.object(config, 'CE_BATCH_SIZE', 24), \
                patch.object(cross_encoder, 'rerank', side_effect=batch_result), \
                patch.object(search.llm, 'complete_json', side_effect=listwise):
            ranked = await search._filter_rerank(req, plan, rows)
        self.assertTrue(failed_ids)
        self.assertTrue(rejected_ids)
        selected = search._select_evidence(req, plan, ranked)
        self.assertTrue(failed_ids & {item['id'] for item in selected})
        self.assertFalse(rejected_ids & {item['id'] for item in selected})
        self.assertTrue(all(item.get('_final') is None for item in selected))
        self.assertTrue(all(item['_score_kind'] == 'listwise' for item in selected))
        self.assertEqual(plan['_cascade']['cross_encoder']['status'], 'partial')
        self.assertEqual(plan['_cascade']['listwise']['status'], 'ok')


if __name__ == '__main__':
    unittest.main(verbosity=2)
