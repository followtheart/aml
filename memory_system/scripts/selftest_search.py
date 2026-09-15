"""Offline regressions for historical recall, summary evidence and search tracing."""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ['AML_FAKE'] = '1'
os.environ['AML_MEMORY_DEBUG_LOG'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, store, schemas, search_pipeline as search, eval_scoring
from app.embeddings import embed


async def score_all(prompt, *args, **kwargs):
    ids = re.findall(r'^(amu_[^:]+|summary_[^:]+):', prompt, re.M)
    return {'scores': [{'id': aid, 'relevance': 0.9, 'keep': True} for aid in ids]}


class SearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'search.jsonl'
        self.setting = patch.object(config, 'SEARCH_DEBUG_LOG', str(self.path))
        self.setting.start()

    def tearDown(self):
        self.st.conn.close()
        self.setting.stop()
        self.tmp.cleanup()

    async def memory(self, content, **kwargs):
        vec = (await embed([content]))[0]
        return self.st.insert_amu(user_id='u', session_id='s', content=content,
                                  embedding=vec, **kwargs)

    def req(self, query='Alice work', **kwargs):
        return schemas.SearchRequest(user_id='u', query=query, **kwargs)

    async def run_search(self, req, plan=None):
        with patch.object(search, '_understand', AsyncMock(return_value=plan or {
                'intent': 'fact', 'entities': ['Alice']})), patch.object(
                search.llm, 'complete_json', side_effect=score_all):
            return await search.run_search(self.st, req)

    async def test_history_across_vector_fts_graph_and_current_exclusion(self):
        old = await self.memory('Alice work Shanghai', valid_from='2020-01-01', valid_to='2024-01-01')
        current = await self.memory('Alice work Hangzhou', valid_from='2024-01-01')
        self.st.insert_triple('u', 'Alice', 'works_in', 'Shanghai', old)
        self.st.insert_triple('u', 'Alice', 'works_in', 'Hangzhou', current)
        response = await self.run_search(self.req('Alice work previously'))
        self.assertEqual({x.id for x in response.data}, {old, current})
        trace = json.loads(self.path.read_text().splitlines()[-1])
        self.assertTrue(trace['include_history'])
        for channel in ['vector', 'full_text', 'graph']:
            ids = {c['id'] for r in trace['routes'] if r['channel'] == channel for c in r['candidates']}
            self.assertIn(old, ids)
        response = await self.run_search(self.req(include_history=False))
        self.assertEqual([x.id for x in response.data], [current])
        vec = (await embed(['Alice work']))[0]
        self.assertNotIn(old, {x['id'] for x in self.st.nearest_by_embedding('u', vec, 100)})
        self.assertIn(old, {x['id'] for x in self.st.get_by_type('u', ['fact'], include_history=True)})

    async def test_summary_only_recall_and_sources_reach_answer(self):
        req = schemas.AddRequest(request_id='r', user_id='u', session_id='s',
            messages=[schemas.Message(role='user', content='Original evidence about Alice work', timestamp=123)])
        self.st.save_messages(req)
        self.st.set_summary('u', 's', 'Alice work is based in Hangzhou')
        # ULM §3.1: the rolling summary is extraction context, not evidence, by default.
        result = await self.run_search(self.req())
        self.assertEqual(result.data, [])
        with patch.object(config, 'SUMMARY_ROUTE', True):
            result = await self.run_search(self.req())
        self.assertEqual(len(result.data), 1)
        self.assertEqual(result.data[0].memory_type, 'session_summary')
        self.assertEqual(result.data[0].sources[0]['timestamp'], 123)
        self.assertEqual(result.data[0].sources[0]['content_omitted'], 'memory_only')
        prompt = eval_scoring.answer_prompt({'question': 'Alice work?'},
                                            [x.model_dump() for x in result.data])
        self.assertNotIn('Original evidence about Alice work', prompt)
        self.assertIn('Hangzhou', prompt)
        self.st.set_summary('u', 's', 'Bob hobbies')
        self.assertEqual(self.st.summary_search('u', 'Hangzhou', 10), [])
        self.assertEqual(len(self.st.summary_search('u', 'Bob', 10)), 1)
        self.assertEqual(self.st.summary_search('another', 'Bob', 10), [])

    async def test_top_100_with_small_r_rerank_head(self):
        for i in range(105):
            await self.memory(f'Alice work record {i}')
        result = await self.run_search(self.req(top_k=100))
        self.assertEqual(len(result.data), 100)
        trace = json.loads(self.path.read_text().splitlines()[-1])
        scored = [x for x in trace['rerank'] if x['reason'] in ('kept', 'rerank_rejected')]
        unscored = [x for x in trace['rerank'] if x['reason'] == 'unscored_fused']
        # §5.4: only the fused head is LLM-scored; the rest keeps fusion order below it.
        self.assertEqual(len(scored), config.RERANK_MAX_CANDIDATES)
        self.assertGreater(len(unscored), 0)
        self.assertEqual(len(trace['returned']), 100)
        returned_ids = [x['id'] for x in trace['returned']]
        self.assertEqual(returned_ids[:len(scored)], [x['id'] for x in scored])
        self.assertEqual(trace['rounds'][0]['verdict']['sufficient'], True)

    async def test_rerank_rejection_and_topk_logged_separately(self):
        ids = [await self.memory(f'Alice work {i}') for i in range(3)]
        async def reject_one(prompt, *args, **kwargs):
            result = await score_all(prompt)
            for entry in result['scores']:
                if entry['id'] == ids[0]:
                    entry.update(keep=False, relevance=0.1)
            return result
        with patch.object(search, '_understand', AsyncMock(return_value={'intent': 'fact'})), \
             patch.object(search.llm, 'complete_json', side_effect=reject_one):
            response = await search.run_search(self.st, self.req(top_k=1))
        self.assertEqual(len(response.data), 1)
        trace = json.loads(self.path.read_text())
        rejected = [d for d in trace['rerank'] if not d['keep']]
        self.assertEqual(rejected[0]['id'], ids[0])
        self.assertEqual(rejected[0]['reason'], 'rerank_rejected')
        self.assertEqual(len(trace['top_k_excluded']), 1)
        self.assertTrue(all('content' in c for r in trace['routes'] for c in r['candidates']))

    async def test_rerank_error_retains_candidates_and_logs_fallback(self):
        aid = await self.memory('Alice work')
        with patch.object(search, '_understand', AsyncMock(return_value={'intent': 'fact'})), \
             patch.object(search.llm, 'complete_json', AsyncMock(side_effect=RuntimeError('offline'))):
            result = await search.run_search(self.st, self.req())
        self.assertEqual(result.data[0].id, aid)
        trace = json.loads(self.path.read_text())
        self.assertEqual(trace['rerank'][0]['reason'], 'global_rrf_fallback')

    async def test_low_score_abstains_even_if_model_keep_is_true(self):
        await self.memory('weak candidate')
        async def weak(prompt, *args, **kwargs):
            ids = re.findall(r'^(amu_[^:]+):', prompt, re.M)
            return {'scores': [{'id': aid, 'relevance': 0.2, 'keep': True}
                               for aid in ids]}
        with patch.object(search, '_understand', AsyncMock(return_value={
                'intent': 'fact'})), patch.object(
                search.llm, 'complete_json', side_effect=weak):
            result = await search.run_search(self.st, self.req('weak'))
        self.assertEqual(result.data, [])

    async def test_incomplete_batch_falls_back_entire_ranking(self):
        ids = [await self.memory(f'Alice work {i}') for i in range(2)]
        async def incomplete(prompt, *args, **kwargs):
            found = re.findall(r'^(amu_[^:]+):', prompt, re.M)
            return {'scores': [{'id': found[0], 'relevance': 0.1, 'keep': True}]}
        with patch.object(search, '_understand', AsyncMock(return_value={
                'intent': 'fact'})), patch.object(
                search.llm, 'complete_json', side_effect=incomplete):
            result = await search.run_search(self.st, self.req())
        self.assertEqual(len(result.data), 2)
        trace = json.loads(self.path.read_text().splitlines()[-1])
        self.assertTrue(all(item['reason'] == 'global_rrf_fallback'
                            for item in trace['rerank']))

    async def test_amu_sources_and_user_isolation(self):
        aid = await self.memory('Alice work')
        req = schemas.AddRequest(request_id='r', user_id='u', session_id='s',
            messages=[schemas.Message(role='user', content='Evidence text')])
        self.st.save_messages(req)
        self.st.link_sources(aid, 'r', [0])
        result = await self.run_search(self.req())
        self.assertNotIn('Evidence text', result.data[0].content)
        self.assertEqual(result.data[0].sources[0]['request_id'], 'r')
        self.assertEqual(result.data[0].sources[0]['content_omitted'], 'memory_only')
        self.assertNotIn('request_id', result.data[0].content)
        result = await self.run_search(schemas.SearchRequest(user_id='other', query='Alice work'))
        self.assertEqual(result.data, [])

    async def test_error_log_and_disabled_log(self):
        with patch.object(search, '_understand', AsyncMock(side_effect=ValueError('bad'))):
            with self.assertRaises(ValueError):
                await search.run_search(self.st, self.req())
        self.assertEqual(json.loads(self.path.read_text())['status'], 'error')
        before = self.path.read_text()
        with patch.object(config, 'SEARCH_DEBUG_LOG', ''):
            await self.run_search(self.req())
        self.assertEqual(self.path.read_text(), before)

    async def test_summary_index_rollback(self):
        self.st.set_summary('u', 's', 'Alice work')
        with self.assertRaises(RuntimeError):
            with self.st.staged('u') as work:
                work.set_summary('u', 's', 'Bob hobbies')
                raise RuntimeError('abort')
        self.assertTrue(self.st.summary_search('u', 'Alice', 10))
        self.assertFalse(self.st.summary_search('u', 'Bob', 10))

    async def test_cjk_term_and_summary_recall(self):
        aid = await self.memory('我目前在杭州工作')
        self.st.set_summary('u', 's', '用户现在居住在杭州')
        self.assertEqual(self.st.fts_search('u', '杭州', 10)[0]['id'], aid)
        self.assertEqual(len(self.st.summary_search('u', '杭州', 10)), 1)

    async def test_temporal_route_and_reference_time(self):
        aid = await self.memory('Alice worked in Shanghai',
                                event_time='2023-05-01T00:00:00Z',
                                valid_from='2023-01-01T00:00:00Z',
                                valid_to='2024-01-01T00:00:00Z')
        plan = {'intent': 'temporal', 'entities': [], 'time_scope': {
            'from': '2023-01-01T00:00:00Z', 'to': '2023-12-31T23:59:59Z'}}
        await self.run_search(self.req('Where did Alice work?',
                                       reference_time='2024-06-01T00:00:00Z'), plan)
        trace = json.loads(self.path.read_text().splitlines()[-1])
        temporal_ids = {c['id'] for route in trace['routes']
                        if route['channel'] == 'temporal' for c in route['candidates']}
        self.assertIn(aid, temporal_ids)
        self.assertEqual(trace['reference_time'], '2024-06-01T00:00:00Z')

    async def test_query_understanding_uses_supplied_reference_time(self):
        complete = AsyncMock(return_value={'intent': 'fact', 'include_history': False,
            'time_scope': None, 'entities': [], 'sub_queries': ['q'],
            'expanded_queries': ['q']})
        with patch.object(search.llm, 'complete_json', complete):
            await search._understand(self.req('q', reference_time='2020-02-03T04:05:06Z'))
        self.assertIn('2020-02-03T04:05:06Z', complete.call_args.args[0])

    async def test_source_context_is_deduplicated_and_bounded(self):
        first = await self.memory('Alice work one')
        second = await self.memory('Alice work two')
        req = schemas.AddRequest(request_id='source', user_id='u', session_id='s',
            messages=[schemas.Message(role='user', content='shared source evidence')])
        self.st.save_messages(req)
        self.st.link_sources(first, 'source', [0])
        self.st.link_sources(second, 'source', [0])
        result = await self.run_search(self.req(top_k=2))
        bodies = '\n'.join(item.content for item in result.data)
        self.assertEqual(bodies.count('shared source evidence'), 0)
        self.assertNotIn('content_omitted', bodies)
        self.assertTrue(all(source.get('content_omitted') == 'memory_only'
                            for item in result.data for source in item.sources))

    async def test_memory_doc_prefers_episode_and_includes_sources(self):
        fact = await self.memory('Alice work fact')
        episode = await self.memory('Alice described her work journey in detail', type='episode')
        req = schemas.AddRequest(request_id='doc-source', user_id='u', session_id='s',
            messages=[schemas.Message(role='user', content='Full narrative source')])
        self.st.save_messages(req)
        self.st.link_sources(episode, 'doc-source', [0])
        result = await self.run_search(self.req('Tell the story'),
                                       {'intent': 'narrative', 'entities': []})
        self.assertEqual([item.id for item in result.data], [episode])
        self.assertNotIn(fact, [item.id for item in result.data])
        self.assertIn('Full narrative source', result.data[0].content)

    async def test_explicit_time_scope_folds_versions(self):
        old = await self.memory('Alice works in Shanghai', valid_from='2020-01-01',
                                valid_to='2024-01-01')
        current = await self.memory('Alice works in Hangzhou', valid_from='2024-01-01',
                                    supersedes=old)
        plan = {'intent': 'temporal', 'include_history': True, 'entities': [],
                'time_scope': {'from': '2022-01-01', 'to': '2022-12-31'}}
        result = await self.run_search(self.req('Where did Alice work in 2022?'), plan)
        self.assertEqual([item.id for item in result.data], [old])
        self.assertNotIn(current, [item.id for item in result.data])

    async def test_expired_transient_profile_is_excluded(self):
        aid = await self.memory('Alice temporarily likes tea', type='preference')
        self.st.conn.execute("UPDATE amu SET expires_at='2020-01-01T00:00:00Z' WHERE id=?", (aid,))
        self.st.conn.commit()
        result = await self.run_search(self.req('What does Alice like?',
                                                reference_time='2024-01-01T00:00:00Z'),
                                       {'intent': 'preference', 'entities': []})
        self.assertEqual(result.data, [])

    async def test_sensitive_memory_requires_server_and_request_opt_in(self):
        aid = await self.memory('Alice bank account 1234', sensitivity='sensitive')
        self.assertEqual((await self.run_search(self.req('Alice bank account'))).data, [])
        with patch.object(config, 'SENSITIVE_RECALL_ENABLED', True):
            result = await self.run_search(self.req(
                'Alice bank account', include_sensitive=True))
        self.assertEqual([item.id for item in result.data], [aid])


if __name__ == '__main__':
    unittest.main(verbosity=2)
