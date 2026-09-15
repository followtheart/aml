"""Offline regression tests for atomic memory writes and evidence integrity."""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import add_pipeline as add, config, llm, schemas, store
from app.embeddings import embed
from app import embeddings
import numpy as np


def request(rid='r1', text='Alice lives in Paris.'):
    return schemas.AddRequest(request_id=rid, user_id='u', session_id='s',
        messages=[schemas.Message(role='user', content=text, timestamp=123)])


class IntegrityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')
        self.episodes = patch.object(config, 'STORE_EPISODES', False)
        self.episodes.start()

    def tearDown(self):
        self.episodes.stop()
        self.st.conn.close()

    def snapshot(self):
        return list(self.st.conn.iterdump())

    async def test_sources_graph_and_replay(self):
        req = request()
        req.messages.append(schemas.Message(role='assistant', content='Bob lives in London.'))
        await add.run_add(self.st, req)
        amus = self.st.get_amus('u')
        self.assertEqual(len(amus), 2)
        for amu in amus:
            sources = self.st.sources_for_amu(amu['id'])
            self.assertEqual(len(sources), 1)
            self.assertIn(sources[0]['content'], amu['content'])
            for triple in self.st.triples_for_user('u'):
                if triple['amu_id'] == amu['id']:
                    self.assertIn(triple['object'], amu['content'])
        before = self.snapshot()
        await add.run_add(self.st, req)
        self.assertEqual(before, self.snapshot())

    async def test_update_refreshes_all_derived_data(self):
        await add.run_add(self.st, request())
        old = self.st.get_amus('u')[0]
        decision = AsyncMock(return_value=('UPDATE', {
            'target_id': old['id'], 'merged_content': 'Alice lives in Paris, near Berlin.'}))
        with patch.object(add, '_govern_one', decision):
            await add.run_add(self.st, request('r2', 'Alice lives near Berlin.'))
        new = self.st.get_amus('u')[0]
        self.assertEqual(new['id'], old['id'])
        self.assertEqual(new['content'], 'Alice lives in Paris, near Berlin.')
        self.assertNotEqual(new['retrieval_key'], old['retrieval_key'])
        # ULM §3.5: complementary UPDATE keeps the union of both memories' metadata.
        self.assertIn('Paris', new['entities'])
        self.assertIn('Berlin', new['entities'])
        expected = (await embed([add._embed_text(new)]))[0]
        np.testing.assert_allclose(new['embedding'], expected)
        self.assertEqual(self.st.fts_search('u', 'Berlin', 10)[0]['id'], old['id'])
        objects = {t['object'] for t in self.st.triples_for_user('u')}
        self.assertTrue({'Paris', 'Berlin'} <= objects)
        self.assertEqual(len(self.st.sources_for_amu(old['id'])), 2)
        self.assertEqual(len(new['support_sessions']), 1)

    async def test_update_does_not_re_extract(self):
        await add.run_add(self.st, request())
        old = self.st.get_amus('u')[0]
        decision = AsyncMock(return_value=('UPDATE', {
            'target_id': old['id'], 'merged_content': 'Alice lives elsewhere.'}))
        original_extract = add._extract
        calls = 0
        async def counting(st, req):
            nonlocal calls
            calls += 1
            return await original_extract(st, req)
        with patch.object(add, '_govern_one', decision), patch.object(
                add, '_extract', side_effect=counting):
            await add.run_add(self.st, request('r2', 'Alice moved.'))
        self.assertEqual(calls, 1)
        self.assertEqual(self.st.get_amus('u')[0]['content'], 'Alice lives elsewhere.')

    async def test_same_entity_is_in_governance_candidates(self):
        vec = (await embed(['unrelated words']))[0]
        self.st.insert_amu(user_id='u', session_id='s', content='Alice old state',
                           entities=['Alice'], embedding=vec)
        fact = {'content': 'Alice new state', 'entities': ['Alice']}
        with patch.object(llm, 'complete_json', AsyncMock(return_value={
                'operation': 'NOOP', 'target_id': None,
                'merged_content': None, 'reason': 'test'})) as complete:
            await add._govern_one(self.st, 'u', fact, -vec)
        self.assertIn('Alice old state', complete.call_args.args[0])

    async def test_real_embedding_failure_never_falls_back_to_hash(self):
        import litellm
        with patch.object(config, 'FAKE', False), patch.object(
                litellm, 'aembedding', AsyncMock(side_effect=RuntimeError('network'))):
            with self.assertRaises(RuntimeError):
                await embeddings.embed(['must not hash'], stage='test.embedding')

    async def test_vector_search_isolated_by_embedding_space(self):
        vec = (await embed(['Alice']))[0]
        current = self.st.insert_amu(user_id='u', session_id='s', content='current',
                                     embedding=vec)
        legacy = self.st.insert_amu(user_id='u', session_id='s', content='legacy',
                                    embedding=vec)
        self.st.conn.execute('UPDATE amu SET embedding_space=NULL WHERE id=?', (legacy,))
        self.st.conn.commit()
        found = self.st.nearest_by_embedding('u', vec, 10)
        self.assertEqual([item['id'] for item in found], [current])

    async def test_failure_rolls_back_and_retry_succeeds(self):
        before = self.snapshot()
        with patch.object(add, '_update_summary', AsyncMock(side_effect=RuntimeError('failed'))):
            with self.assertRaises(RuntimeError):
                await add.run_add(self.st, request())
        self.assertEqual(before, self.snapshot())
        await add.run_add(self.st, request())
        self.assertTrue(self.st.request_seen('r1'))

    async def test_supersede_failure_preserves_old_memory(self):
        await add.run_add(self.st, request())
        old = self.st.get_amus('u')[0]
        before = self.snapshot()
        decision = AsyncMock(return_value=('SUPERSEDE', {'target_id': old['id']}))
        with patch.object(add, '_govern_one', decision), patch.object(
                add, '_update_summary', AsyncMock(side_effect=RuntimeError())):
            with self.assertRaises(RuntimeError):
                await add.run_add(self.st, request('r2', 'Alice lives in Berlin.'))
        self.assertEqual(before, self.snapshot())

    async def test_uncommitted_work_invisible(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original = add._update_summary
        async def pause(work, req):
            entered.set()
            await release.wait()
            await original(work, req)
        with patch.object(add, '_update_summary', pause):
            task = asyncio.create_task(add.run_add(self.st, request()))
            await entered.wait()
            self.assertEqual(self.st.get_amus('u'), [])
            self.assertFalse(self.st.request_seen('r1'))
            release.set()
            await task
        self.assertTrue(self.st.get_amus('u'))

    async def test_same_user_adds_are_serialized_in_process(self):
        first = request('parallel-1', 'Alice lives in Paris.')
        second = request('parallel-2', 'Bob lives in London.')
        await asyncio.gather(add.run_add(self.st, first), add.run_add(self.st, second))
        self.assertTrue(self.st.request_seen(first.request_id))
        self.assertTrue(self.st.request_seen(second.request_id))
        self.assertEqual(len(self.st.get_amus('u')), 2)

    async def test_cancellation_discards_work(self):
        entered = asyncio.Event()
        async def pause(work, req):
            entered.set()
            await asyncio.Event().wait()
        before = self.snapshot()
        with patch.object(add, '_update_summary', pause):
            task = asyncio.create_task(add.run_add(self.st, request()))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(before, self.snapshot())

    async def test_cross_user_governance_rejected(self):
        other = request('other')
        other.user_id = 'other'
        await add.run_add(self.st, other)
        target = self.st.get_amus('other')[0]['id']
        before = self.snapshot()
        with patch.object(add, '_govern_one', AsyncMock(return_value=(
                'UPDATE', {'target_id': target}))):
            with self.assertRaises(ValueError):
                await add.run_add(self.st, request())
        self.assertEqual(before, self.snapshot())

    async def test_noop_retains_new_evidence_without_new_graph(self):
        await add.run_add(self.st, request())
        target = self.st.get_amus('u')[0]['id']
        graph = self.st.triples_for_user('u')
        with patch.object(add, '_govern_one', AsyncMock(return_value=(
                'NOOP', {'target_id': target}))):
            await add.run_add(self.st, request('r2'))
        self.assertEqual(len(self.st.get_amus('u')), 1)
        self.assertEqual(len(self.st.sources_for_amu(target)), 2)
        self.assertEqual(graph, self.st.triples_for_user('u'))

    async def test_nested_triples_stay_with_their_fact(self):
        data = {'facts': [{'content': 'Alice lives in Paris.',
                          'evidence': [{'message_index': 0, 'quote': 'Alice lives in Paris.'}],
                          'triples': [{'subject': 'Alice', 'relation': 'lives_in', 'object': 'Paris'}]}],
                'triples': [{'subject': 'bad', 'relation': 'bad', 'object': 'bad', 'fact_index': 0}]}
        with patch.object(llm, 'complete_json', AsyncMock(return_value=data)):
            extracted = await add._extract(self.st, request())
        self.assertEqual(len(extracted['facts']), 1)
        self.assertNotIn('triples', extracted)
        self.assertEqual(extracted['facts'][0]['triples'][0]['object'], 'Paris')

    async def test_batch_indices_are_request_relative(self):
        req = request()
        req.messages.append(schemas.Message(role='user', content='Bob lives in London.'))
        with patch.object(config, 'EXTRACT_BATCH_MESSAGES', 1):
            await add.run_add(self.st, req)
        for amu in self.st.get_amus('u'):
            source = self.st.sources_for_amu(amu['id'])[0]
            self.assertIn(source['content'], amu['content'])
            self.assertEqual(source['message_index'], 0 if 'Alice' in amu['content'] else 1)
        self.assertEqual({t['amu_id'] for t in self.st.triples_for_user('u')},
                         {a['id'] for a in self.st.get_amus('u')})

    async def test_real_summary_error_is_not_swallowed(self):
        before = self.snapshot()
        original = llm.complete
        async def fail_summary(*args, **kwargs):
            if kwargs.get('stage') == 'add.summary':
                raise RuntimeError('provider unavailable')
            return await original(*args, **kwargs)
        with patch.object(llm, 'complete', fail_summary):
            with self.assertRaises(RuntimeError):
                await add.run_add(self.st, request())
        self.assertEqual(before, self.snapshot())

    def test_old_schema_gains_source_tables_without_data_loss(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'old.db')
            conn = sqlite3.connect(path)
            legacy_schema = store.SCHEMA
            for column in ("embedding_space", "temporal", "state", "evidence"):
                legacy_schema = legacy_schema.replace(f"  {column} TEXT,\n", "")
            conn.executescript(legacy_schema)
            conn.executescript(store.FTS_SCHEMA)
            conn.execute("INSERT INTO sessions VALUES ('u','s','old summary')")
            conn.commit()
            conn.close()
            upgraded = store.Store(path)
            try:
                columns = {row['name'] for row in upgraded.conn.execute(
                    'PRAGMA table_info(amu)')}
                self.assertTrue({"embedding_space", "temporal", "state", "evidence"} <= columns)
                self.assertEqual(upgraded.get_summary('u', 's'), 'old summary')
                upgraded.save_messages(request())
                self.assertEqual(upgraded.conn.execute(
                    'SELECT content FROM source_messages').fetchone()[0], 'Alice lives in Paris.')
            finally:
                upgraded.conn.close()

    def test_publish_failure_rolls_back_fts_and_rows(self):
        before = self.snapshot()
        with self.assertRaises(Exception):
            with self.st.staged('u') as work:
                work.insert_amu(user_id='u', session_id='s', content='Alice')
                work._pending.append(('INSERT INTO nonexistent VALUES (?)', (1,)))
        self.assertEqual(before, self.snapshot())

    def test_conflicting_connection_and_existing_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'memory.db')
            first, second = store.Store(path), store.Store(path)
            try:
                with self.assertRaisesRegex(RuntimeError, 'retry'):
                    with first.staged('u') as work:
                        work.insert_amu(user_id='u', session_id='s', content='staged')
                        second.insert_amu(user_id='u', session_id='s', content='concurrent')
                self.assertEqual([a['content'] for a in first.get_amus('u')], ['concurrent'])
            finally:
                first.conn.close()
                second.conn.close()

    def test_different_users_do_not_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'memory.db')
            first, second = store.Store(path), store.Store(path)
            try:
                with first.staged('u1') as work:
                    work.insert_amu(user_id='u1', session_id='s', content='first')
                    second.insert_amu(user_id='u2', session_id='s', content='second')
                self.assertEqual(first.get_amus('u1')[0]['content'], 'first')
                self.assertEqual(first.get_amus('u2')[0]['content'], 'second')
            finally:
                first.conn.close()
                second.conn.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
