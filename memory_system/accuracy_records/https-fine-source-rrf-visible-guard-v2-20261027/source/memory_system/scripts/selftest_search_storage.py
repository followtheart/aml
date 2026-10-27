"""Offline acceptance tests for search snapshots and vector index safety.

These tests exercise public storage behavior, not the cache representation.
They use temporary databases and do not load deployment dotenv files.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_read_text = Path.read_text
with patch.object(Path, 'read_text', lambda path, *a, **kw:
                  '' if path.name == '.env' else _read_text(path, *a, **kw)):
    from app import budget, config, local_work, schemas, search_pipeline, store, vector_index


class SearchStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='aml-search-storage-')
        self.path = str(Path(self.directory.name) / 'memory.sqlite3')
        self.st = store.Store(self.path)
        self.writer = store.Store(self.path)

    def tearDown(self):
        self.writer.conn.close()
        self.st.conn.close()
        self.directory.cleanup()

    def insert(self, content='Alpha', user_id='u', vector=None, **kwargs):
        return self.st.insert_amu(
            user_id=user_id, session_id='session-' + user_id, content=content,
            embedding=np.asarray(vector, dtype=np.float32) if vector is not None else None,
            **kwargs)

    def query_ids(self, snapshot, vector=(1, 0, 0), k=20, **kwargs):
        return [row['id'] for row in snapshot.nearest_many_by_embedding(
            'u', np.asarray([vector], dtype=np.float32), k, **kwargs)[0]]

    def test_file_snapshot_does_not_copy_user_tables(self):
        aid = self.insert()
        with patch.object(store.Store, '_copy_user_to', side_effect=AssertionError(
                'ordinary file search must not materialize a full user copy')):
            with self.st.snapshot('u') as snapshot:
                self.assertEqual([row['id'] for row in snapshot.get_amus('u')], [aid])

    def test_snapshot_is_consistent_while_another_connection_commits(self):
        aid = self.insert('Before')
        revision = self.st.user_state('u')['revision']
        with self.st.snapshot('u') as snapshot:
            self.assertEqual(snapshot.read_revision, revision)
            self.assertEqual(snapshot.get_amus_by_ids([aid])[0]['content'], 'Before')
            self.writer.update_amu_content(aid, 'After')
            self.assertEqual(snapshot.get_amus_by_ids([aid])[0]['content'], 'Before')
            self.assertEqual(snapshot.read_revision, revision)
        with self.st.snapshot('u') as snapshot:
            self.assertGreater(snapshot.read_revision, revision)
            self.assertEqual(snapshot.get_amus_by_ids([aid])[0]['content'], 'After')

    def test_snapshot_is_read_only(self):
        self.insert()
        with self.st.snapshot('u') as snapshot:
            with self.assertRaises(Exception) as caught:
                snapshot.conn.execute("UPDATE amu SET content='unwanted'")
            self.assertIn('readonly', str(caught.exception).lower().replace('-', ''))
        self.assertEqual(self.st.get_amus('u')[0]['content'], 'Alpha')

    def test_snapshot_id_sources_scene_and_dependencies_are_user_scoped(self):
        fixtures = {}
        for user in ('u', 'v'):
            source = self.insert('Private ' + user, user_id=user)
            self.st.save_messages(schemas.AddRequest(
                request_id='request-' + user, user_id=user, session_id='session-' + user,
                messages=[schemas.Message(role='user', content='Source ' + user)]))
            self.st.link_sources(source, 'request-' + user, [0])
            derived = self.insert('Derived ' + user, user_id=user)
            self.st.register_dependencies('amu', derived, [source])
            scene = self.st.insert_scene(user, np.ones(3, dtype=np.float32), ['private'])
            self.st.assign_scene([source], scene, 'cell-' + user)
            fixtures[user] = (source, derived, scene)
        own, own_derived, own_scene = fixtures['u']
        foreign, foreign_derived, foreign_scene = fixtures['v']
        with self.st.snapshot('u') as snapshot:
            self.assertEqual({row['id'] for row in snapshot.get_amus_by_ids(
                [own, foreign], include_history=True, include_sensitive=True)}, {own})
            self.assertEqual(len(snapshot.sources_for_amu(own)), 1)
            self.assertEqual(snapshot.sources_for_amu(foreign), [])
            self.assertEqual({row['id'] for row in snapshot.get_scenes_by_ids(
                [own_scene, foreign_scene])}, {own_scene})
            self.assertEqual(len(snapshot.dependencies_for(own_derived)), 1)
            self.assertEqual(snapshot.dependencies_for(foreign_derived), [])
            self.assertEqual(snapshot.scene_cell_ids(own_scene), [own])
            self.assertEqual(snapshot.scene_cell_ids(foreign_scene), [])

    def test_snapshot_purge_guard_observes_live_epoch(self):
        aid = self.insert()
        with self.st.snapshot('u') as snapshot:
            old_epoch = snapshot.read_epoch
            self.writer.purge_user('u')
            with self.assertRaises(store.MemoryDeleted):
                snapshot.assert_epoch('u', old_epoch)
        with self.st.snapshot('u') as snapshot:
            self.assertGreater(snapshot.read_epoch, old_epoch)
            self.assertEqual(snapshot.get_amus_by_ids([aid]), [])

    def test_snapshot_cannot_read_another_users_feedback_or_open_their_snapshot(self):
        self.st.save_feedback('u', 'event-u', 'hash-u', [], {'note': 'own'})
        self.st.save_feedback('v', 'event-v', 'hash-v', [], {'note': 'private'})
        with self.st.snapshot('u') as snapshot:
            self.assertEqual(snapshot.get_feedback('u', 'event-u')['payload'], {'note': 'own'})
            with self.assertRaises(ValueError):
                snapshot.get_feedback('v', 'event-v')
            with self.assertRaises(ValueError):
                with snapshot.snapshot('v'):
                    self.fail('user-scoped snapshot opened another user scope')

    def test_snapshot_min_revision_and_future_time_are_enforced(self):
        self.insert()
        with self.assertRaises(ValueError):
            with self.st.snapshot('u', min_revision=self.st.user_state('u')['revision'] + 1):
                self.fail('uncommitted revision was accepted')
        with self.assertRaises(ValueError):
            with self.st.snapshot('u', as_of='2999-01-01T00:00:00Z'):
                self.fail('future knowledge snapshot was accepted')

    def test_new_snapshot_refilters_changed_eligibility_without_revision_bump(self):
        cases = (
            ('sensitivity', 'sensitive', 'normal', {}),
            ('sensitivity', 'suppressed', 'normal', {'include_sensitive': True}),
            ('view_status', 'stale', 'ready', {}),
            ('resolution_status', 'retracted', 'accepted', {}),
            ('tier', 'cold', 'hot', {}),
            ('valid_to', '2025-01-01T00:00:00Z', None, {}),
        )
        for column, hidden, visible, options in cases:
            with self.subTest(column=column, hidden=hidden):
                aid = self.insert(column + str(hidden), vector=(1, 0, 0))
                with self.st.snapshot('u') as snapshot:
                    self.assertIn(aid, self.query_ids(snapshot, **options))
                revision = self.st.user_state('u')['revision']
                self.writer.conn.execute(f'UPDATE amu SET {column}=? WHERE id=?', (hidden, aid))
                self.writer.conn.commit()
                self.assertEqual(self.st.user_state('u')['revision'], revision)
                with self.st.snapshot('u') as snapshot:
                    self.assertNotIn(aid, self.query_ids(snapshot, **options))
                self.writer.conn.execute(f'UPDATE amu SET {column}=? WHERE id=?', (visible, aid))
                self.writer.conn.commit()
                with self.st.snapshot('u') as snapshot:
                    self.assertIn(aid, self.query_ids(snapshot, **options))

    def test_new_snapshot_refilters_scene_assignment_without_revision_bump(self):
        aid = self.insert(vector=(1, 0, 0))
        first = self.st.insert_scene('u', np.ones(3, dtype=np.float32), ['first'])
        second = self.st.insert_scene('u', np.ones(3, dtype=np.float32), ['second'])
        self.st.assign_scene([aid], first, 'cell-first')
        with self.st.snapshot('u') as snapshot:
            self.assertEqual(self.query_ids(snapshot, scene_ids=[first]), [aid])
        revision = self.st.user_state('u')['revision']
        self.writer.assign_scene([aid], second, 'cell-second')
        self.assertEqual(self.st.user_state('u')['revision'], revision)
        with self.st.snapshot('u') as snapshot:
            self.assertEqual(self.query_ids(snapshot, scene_ids=[first]), [])
            self.assertEqual(self.query_ids(snapshot, scene_ids=[second]), [aid])

    def test_external_embedding_changes_and_space_changes_are_visible(self):
        first = self.insert('first', vector=(1, 0, 0))
        second = self.insert('second', vector=(0, 1, 0))
        with self.st.snapshot('u') as snapshot:
            self.assertEqual(self.query_ids(snapshot, k=1), [first])
        self.writer.conn.execute('UPDATE amu SET embedding=? WHERE id=?', (
            np.asarray([2, 0, 0], dtype=np.float32).tobytes(), second))
        self.writer.conn.commit()
        with self.st.snapshot('u') as snapshot:
            self.assertEqual(self.query_ids(snapshot, k=1), [second])
        self.writer.conn.execute('UPDATE amu SET embedding_space=NULL WHERE id=?', (second,))
        self.writer.conn.commit()
        with self.st.snapshot('u') as snapshot:
            self.assertEqual(self.query_ids(snapshot, k=1), [first])

    def test_new_snapshot_metadata_is_current_after_matrix_was_used(self):
        aid = self.insert('Current content', vector=(1, 0, 0))
        with self.st.snapshot('u') as snapshot:
            self.assertEqual(self.query_ids(snapshot), [aid])
        self.writer.conn.execute("UPDATE amu SET content='Changed content',helpful=9 WHERE id=?", (aid,))
        self.writer.conn.commit()
        with self.st.snapshot('u') as snapshot:
            row = snapshot.nearest_many_by_embedding(
                'u', np.asarray([[1, 0, 0]], dtype=np.float32), 1)[0][0]
            self.assertEqual(row['content'], 'Changed content')
            self.assertEqual(row['helpful'], 9)

    def test_small_vector_topk_matches_numpy_reference(self):
        rng = np.random.default_rng(61)
        matrix = rng.normal(size=(53, 9)).astype(np.float32)
        queries = rng.normal(size=(7, 9)).astype(np.float32)
        ids = [self.insert(str(index), vector=vector) for index, vector in enumerate(matrix)]
        expected_scores = queries @ matrix.T
        with self.st.snapshot('u') as snapshot:
            actual = snapshot.nearest_many_by_embedding('u', queries, 7)
        for result, scores in zip(actual, expected_scores):
            expected = np.argsort(-scores)[:7]
            self.assertEqual([row['id'] for row in result], [ids[index] for index in expected])
            np.testing.assert_allclose([row['_score'] for row in result], scores[expected], rtol=1e-5)

    def test_disabled_approximation_keeps_large_pool_exact(self):
        rng = np.random.default_rng(924)
        matrix = rng.normal(size=(192, 9)).astype(np.float32)
        queries = rng.normal(size=(5, 9)).astype(np.float32)
        ids = [self.insert(str(index), vector=vector) for index, vector in enumerate(matrix)]
        expected_scores = queries @ matrix.T
        with patch.object(config, 'VECTOR_APPROXIMATE', False), patch.object(
                config, 'VECTOR_EXACT_LIMIT', 64), patch.object(
                config, 'VECTOR_CANDIDATE_LIMIT', 64), patch.object(
                config, 'VECTOR_CHUNK_SIZE', 64):
            with self.st.snapshot('u') as snapshot:
                actual = snapshot.nearest_many_by_embedding('u', queries, 10)
                diagnostics = snapshot.vector_diagnostics['queries']
        self.assertEqual(len(actual), len(queries))
        self.assertEqual(len(diagnostics), len(queries))
        for result, scores, diagnostic in zip(actual, expected_scores, diagnostics):
            expected = np.argsort(-scores)[:10]
            self.assertEqual([row['id'] for row in result], [ids[index] for index in expected])
            np.testing.assert_allclose([row['_score'] for row in result], scores[expected], rtol=1e-5)
            self.assertFalse(diagnostic['approximate'])
            self.assertEqual(diagnostic['eligible'], len(ids))
            self.assertEqual(diagnostic['scored'], len(ids))

    def test_vector_ties_are_deterministic_by_id(self):
        ids = [self.insert(str(index), vector=(1, 0, 0)) for index in range(12)]
        with self.st.snapshot('u') as snapshot:
            self.assertEqual(self.query_ids(snapshot, k=4), sorted(ids)[:4])

    def test_vector_empty_zero_limit_and_dimension_errors(self):
        with self.st.snapshot('u') as snapshot:
            self.assertEqual(self.query_ids(snapshot), [])
        self.insert(vector=(1, 0, 0))
        with self.st.snapshot('u') as snapshot:
            self.assertEqual(self.query_ids(snapshot, k=0), [])
            self.assertEqual(self.query_ids(snapshot, scene_ids=[]), [])
            with self.assertRaises(ValueError):
                self.query_ids(snapshot, vector=(1, 0))

    def test_large_index_refilters_new_sensitive_and_cold_metadata(self):
        rng = np.random.default_rng(716)
        vectors = rng.normal(size=(192, 3)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        ids = [self.insert(str(index), vector=vector) for index, vector in enumerate(vectors)]
        with patch.object(config, 'VECTOR_APPROXIMATE', True), patch.object(
                config, 'VECTOR_EXACT_LIMIT', 64), patch.object(
                config, 'VECTOR_CANDIDATE_LIMIT', 64):
            with self.st.snapshot('u') as snapshot:
                original = self.query_ids(snapshot, k=12)
                self.assertEqual(len(original), 12)
            revision = self.st.user_state('u')['revision']
            sensitive, cold = original[:6], original[6:]
            self.writer.conn.executemany("UPDATE amu SET sensitivity='sensitive' WHERE id=?",
                                         [(mid,) for mid in sensitive])
            self.writer.conn.executemany("UPDATE amu SET tier='cold' WHERE id=?",
                                         [(mid,) for mid in cold])
            self.writer.conn.commit()
            self.assertEqual(self.st.user_state('u')['revision'], revision)
            with self.st.snapshot('u') as snapshot:
                current = self.query_ids(snapshot, k=12)
                self.assertEqual(len(current), 12)
                self.assertFalse(set(current) & set(original))
                self.assertTrue(set(current) <= set(ids))
            # Re-enabling both policies must restore the same immutable ranking.
            with self.st.snapshot('u') as snapshot:
                self.assertEqual(self.query_ids(snapshot, k=12,
                    include_sensitive=True, include_cold=True), original)

    def test_large_index_uses_exact_small_eligible_pool_after_restriction(self):
        ids = [self.insert(str(index), vector=(1, index / 200, 0)) for index in range(128)]
        with patch.object(config, 'VECTOR_APPROXIMATE', True), patch.object(
                config, 'VECTOR_EXACT_LIMIT', 64), patch.object(
                config, 'VECTOR_CANDIDATE_LIMIT', 64):
            with self.st.snapshot('u') as snapshot:
                self.assertEqual(len(self.query_ids(snapshot, k=8)), 8)
            eligible = set(ids[-8:])
            self.writer.conn.executemany("UPDATE amu SET sensitivity='suppressed' WHERE id=?",
                                         [(mid,) for mid in ids if mid not in eligible])
            self.writer.conn.commit()
            with self.st.snapshot('u') as snapshot:
                self.assertEqual(set(self.query_ids(snapshot, k=8,
                    include_sensitive=True, include_cold=True)), eligible)

    def test_large_index_rebuilds_after_external_vector_update_without_revision(self):
        for index in range(96):
            self.insert(str(index), vector=(-1, 0, 0))
        changed = self.insert('changed vector', vector=(-1, 0, 0))
        with patch.object(config, 'VECTOR_APPROXIMATE', True), patch.object(
                config, 'VECTOR_EXACT_LIMIT', 64), patch.object(
                config, 'VECTOR_CANDIDATE_LIMIT', 64):
            with self.st.snapshot('u') as snapshot:
                self.query_ids(snapshot, k=1)
            revision = self.st.user_state('u')['revision']
            self.writer.conn.execute('UPDATE amu SET embedding=? WHERE id=?', (
                np.asarray([1, 0, 0], dtype=np.float32).tobytes(), changed))
            self.writer.conn.commit()
            self.assertEqual(self.st.user_state('u')['revision'], revision)
            with self.st.snapshot('u') as snapshot:
                self.assertEqual(self.query_ids(snapshot, k=1), [changed])
            self.writer.conn.execute('UPDATE amu SET embedding_space=NULL WHERE id=?', (changed,))
            self.writer.conn.commit()
            with self.st.snapshot('u') as snapshot:
                self.assertNotIn(changed, self.query_ids(snapshot, k=12))

    def test_index_built_from_old_snapshot_after_purge_is_not_republished(self):
        aid = self.insert('Pending private vector', vector=(1, 0, 0))
        built, release = threading.Event(), threading.Event()
        original_index = vector_index.Index

        def blocked_index(rows):
            result = original_index(rows)
            built.set()
            if not release.wait(10):
                raise AssertionError('test failed to release the index builder')
            return result

        with patch.object(config, 'VECTOR_CACHE_BYTES', 1024 * 1024), self.st.snapshot('u') as snapshot:
            with ThreadPoolExecutor(max_workers=1) as executor:
                with patch.object(vector_index, 'Index', side_effect=blocked_index):
                    pending = executor.submit(self.query_ids, snapshot)
                    try:
                        self.assertTrue(built.wait(5), 'index builder did not reach publication barrier')
                        # The vector has been loaded from the pinned WAL read,
                        # but get() has not yet had a chance to publish it.
                        self.writer.purge_user('u')
                        release.set()
                        self.assertEqual(pending.result(timeout=5), [aid])
                    finally:
                        release.set()
            self.assertFalse(snapshot.vector_diagnostics['cache_hit'])
            # A second call on that same old snapshot must rebuild again. A
            # cache hit here would expose resurrection after purge eviction.
            self.assertEqual(self.query_ids(snapshot), [aid])
            self.assertFalse(snapshot.vector_diagnostics['cache_hit'])
            with self.assertRaises(store.MemoryDeleted):
                snapshot.assert_epoch('u', snapshot.read_epoch)
        with self.st.snapshot('u') as snapshot:
            self.assertEqual(self.query_ids(snapshot), [])


class SearchWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_rechecks_live_epoch_after_snapshot_cleanup(self):
        st = store.Store(':memory:')
        aid = st.insert_amu(user_id='u', session_id='s', content='Private response')
        response = schemas.SearchResponse(
            data=[schemas.SearchItem(id=aid, content='Private response')],
            evidence_status='retrieved')
        original_close = local_work.close_snapshot
        purged = []

        async def close_then_purge(cm):
            await original_close(cm)
            purged.append(st.purge_user('u'))

        try:
            with patch.object(search_pipeline, '_run_search', AsyncMock(return_value=response)) as search, patch.object(
                    local_work, 'close_snapshot', side_effect=close_then_purge):
                with self.assertRaises(store.MemoryDeleted):
                    await search_pipeline.run_search(st, schemas.SearchRequest(user_id='u', query='private'))
            search.assert_awaited_once()
            self.assertEqual(len(purged), 1)
            self.assertEqual(st.get_amus('u'), [])
        finally:
            st.conn.close()

    async def test_expired_budget_does_not_start_local_call(self):
        called = threading.Event()
        with budget.scope(seconds=-1):
            with self.assertRaises(TimeoutError):
                await local_work.run(called.set)
        self.assertFalse(called.is_set())

    async def test_explicit_budget_cancellation_does_not_start_local_call(self):
        called = threading.Event()
        with budget.scope(seconds=30) as limits:
            limits.cancelled.set()
            with self.assertRaises(TimeoutError):
                await local_work.run(called.set)
        self.assertFalse(called.is_set())

    async def test_worker_checks_deadline_after_blocking_phase(self):
        loop = asyncio.get_running_loop()
        started, other_task_ran = asyncio.Event(), asyncio.Event()
        release = threading.Event()

        def work():
            loop.call_soon_threadsafe(started.set)
            if not release.wait(10):
                raise AssertionError('test failed to release the worker')
            budget.check()
            self.fail('expired worker passed its cooperative checkpoint')

        async def independent_task():
            other_task_ran.set()

        with budget.scope(seconds=30) as limits:
            task = asyncio.create_task(local_work.run(work))
            try:
                await asyncio.wait_for(started.wait(), 5)
                observer = asyncio.create_task(independent_task())
                await asyncio.wait_for(other_task_ran.wait(), 5)
                await observer
                self.assertFalse(task.done())
                limits.deadline = time.monotonic() - 1
                release.set()
                with self.assertRaises(TimeoutError):
                    await task
            finally:
                release.set()
                if not task.done():
                    await asyncio.gather(task, return_exceptions=True)

    async def test_cancelled_worker_finishes_before_snapshot_connection_closes(self):
        st = store.Store(':memory:')
        st.insert_amu(user_id='u', session_id='s', content='Still available to worker')
        loop = asyncio.get_running_loop()
        started, closed, other_task_ran = asyncio.Event(), asyncio.Event(), asyncio.Event()
        release = threading.Event()
        checked = []

        def blocked_read(snapshot):
            loop.call_soon_threadsafe(started.set)
            if not release.wait(10):
                raise AssertionError('test failed to release the worker')
            # A small SQL statement tests connection lifetime independently of
            # the progress handler's cooperative deadline interruption.
            checked.append(snapshot.conn.execute('SELECT 1').fetchone()[0])
            budget.check()

        async def request():
            cm = st.snapshot('u')
            snapshot = await local_work.run(cm.__enter__)
            try:
                await local_work.run(blocked_read, snapshot)
            finally:
                await local_work.close_snapshot(cm)
                closed.set()

        async def observe_cancel(limits):
            while not limits.cancelled.is_set():
                await asyncio.sleep(0)
            other_task_ran.set()

        try:
            with budget.scope(seconds=30) as limits:
                task = asyncio.create_task(request())
                try:
                    await asyncio.wait_for(started.wait(), 5)
                    task.cancel()
                    await asyncio.wait_for(observe_cancel(limits), 5)
                    self.assertTrue(other_task_ran.is_set())
                    self.assertFalse(task.done())
                    self.assertFalse(closed.is_set())
                    release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self.assertEqual(checked, [1])
                    self.assertTrue(closed.is_set())
                finally:
                    release.set()
                    if not task.done():
                        await asyncio.gather(task, return_exceptions=True)
        finally:
            st.conn.close()


if __name__ == '__main__':
    unittest.main()
