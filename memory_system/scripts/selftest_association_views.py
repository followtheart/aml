"""Source-bound view lifecycle, isolated transactions and retrieval contracts."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

os.environ['AML_FAKE'] = '1'
os.environ['AML_MEMORY_DEBUG_LOG'] = ''
os.environ['AML_SEARCH_DEBUG_LOG'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import add_pipeline, association_views as av, config, graph_fusion, schemas, store, search_pipeline


def vector(index=0):
    result = np.zeros(config.EMBED_DIM, dtype=np.float32)
    result[index] = 1
    return result


def request(user='u', rid='r'):
    return schemas.AddRequest(user_id=user, request_id=rid, session_id='s', messages=[
        schemas.Message(role='user', content='I am allergic to peanuts.', timestamp=100)])


async def generated(sources):
    return [(s, 'Dietary restrictions for dinner menus', 'allergic to peanuts') for s in sources]


async def embedded(texts, **kwargs):
    return np.stack([vector() for _ in texts])


class Validation(unittest.TestCase):
    def setUp(self):
        self.sources = {'s0': dict(content='I am allergic to peanuts.')}

    def clean(self, **kwargs):
        row = dict(source='s0', text='Dinner dietary restrictions', anchor='allergic to peanuts')
        row.update(kwargs)
        return av.clean_views({'views': [row]}, self.sources, 3)

    def test_unique_verbatim_anchor(self):
        self.assertEqual(len(self.clean()), 1)
        for kwargs in ({'anchor': 'allergic to almonds'}, {'source': 's1'},
                       {'anchor': ''}, {'text': 'x'*601}, {'source': []}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.clean(**kwargs), [])

    def test_repeated_anchor_is_rejected(self):
        self.sources['s0']['content'] *= 2
        self.assertEqual(self.clean(), [])

    def test_invalid_anchor_does_not_discard_another_valid_view(self):
        rows = [dict(source='s0', text='Dinner restriction', anchor='peanuts'),
                dict(source='s0', text='Invalid premise', anchor='almonds')]
        from collections import Counter
        rejected = Counter()
        self.assertEqual(len(av.clean_views({'views': rows}, self.sources, 3, rejected)), 1)
        self.assertEqual(rejected['invalid_text_or_anchor'], 1)

    def test_duplicate_views_do_not_multiply(self):
        row = dict(source='s0', text='Dinner restriction', anchor='peanuts')
        self.assertEqual(len(av.clean_views({'views': [row, row]}, self.sources, 3)), 1)

    def test_provider_alignment(self):
        for values in ([vector()[:-1]], [vector()*float('nan')], [vector()*0], [vector(), vector()]):
            with self.assertRaises(ValueError):
                av.normalized(values, 1)


class Lifecycle(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')
        self.gen = patch.object(av, 'generate', side_effect=generated)
        self.emb = patch.object(av.embeddings, 'embed', side_effect=embedded)
        self.gen.start(); self.emb.start()

    def tearDown(self):
        self.emb.stop(); self.gen.stop(); self.st.conn.close()

    def source(self, user='u', rid='r', **kwargs):
        self.st.save_messages(request(user, rid))
        mid = self.st.insert_amu(user_id=user, session_id='s', content='Peanut allergy',
            embedding=vector(1), **kwargs)
        self.st.link_sources(mid, rid, [0])
        return mid

    async def index(self, user='u', rid=None):
        with self.st.staged(user) as work:
            return await av.index_sources(work, user, rid)

    def recall(self, user='u', **kwargs):
        with self.st.snapshot(user) as snap:
            return av.recall(snap, user, [vector()], **kwargs)[0]

    async def test_recall_returns_only_original_and_separate_score(self):
        mid = self.source()
        stats = await self.index()
        rows = self.recall()
        self.assertEqual(stats['views'], 1)
        self.assertEqual([r['id'] for r in rows], [mid])
        self.assertEqual(rows[0]['content'], 'Peanut allergy')
        self.assertNotIn('Dietary restrictions', str(rows))
        np.testing.assert_array_equal(rows[0]['embedding'], vector(1))

    async def test_tenant_isolation_and_wrong_snapshot(self):
        first = self.source()
        self.source('other', 'other-r')
        await self.index('u'); await self.index('other')
        self.assertEqual([r['id'] for r in self.recall()], [first])
        self.assertEqual(self.recall('missing'), [])
        with self.st.snapshot('u') as snap, self.assertRaises(ValueError):
            av.recall(snap, 'other', [vector()])

    async def test_unpublished_views_are_private_and_rollback(self):
        self.source()
        with self.assertRaises(RuntimeError):
            with self.st.staged('u') as work:
                await av.index_sources(work, 'u')
                self.assertEqual(self.recall(), [])
                raise RuntimeError('rollback')
        self.assertEqual(self.recall(), [])

    async def test_purge_rejects_inflight_publication(self):
        self.source()
        with self.assertRaises(store.MemoryDeleted):
            with self.st.staged('u') as work:
                await av.index_sources(work, 'u')
                self.st.purge_user('u')
        self.assertEqual(self.st.conn.execute('select count(*) from association_views').fetchone()[0], 0)

    async def test_purge_removes_views_only_for_target_user(self):
        self.source(); self.source('other', 'other-r')
        await self.index(); await self.index('other')
        receipt = self.st.purge_user('u')
        self.assertEqual(receipt['row_counts']['association_views'], 1)
        self.assertEqual(self.recall(), [])
        self.assertEqual(len(self.recall('other')), 1)

    async def test_version_change_withdraws_old_views(self):
        mid = self.source(); await self.index()
        self.st.replace_fact(mid, dict(content='Changed claim', retrieval_key='', entities=[], keywords=[]), vector(1))
        self.assertEqual(self.recall(include_history=True), [])

    async def test_unlinked_sources_cannot_recall(self):
        mid = self.source(); await self.index()
        self.st.conn.execute('DELETE FROM amu_sources WHERE amu_id=?', (mid,)); self.st.conn.commit()
        self.assertEqual(self.recall(), [])

    async def test_sensitivity_and_history_match_original_paths(self):
        mid = self.source(); await self.index()
        self.st.conn.execute("UPDATE source_messages SET sensitivity='sensitive'")
        self.st.conn.commit()
        self.assertEqual(self.recall(), [])
        self.assertEqual(len(self.recall(include_sensitive=True)), 1)
        # SQL simulates validity filtering independently of version withdrawal.
        self.st.conn.execute("UPDATE amu SET valid_to='2020-01-01' WHERE id=?", (mid,)); self.st.conn.commit()
        self.assertEqual(self.recall(include_sensitive=True), [])
        self.assertEqual(len(self.recall(include_sensitive=True, include_history=True)), 1)

    async def test_suppressed_retracted_stale_never_recall(self):
        mid = self.source(); await self.index()
        for field, value, reset in (('sensitivity', 'suppressed', 'normal'),
                                    ('resolution_status', 'retracted', 'accepted'),
                                    ('view_status', 'stale', 'ready')):
            self.st.conn.execute(f'UPDATE amu SET {field}=? WHERE id=?', (value, mid)); self.st.conn.commit()
            self.assertEqual(self.recall(include_sensitive=True, include_history=True), [])
            self.st.conn.execute(f'UPDATE amu SET {field}=? WHERE id=?', (reset, mid)); self.st.conn.commit()

    async def test_source_fingerprint_guards_content_and_role(self):
        self.source(); await self.index()
        self.st.conn.execute("UPDATE source_messages SET role='assistant'"); self.st.conn.commit()
        self.assertEqual(self.recall(), [])

    async def test_as_of_does_not_see_future_materialization(self):
        self.source(); await self.index()
        with self.st.snapshot('u') as snap:
            snap.read_as_of = '2000-01-01T00:00:00+00:00'
            self.assertEqual(av.recall(snap, 'u', [vector()])[0], [])

    async def test_source_duplicates_do_not_fill_recall_slots(self):
        self.source()
        extra = self.st.insert_amu(user_id='u', session_id='s', content='Related fact', embedding=vector(1))
        self.st.link_sources(extra, 'r', [0]); await self.index()
        self.assertEqual(self.st.conn.execute('select count(*) from association_views').fetchone()[0], 2)
        self.assertEqual(len(self.recall()), 1)

    async def test_embedding_failure_publishes_no_partial_views(self):
        self.source()
        with patch.object(av.embeddings, 'embed', AsyncMock(return_value=[vector()[:-1]])):
            stats = await self.index()
        self.assertEqual(stats['failed_batches'], 1)
        self.assertEqual(self.recall(), [])
        self.assertEqual(len(self.st.get_amus('u')), 1)

    async def test_large_sources_are_reported_not_truncated(self):
        self.source()
        self.st.conn.execute('UPDATE source_messages SET content=?', ('x'*13000,)); self.st.conn.commit()
        stats = await self.index()
        self.assertEqual(stats['skipped_large'], 1)
        self.assertEqual(self.recall(), [])

    async def test_embedding_space_isolation(self):
        self.source(); await self.index()
        with patch.object(config, 'EMBEDDING_SPACE', 'other-space'):
            self.assertEqual(self.recall(), [])

    async def test_direct_live_indexing_is_rejected(self):
        with self.assertRaises(ValueError):
            await av.index_sources(self.st, 'u')

    async def test_successful_rebuild_removes_old_view_tail(self):
        self.source(); await self.index()
        with patch.object(av, 'generate', AsyncMock(return_value=[])):
            await self.index()
        self.assertEqual(self.recall(), [])

    async def test_corrupt_vector_cannot_poison_original_recall(self):
        self.source(); await self.index()
        self.st.conn.execute('UPDATE association_views SET embedding=?', (b'broken',)); self.st.conn.commit()
        self.assertEqual(self.recall(), [])
        self.assertEqual(len(self.st.get_amus('u')), 1)

    async def test_recall_switch_preserves_existing_routes(self):
        mid = self.source(); await self.index()
        req = schemas.SearchRequest(user_id='u', query='dinner menus')
        plans = []
        with patch.object(search_pipeline, 'embed', side_effect=embedded), \
             patch.object(search_pipeline, '_expand_recall', side_effect=lambda st,req,plan,routes,specs,vecs: routes):
            for enabled in (False, True):
                plan = dict(intent='preference', entities=[], time_scope={})
                with patch.object(config, 'ASSOCIATION_RECALL', enabled), self.st.snapshot('u') as snap:
                    routes = await search_pipeline._recall(snap, req, plan)
                plans.append((plan, routes))
        self.assertEqual([[c['id'] for c in r] for r in plans[0][1]],
                         [[c['id'] for c in r] for r in plans[1][1][:3]])
        self.assertNotIn('_association_recall', plans[0][0])
        self.assertEqual(plans[1][0]['_routes'][3]['channel'], 'association')
        self.assertEqual([c['id'] for c in plans[1][1][3]], [mid])


class Integration(unittest.IsolatedAsyncioTestCase):
    async def test_default_add_is_unchanged_and_enabled_add_is_idempotent(self):
        st = store.Store(':memory:')
        try:
            with patch.object(config, 'ASSOCIATION_WRITE', False), patch.object(av, 'index_sources', AsyncMock()) as index:
                await add_pipeline.run_add(st, request())
                index.assert_not_called()
            with patch.object(config, 'ASSOCIATION_WRITE', True), patch.object(av, 'generate', side_effect=generated), \
                 patch.object(av.embeddings, 'embed', side_effect=embedded):
                await add_pipeline.run_add(st, request(rid='r2'))
                count = st.conn.execute('SELECT COUNT(*) FROM association_views').fetchone()[0]
                self.assertGreater(count, 0)
                await add_pipeline.run_add(st, request(rid='r2'))
                self.assertEqual(st.conn.execute('SELECT COUNT(*) FROM association_views').fetchone()[0], count)
        finally:
            st.conn.close()

    def test_association_is_max_score_not_an_independent_vote(self):
        candidate = dict(id='a', content='Peanut allergy', type='fact', embedding=vector(1),
                         embedding_space=config.EMBEDDING_SPACE, _score=.9)
        direct_plan = dict(query='dinner', _routes=[dict(channel='vector')])
        joined_plan = dict(query='dinner', _routes=[dict(channel='vector'), dict(channel='association')])
        baseline = graph_fusion.fuse([[candidate]], direct_plan)
        repeated = graph_fusion.fuse([[candidate], [candidate, candidate]], joined_plan)
        self.assertEqual(baseline[0]['_fused'], repeated[0]['_fused'])

    def test_association_cannot_lower_existing_source_semantics(self):
        candidate = dict(id='a', content='Peanut allergy', type='fact', embedding=vector(),
                         embedding_space=config.EMBEDDING_SPACE, _score=.4)
        plan = dict(query='dinner', _fusion_query_vectors=[vector()], _routes=[dict(channel='source_text')])
        baseline = graph_fusion.fuse([[candidate]], plan)
        plan['_routes'].append(dict(channel='association'))
        joined = graph_fusion.fuse([[candidate], [candidate]], plan)
        self.assertEqual(baseline[0]['_fusion']['semantic'], joined[0]['_fusion']['semantic'])


if __name__ == '__main__':
    unittest.main()
