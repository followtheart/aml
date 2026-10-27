"""Offline acceptance cases for bitemporal provenance and snapshot reads."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import store, schemas, add_pipeline, integrity


class VersionedStorageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')

    def tearDown(self):
        self.st.conn.close()

    def insert(self, content='Lives in Paris'):
        return self.st.insert_amu(user_id='u', session_id='s', content=content,
                                  valid_from='2024-01-01T00:00:00+00:00')

    def test_old_knowledge_survives_world_change(self):
        with patch.object(store, '_now', return_value='2024-06-01T00:00:00+00:00'):
            aid = self.insert()
        with patch.object(store, '_now', return_value='2024-06-10T00:00:00+00:00'):
            self.st.close_validity(aid, '2024-06-01T00:00:00+00:00')
        with self.st.snapshot('u', as_of='2024-06-05T00:00:00+00:00') as snap:
            self.assertIsNone(snap.get_amus('u')[0]['valid_to'])
        self.assertEqual(self.st.get_amus('u', False)[0]['valid_to'], '2024-06-01T00:00:00+00:00')

    def test_content_correction_preserves_old_content(self):
        with patch.object(store, '_now', return_value='2024-06-01T00:00:00+00:00'):
            aid = self.insert()
        with patch.object(store, '_now', return_value='2024-06-10T00:00:00+00:00'):
            self.st.update_amu_content(aid, 'Lives in Shanghai')
        with self.st.snapshot('u', as_of='2024-06-05T00:00:00+00:00') as snap:
            self.assertEqual(snap.get_amus('u')[0]['content'], 'Lives in Paris')
        self.assertEqual(self.st.get_amus('u')[0]['version'], 2)

    def test_snapshot_min_revision_and_future_guard(self):
        self.insert()
        with self.assertRaises(ValueError):
            with self.st.snapshot('u', min_revision=9999):
                pass
        with self.assertRaises(ValueError):
            with self.st.snapshot('u', as_of='2999-01-01T00:00:00Z'):
                pass

    def test_source_identity_support_dedup_and_stale_dependency(self):
        msg = schemas.Message(role='user', content='I live in Paris.', source_event_id='event-1',
                              speaker_id='alice', source_kind='dialog', trust_scope=['personal'])
        self.st.save_messages(schemas.AddRequest(request_id='r', user_id='u', session_id='s', messages=[msg]))
        aid = self.insert()
        self.st.link_sources(aid, 'r', [0])
        self.st.link_sources(aid, 'r', [0])
        self.assertEqual(self.st.support_events(aid), ['event-1'])
        child = self.insert('Alice likes France')
        self.st.register_dependencies('amu', child, [aid])
        self.st.update_amu_content(aid, 'Lives in Shanghai')
        self.assertEqual(next(r for r in self.st.get_amus('u', False) if r['id'] == child)['view_status'], 'stale')
        self.assertEqual(self.st.sources_for_amu(aid)[0]['speaker_id'], 'alice')

    def test_purge_erases_versions(self):
        aid = self.insert()
        self.st.update_amu_content(aid, 'Corrected')
        self.st.purge_user('u')
        self.assertEqual(self.st.conn.execute('SELECT COUNT(*) FROM claim_versions').fetchone()[0], 0)

    def test_dependency_permissions_intersect_and_sensitive_inherits(self):
        a, b, child = self.insert(), self.insert(), self.insert()
        self.st.conn.execute("UPDATE amu SET sensitivity='sensitive',trust_scope='[\"personal\",\"team\"]' WHERE id=?", (a,))
        self.st.conn.execute("UPDATE amu SET trust_scope='[\"personal\"]' WHERE id=?", (b,))
        self.st.conn.commit()
        self.st.register_dependencies('amu', child, [a, b])
        row = next(r for r in self.st.get_amus('u') if r['id'] == child)
        self.assertEqual(row['sensitivity'], 'sensitive')
        self.assertEqual(row['trust_scope'], '["personal"]')

    def test_day_boundary_and_unknown_precision(self):
        self.assertEqual(integrity.state_boundary({'temporal': integrity.resolve_time('2024-06-01', None)}),
                         '2024-06-01T00:00:00+00:00')
        self.assertIsNone(integrity.state_boundary({'temporal': integrity.resolve_time('2024-06', None)}))
        self.assertIsNone(integrity.state_boundary({}))

    def test_unicode_span_rejects_incorrect_offsets(self):
        message = schemas.Message(role='user', content='我住上海。')
        with self.assertRaises(ValueError):
            integrity.verify_quotes({'evidence': [{'message_index': 0, 'quote': '上海', 'start': 3, 'end': 5}]}, [message])

    def test_cold_fts_filters_before_limit(self):
        self.st.insert_amu(user_id='u', session_id='s', content='Secret secret secret', sensitivity='sensitive')
        aid = self.st.insert_amu(user_id='u', session_id='s', content='Secret')
        self.st.conn.execute("UPDATE amu SET tier='cold' WHERE id=?", (aid,))
        self.st.conn.commit()
        self.assertEqual(self.st.fts_search('u', 'Secret', 1), [])
        self.assertEqual(self.st.fts_search('u', 'Secret', 1, include_cold=True)[0]['id'], aid)

    def test_feedback_ledger_is_idempotent_and_purged(self):
        with self.st.staged('u') as work:
            work.save_feedback('u', 'f1', 'hash', [], {'outcome': 'success'})
        self.assertEqual(self.st.get_feedback('u', 'f1')['payload'], {'outcome': 'success'})
        with self.assertRaises(ValueError):
            self.st.save_feedback('u', 'f1', 'different', [], {})
        self.st.purge_user('u')
        self.assertIsNone(self.st.get_feedback('u', 'f1'))

    def test_suppression_invalidates_profile_and_cannot_be_replayed(self):
        with patch.object(store, '_now', return_value='2024-06-01T00:00:00+00:00'):
            aid = self.insert()
            child = self.st.insert_amu(user_id='u', session_id='s', content='Profile', type='profile')
            self.st.register_dependencies('amu', child, [aid])
        self.assertTrue(self.st.core_profile('u', 10))
        with patch.object(store, '_now', return_value='2024-06-10T00:00:00+00:00'):
            self.st.suppress(aid, '2024-06-10T00:00:00+00:00')
        self.assertEqual(self.st.core_profile('u', 10), [])
        with self.st.snapshot('u', as_of='2024-06-05T00:00:00+00:00') as snap:
            self.assertFalse(snap.get_amus_by_ids([aid], include_history=True, include_sensitive=True))

    def test_same_event_retains_all_source_permissions(self):
        messages = [schemas.Message(role='user', content='Address A', source_event_id='same'),
                    schemas.Message(role='user', content='Address B', source_event_id='same', sensitivity='sensitive')]
        self.st.save_messages(schemas.AddRequest(request_id='r', user_id='u', session_id='s', messages=messages))
        aid = self.insert()
        self.st.link_sources(aid, 'r', [0, 1])
        self.assertEqual(len(self.st.sources_for_amu(aid)), 2)
        self.assertEqual(self.st.support_events(aid), ['same'])
        self.assertEqual(self.st.get_amus('u')[0]['sensitivity'], 'sensitive')

    async def test_late_report_uses_actual_day_boundary(self):
        with patch.object(store, '_now', return_value='2024-06-01T00:00:00+00:00'):
            old = self.st.insert_amu(user_id='u', session_id='s', content='Alice lives in Paris',
                valid_from='2024-01-01T00:00:00+00:00', state={'subject':'Alice','attribute':'residence','value':'Paris'})
        req = schemas.AddRequest(request_id='r', user_id='u', session_id='s', messages=[schemas.Message(role='user', content='Alice moved on 2024-06-01')])
        fact = dict(content='Alice lives in Shanghai', state={'subject':'Alice','attribute':'residence','value':'Shanghai'},
                    temporal=integrity.resolve_time('2024-06-01', None))
        with patch.object(store, '_now', return_value='2024-06-10T00:00:00+00:00'), patch.object(add_pipeline, '_govern_one', AsyncMock(return_value=('SUPERSEDE', {'target_id':old}))):
            new = await add_pipeline._persist_fact(self.st, req, fact, np.zeros(3))
        self.assertEqual(self.st.get_amus_by_ids([new])[0]['valid_from'], '2024-06-01T00:00:00+00:00')
        with self.st.snapshot('u', as_of='2024-06-05T00:00:00+00:00') as snap:
            self.assertIsNone(snap.get_amus_by_ids([old])[0]['valid_to'])


if __name__ == '__main__':
    unittest.main()
