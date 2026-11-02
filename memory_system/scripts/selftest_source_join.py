"""Source lookup must scale with matching links, not all memories of a user."""
import os
from itertools import product
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, schemas, store


class SourceJoinTests(unittest.TestCase):
    def populate(self, st):
        st.save_messages(schemas.AddRequest(
            request_id='sources', user_id='u', session_id='s', messages=[
                schemas.Message(role='user', content=f'OrchidLedger 冲浪 observation {i}')
                for i in range(12)]))
        for i in range(12):
            aid = st.insert_amu(user_id='u', session_id='s', content=f'Claim {i}')
            st.link_sources(aid, 'sources', [i])

    def measured_search(self, st, query):
        steps = 0

        def count():
            nonlocal steps
            steps += 100
            return 0

        st.conn.set_progress_handler(count, 100)
        try:
            rows = st.source_search('u', query, 5)
            return [(r['id'], r['_score']) for r in rows], steps
        finally:
            st.conn.set_progress_handler(None, 0)

    def test_unlinked_memories_do_not_multiply_source_lookup_work(self):
        st = store.Store(':memory:')
        self.addCleanup(st.conn.close)
        self.populate(st)
        before = {q: self.measured_search(st, q) for q in ('OrchidLedger', '冲浪')}
        # These memories have no source links, so they must not add join work.
        st.conn.executemany(
            'INSERT INTO amu(id,user_id,session_id,content,created_at) VALUES (?,?,?,?,?)',
            [(f'unlinked-{i}', 'u', 's', 'Unrelated memory', '2026-01-01') for i in range(2000)])
        st.conn.commit()
        for query, (expected, baseline_steps) in before.items():
            with self.subTest(query=query):
                actual, steps = self.measured_search(st, query)
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual), 5)
                self.assertLess(steps, max(3000, baseline_steps * 3))

    def test_existing_database_migration_preserves_source_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'old.sqlite')
            st = store.Store(path)
            self.populate(st)
            expected = {q: self.measured_search(st, q)[0] for q in ('OrchidLedger', '冲浪')}
            state = st.user_state('u')
            st.conn.execute('DROP INDEX idx_amu_sources_source')
            st.conn.execute('DROP INDEX idx_source_messages_user')
            st.conn.commit()
            st.conn.close()
            with patch.multiple(config, SOURCE_FTS_BACKFILL=False, SESSION_FTS_BACKFILL=False):
                for _ in range(2):
                    st = store.Store(path)
                    try:
                        with st.snapshot('u') as view:
                            for query, result in expected.items():
                                rows = view.source_search('u', query, 5)
                                self.assertEqual([(r['id'], r['_score']) for r in rows], result)
                        self.assertEqual(st.user_state('u'), state)
                        self.assertEqual(st.conn.execute('SELECT COUNT(*) FROM amu_sources').fetchone()[0], 12)
                    finally:
                        st.conn.close()

    def test_other_users_sources_do_not_add_chinese_scan_work(self):
        st = store.Store(':memory:')
        self.addCleanup(st.conn.close)
        self.populate(st)
        expected, baseline_steps = self.measured_search(st, '冲浪')
        st.conn.executemany('''INSERT INTO source_messages
            (request_id,message_index,user_id,session_id,role,content)
            VALUES (?,0,'other','other','user',?)''',
            [(f'foreign-{i}', '其他用户的冲浪记录' * 50) for i in range(3000)])
        st.conn.execute('''INSERT INTO source_fts(request_id,message_index,user_id,content)
            SELECT request_id,message_index,user_id,content FROM source_messages
            WHERE user_id='other' ''')
        st.conn.commit()
        actual, steps = self.measured_search(st, '冲浪')
        self.assertEqual(actual, expected)
        self.assertLess(steps, max(1000, baseline_steps * 2))

    def test_chinese_preserves_all_eligibility_gates_and_scores(self):
        st = store.Store(':memory:')
        self.addCleanup(st.conn.close)

        def insert(name, user='u', source_sensitivity='normal', **columns):
            st.save_messages(schemas.AddRequest(
                request_id=name, user_id=user, session_id='s', messages=[
                    schemas.Message(role='user', content='历史复查记录',
                                    timestamp=946684800000, sensitivity=source_sensitivity)]))
            aid = st.insert_amu(user_id=user, session_id='s', content='Evidence', **columns)
            st.link_sources(aid, name, [0])
            return aid

        current = insert('current')
        old = insert('old', valid_from='2000-01-01', valid_to='2024-01-01')
        secret = insert('secret', sensitivity='sensitive')
        source_secret = insert('source-secret', source_sensitivity='sensitive')
        # Check source privacy independently of the AMU's propagated label.
        st.conn.execute("UPDATE amu SET sensitivity='normal' WHERE id=?", (source_secret,))
        cold = insert('cold')
        st.conn.execute("UPDATE amu SET tier='cold' WHERE id=?", (cold,))
        insert('suppressed', sensitivity='suppressed')
        insert('source-suppressed', source_sensitivity='suppressed')
        stale = insert('stale')
        retracted = insert('retracted')
        st.conn.execute("UPDATE amu SET view_status='stale' WHERE id=?", (stale,))
        st.conn.execute("UPDATE amu SET resolution_status='retracted' WHERE id=?", (retracted,))
        foreign = insert('foreign', user='other')
        # Even malformed legacy cross-user links must not expose either owner.
        st.conn.execute('INSERT INTO amu_sources VALUES (?,?,0)', (current, 'foreign'))
        st.conn.execute('INSERT INTO amu_sources VALUES (?,?,0)', (foreign, 'current'))
        st.conn.commit()
        for history, sensitive, include_cold, diverse in product((False, True), repeat=4):
            with self.subTest(history=history, sensitive=sensitive, cold=include_cold, diverse=diverse), \
                    patch.object(config, 'SOURCE_RECALL_DIVERSITY', diverse):
                expected = {current}
                if history:
                    expected.add(old)
                if sensitive:
                    expected.update((secret, source_secret))
                if include_cold:
                    expected.add(cold)
                with st.snapshot('u') as view:
                    result = view.source_search('u', '历史复查', 100,
                        include_history=history, include_sensitive=sensitive, include_cold=include_cold)
                    self.assertEqual([r['id'] for r in result], sorted(expected))
                    self.assertTrue(all(r['_score'] == 3 for r in result))

    def test_chinese_keeps_diversity_before_limit_and_late_historical_match(self):
        st = store.Store(':memory:')
        self.addCleanup(st.conn.close)
        st.save_messages(schemas.AddRequest(request_id='r', user_id='u', session_id='s', messages=[
            schemas.Message(role='user', content='复查记录', timestamp=1704067200000),
            schemas.Message(role='user', content='历史复查记录', timestamp=946684800000)]))
        for aid, index in [('a', 0), ('b', 0), ('c', 1)]:
            mid = st.insert_amu(user_id='u', session_id='s', content='Evidence')
            st.conn.execute('UPDATE amu SET id=? WHERE id=?', (aid, mid))
            st.link_sources(aid, 'r', [index])
        with patch.object(config, 'SOURCE_RECALL_DIVERSITY', False):
            self.assertEqual([r['id'] for r in st.source_search('u', '复查', 2)], ['a', 'b'])
        with patch.object(config, 'SOURCE_RECALL_DIVERSITY', True):
            self.assertEqual([r['id'] for r in st.source_search('u', '复查', 2)], ['a', 'c'])
        # The stronger match is older and was inserted later: it must not be
        # lost to any date window or source pre-limit, even when k=1.
        self.assertEqual([(r['id'], r['_score']) for r in st.source_search('u', '历史复查', 1)], [('c', 3)])

    def test_chinese_as_of_snapshot_and_purge(self):
        st = store.Store(':memory:')
        self.addCleanup(st.conn.close)
        with patch.object(store, '_now', return_value='2024-06-01T00:00:00+00:00'):
            st.save_messages(schemas.AddRequest(request_id='r', user_id='u', session_id='s', messages=[
                schemas.Message(role='user', content='历史复查记录')]))
            aid = st.insert_amu(user_id='u', session_id='s', content='Earlier claim', valid_from='2024-01-01')
            st.link_sources(aid, 'r', [0])
        with patch.object(store, '_now', return_value='2024-06-10T00:00:00+00:00'):
            st.close_validity(aid, '2024-06-10T00:00:00+00:00')
            st.save_messages(schemas.AddRequest(request_id='later', user_id='u', session_id='s', messages=[
                schemas.Message(role='user', content='历史复查记录')]))
            later = st.insert_amu(user_id='u', session_id='s', content='Later claim')
            st.link_sources(later, 'later', [0])
        with st.snapshot('u', as_of='2024-06-05T00:00:00+00:00') as view:
            self.assertEqual([r['id'] for r in view.source_search('u', '历史复查', 100)], [aid])
        self.assertEqual([r['id'] for r in st.source_search('u', '历史复查', 100)], [later])
        st.purge_user('u')
        self.assertEqual(st.source_search('u', '历史复查', 100, include_history=True), [])


if __name__ == '__main__':
    unittest.main()
