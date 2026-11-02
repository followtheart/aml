"""Source lookup must scale with matching links, not all memories of a user."""
import os
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
            expected = self.measured_search(st, 'OrchidLedger')[0]
            state = st.user_state('u')
            st.conn.execute('DROP INDEX idx_amu_sources_source')
            st.conn.commit()
            st.conn.close()
            with patch.multiple(config, SOURCE_FTS_BACKFILL=False, SESSION_FTS_BACKFILL=False):
                for _ in range(2):
                    st = store.Store(path)
                    try:
                        with st.snapshot('u') as view:
                            rows = view.source_search('u', 'OrchidLedger', 5)
                            self.assertEqual([(r['id'], r['_score']) for r in rows], expected)
                        self.assertEqual(st.user_state('u'), state)
                        self.assertEqual(st.conn.execute('SELECT COUNT(*) FROM amu_sources').fetchone()[0], 12)
                    finally:
                        st.conn.close()


if __name__ == '__main__':
    unittest.main()
