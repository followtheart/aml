"""Regressions for deletion races, source policy and independent support."""
import asyncio
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import add_pipeline, config, llm, profile, schemas, search_pipeline, store
from app.embeddings import embed


class SafetyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')

    def tearDown(self):
        self.st.conn.close()

    def test_first_staged_add_cannot_survive_purge(self):
        with self.assertRaises(store.MemoryDeleted):
            with self.st.staged('u') as work:
                work.insert_amu(user_id='u', session_id='s', content='private')
                self.st.purge_user('u')
        self.assertEqual(self.st.get_amus('u'), [])

    def test_old_epoch_cannot_start_new_transaction(self):
        epoch = self.st.user_state('u')['epoch']
        self.st.purge_user('u')
        with self.assertRaises(store.MemoryDeleted):
            with self.st.staged('u', expected_epoch=epoch):
                self.fail('obsolete operation entered transaction')

    async def test_search_cannot_return_deleted_candidate(self):
        aid = self.st.insert_amu(user_id='u', session_id='s', content='private')
        rows = self.st.get_amus('u')
        async def deleting_rerank(*args):
            self.st.purge_user('u')
            return rows
        with patch.object(search_pipeline, '_understand', AsyncMock(return_value={'intent': 'fact'})), patch.object(
                search_pipeline, '_recall', AsyncMock(return_value=[rows])), patch.object(
                search_pipeline, '_filter_rerank', side_effect=deleting_rerank):
            with self.assertRaises(store.MemoryDeleted):
                await search_pipeline.run_search(self.st, schemas.SearchRequest(user_id='u', query='private'))

    async def test_sensitive_profile_inherits_policy_and_one_observation(self):
        req = schemas.AddRequest(request_id='r', user_id='u', session_id='s', messages=[
            schemas.Message(role='user', content='The user likes a private hobby.')])
        self.st.save_messages(req)
        vec = (await embed(['The user likes a private hobby.']))[0]
        aid = self.st.insert_amu(user_id='u', session_id='s', content=req.messages[0].content,
                                 type='fact', sensitivity='sensitive', embedding=vec)
        fact = {'content': req.messages[0].content, 'type': 'fact', 'sensitivity': 'sensitive',
                '_sources': [0], 'evidence': []}
        with patch.object(llm, 'complete_json', AsyncMock(return_value={'items': [{
                'content': req.messages[0].content, 'basis': 'stated', 'support_ids': [aid]}]})):
            await profile.consolidate(self.st, req, [(aid, fact)])
        prefs = self.st.get_by_type('u', ['preference'], include_sensitive=True)
        self.assertEqual(len(prefs), 1)
        self.assertEqual(prefs[0]['sensitivity'], 'sensitive')
        self.assertEqual(prefs[0]['profile_status'], 'transient')
        self.assertEqual(len(prefs[0]['support_sessions']), 1)
        self.assertEqual(self.st.core_profile('u', 10), [])

    def test_sensitive_requires_request_opt_in(self):
        self.assertFalse(schemas.SearchRequest(user_id='u', query='x').include_sensitive)


if __name__ == '__main__':
    unittest.main()
