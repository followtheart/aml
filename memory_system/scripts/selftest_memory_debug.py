"""Offline checks for committed memory JSONL diagnostics."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import add_pipeline as add, config, schemas, store


class DebugLogTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'logs' / 'memory.jsonl'
        self.setting = patch.object(config, 'MEMORY_DEBUG_LOG', str(self.path))
        self.setting.start()
        self.episodes = patch.object(config, 'STORE_EPISODES', False)
        self.episodes.start()
        self.st = store.Store(':memory:')
        self.req = schemas.AddRequest(request_id='debug-1', user_id='u', session_id='s',
            messages=[schemas.Message(role='user', content='Alice lives in Paris. 中文', timestamp=123)])

    def tearDown(self):
        self.st.conn.close()
        self.setting.stop()
        self.episodes.stop()
        self.tmp.cleanup()

    def records(self):
        return [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines()]

    async def test_saved_values_sources_summary_and_replay(self):
        await add.run_add(self.st, self.req)
        event = self.records()[0]
        memory = event['memories'][0]
        stored = self.st.get_amus('u')[0]
        self.assertEqual(memory['content'], stored['content'])
        self.assertEqual(memory['embedding']['values'], stored['embedding'].tolist())
        self.assertEqual(memory['embedding']['dimensions'], config.EMBED_DIM)
        self.assertEqual(memory['full_text_index'][0]['content'], stored['content'])
        self.assertEqual(memory['triples'], self.st.triples_for_user('u'))
        self.assertEqual(memory['sources'][0]['content'], self.req.messages[0].content)
        self.assertEqual(event['session_summary'], self.st.get_summary('u', 's'))
        self.assertEqual(event['event'], 'memory.add.committed')
        self.assertIn('中文', self.path.read_text(encoding='utf-8'))
        await add.run_add(self.st, self.req)
        self.assertEqual(len(self.records()), 1)

    async def test_rollback_does_not_emit_committed_event(self):
        with patch.object(add, '_update_summary', AsyncMock(side_effect=RuntimeError('fail'))):
            with self.assertRaises(RuntimeError):
                await add.run_add(self.st, self.req)
        self.assertFalse(self.path.exists())

    async def test_update_logs_final_body_and_retained_sources(self):
        await add.run_add(self.st, self.req)
        aid = self.st.get_amus('u')[0]['id']
        self.req.request_id = 'debug-2'
        with patch.object(add, '_govern_one', AsyncMock(return_value=(
                'UPDATE', {'target_id': aid, 'merged_content': 'Alice lives in Berlin.'}))):
            await add.run_add(self.st, self.req)
        memory = self.records()[1]['memories'][0]
        self.assertEqual(memory['id'], aid)
        self.assertEqual(memory['content'], 'Alice lives in Berlin.')
        self.assertEqual(len(memory['sources']), 2)
        self.assertEqual(memory['embedding']['values'], self.st.get_amus('u')[0]['embedding'].tolist())

    async def test_supersede_logs_closed_predecessor(self):
        original = add._extract
        async def state_extract(st, req):
            data = await original(st, req)
            for fact in data['facts']:
                fact['state'] = dict(subject='Alice', attribute='primary_residence',
                                     value='Berlin' if 'Berlin' in fact['content'] else 'Paris')
            return data
        with patch.object(add, '_extract', state_extract):
            await add.run_add(self.st, self.req)
            aid = self.st.get_amus('u')[0]['id']
            self.req.request_id = 'debug-2'
            self.req.messages[0].content = 'Alice lives in Berlin.'
            self.req.messages[0].timestamp = 124
            with patch.object(add, '_govern_one', AsyncMock(return_value=(
                    'SUPERSEDE', {'target_id': aid}))):
                await add.run_add(self.st, self.req)
        memories = {m['id']: m for m in self.records()[1]['memories']}
        self.assertEqual(len(memories), 2)
        self.assertIsNotNone(memories[aid]['valid_to'])

    async def test_disabled(self):
        with patch.object(config, 'MEMORY_DEBUG_LOG', ''):
            await add.run_add(self.st, self.req)
        self.assertFalse(self.path.exists())

    async def test_file_failure_does_not_fail_committed_add(self):
        self.path.parent.mkdir(parents=True)
        self.path.mkdir()
        with self.assertLogs('aml.memory_debug', level='WARNING'):
            await add.run_add(self.st, self.req)
        self.assertTrue(self.st.request_seen(self.req.request_id))


if __name__ == '__main__':
    unittest.main(verbosity=2)
