"""Regressions for governance target eligibility and invalid model decisions."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import add_pipeline as add, config, llm, schemas, store
from app.embeddings import embed


class GovernanceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')
        self.debug = patch.object(config, 'MEMORY_DEBUG_LOG', '')
        self.debug.start()

    def tearDown(self):
        self.debug.stop()
        self.st.conn.close()

    def request(self, rid='new', text='Alice enjoys pottery.'):
        return schemas.AddRequest(request_id=rid, user_id='u', session_id='s',
                                  messages=[schemas.Message(role='user', content=text)])

    def fact(self, text='Alice enjoys pottery.', **extra):
        return dict(content=text, type='fact', entities=['Alice'], _sources=[0],
                    evidence=[dict(message_index=0, request_id='new', quote=text)], **extra)

    async def existing(self, **extra):
        req = self.request('old')
        self.st.save_messages(req)
        aid = self.st.insert_amu(user_id='u', session_id='s', content=req.messages[0].content,
                                 entities=['Alice'], embedding=(await embed(['pottery']))[0],
                                 **extra)
        self.st.link_sources(aid, req.request_id, [0])
        return aid

    def decision(self, op, target, merged=None):
        return dict(operation=op, target_id=target, merged_content=merged, reason='test')

    async def test_sensitive_entity_noop_commits_and_retains_sources(self):
        target = await self.existing(sensitivity='sensitive')
        req = self.request()
        fact = self.fact(sensitivity='sensitive')
        with patch.object(add, '_extract', AsyncMock(return_value={'facts': [fact]})), patch.object(
                llm, 'complete_json', AsyncMock(return_value=self.decision('NOOP', target))):
            await add.run_add(self.st, req)
        self.assertTrue(self.st.request_seen(req.request_id))
        self.assertEqual(len(self.st.get_amus('u')), 1)
        self.assertEqual(len(self.st.sources_for_amu(target)), 2)
        self.assertEqual(self.st.get_amus_by_ids([target]), [])
        self.assertEqual(self.st.get_amus_by_ids([target], include_sensitive=True)[0]['sensitivity'],
                         'sensitive')

    async def test_sensitive_update_preserves_privacy(self):
        target = await self.existing(sensitivity='sensitive')
        text = 'Alice enjoys pottery on weekends.'
        req = self.request(text=text)
        self.st.save_messages(req)
        with patch.object(llm, 'complete_json', AsyncMock(return_value=self.decision(
                'UPDATE', target, text))):
            await add._persist_fact(self.st, req, self.fact(text), (await embed([text]))[0])
        row = self.st.get_amus_by_ids([target], include_sensitive=True)[0]
        self.assertEqual(row['content'], text)
        self.assertEqual(row['sensitivity'], 'sensitive')
        self.assertEqual(self.st.get_amus_by_ids([target]), [])
        self.assertEqual(len(self.st.sources_for_amu(target)), 2)

    async def test_sensitive_state_supersede_preserves_history(self):
        state = dict(subject='Alice', attribute='primary_residence', value='Paris')
        target = await self.existing(sensitivity='sensitive', state=state, valid_from='2023-01-01')
        fact = self.fact('Alice lives in Berlin.', sensitivity='sensitive',
                         state=dict(state, value='Berlin'),
                         temporal=dict(start='2023-06-01T00:00:00Z', precision='day'))
        req = self.request(text=fact['content'])
        self.st.save_messages(req)
        with patch.object(llm, 'complete_json', AsyncMock(return_value=self.decision('SUPERSEDE', target))):
            aid = await add._persist_fact(self.st, req, fact, (await embed(['Berlin']))[0])
        new = self.st.get_amus_by_ids([aid], include_sensitive=True)[0]
        old = self.st.get_amus_by_ids([target], include_sensitive=True, include_history=True)[0]
        self.assertEqual(new['supersedes'], target)
        self.assertEqual(old['superseded_by'], aid)
        self.assertIsNotNone(old['valid_to'])

    async def test_inactive_entity_and_state_candidates_are_not_shown(self):
        state = dict(subject='Alice', attribute='primary_residence', value='Paris')
        good = await self.existing(sensitivity='sensitive', state=state)
        blocked = []
        for field, value in (('sensitivity', 'suppressed'), ('resolution_status', 'retracted'),
                             ('view_status', 'stale'), ('valid_to', '2023-01-01')):
            aid = self.st.insert_amu(user_id='u', session_id='s', content='Blocked candidate ' + field,
                                     entities=['Alice'], state=state)
            self.st.conn.execute(f'UPDATE amu SET {field}=? WHERE id=?', (value, aid))
            blocked.append(aid)
        self.st.conn.commit()
        response = AsyncMock(return_value=self.decision('NOOP', good))
        with patch.object(llm, 'complete_json', response):
            await add._govern_one(self.st, 'u', self.fact(state=state), (await embed(['new']))[0])
        prompt = response.await_args.args[0]
        self.assertIn(good, prompt)
        for aid in blocked:
            self.assertNotIn(aid, prompt)

    async def test_invalid_model_targets_fall_back_to_add_without_mutation(self):
        target = await self.existing()
        unseen = self.st.insert_amu(user_id='u', session_id='s', content='Unrelated Bob memory')
        foreign = self.st.insert_amu(user_id='other', session_id='s', content='Another user memory')
        originals = {aid: self.st.get_amus_by_ids([aid])[0] for aid in (target, unseen, foreign)}
        vec = (await embed(['new unrelated detail']))[0]
        for op in ('UPDATE', 'SUPERSEDE', 'NOOP'):
            for index, invalid in enumerate(('amu_missing', unseen, foreign, None, '', 'null')):
                # Distinct facts keep later cases out of the exact-duplicate
                # novelty gate, so each exercises the model decision boundary.
                text = f'Alice enjoys pottery, detail {op} {index}.'
                req = self.request(f'{op}-{index}', text)
                self.st.save_messages(req)
                fact = self.fact(text)
                fact['evidence'][0]['request_id'] = req.request_id
                with self.subTest(op=op, target=invalid), patch.object(
                        llm, 'complete_json', AsyncMock(return_value=self.decision(op, invalid))), \
                        self.assertLogs('aml.add', level='WARNING') as logs:
                    aid = await add._persist_fact(self.st, req, fact, vec)
                    self.assertIsNotNone(aid)
                    self.assertIsNone(self.st.get_amus_by_ids([aid])[0]['supersedes'])
                    self.assertIn('governance', ' '.join(logs.output).lower())
        for aid, before in originals.items():
            after = self.st.get_amus_by_ids([aid])[0]
            for key in ('content', 'version', 'valid_to', 'superseded_by', 'sensitivity'):
                self.assertEqual(before.get(key), after.get(key))

    async def test_invalid_model_target_does_not_abort_chunk_or_drop_fact(self):
        target = await self.existing()
        req = self.request(text='Alice enjoys hiking.')
        facts = [self.fact(req.messages[0].content),
                 dict(content='user: ' + req.messages[0].content, type='episode', _sources=[0])]
        with patch.object(add, '_extract', AsyncMock(return_value={'facts': facts})), patch.object(
                llm, 'complete_json', AsyncMock(return_value=self.decision('NOOP', 'amu_missing'))):
            await add.run_add(self.st, req)
        self.assertTrue(self.st.request_seen(req.request_id))
        rows = self.st.get_amus('u')
        self.assertEqual(len(rows), 3)
        for row in rows:
            if row['id'] != target:
                self.assertEqual(self.st.sources_for_amu(row['id'])[0]['request_id'], req.request_id)


if __name__ == '__main__':
    unittest.main(verbosity=2)
