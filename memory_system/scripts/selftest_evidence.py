"""Offline regression cases drawn from the memory debug review."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import add_pipeline as add, config, integrity, llm, schemas, store
from app.embeddings import embed
from app.search_pipeline import _time_prefix


class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')
        self.logging = patch.object(config, 'MEMORY_DEBUG_LOG', '')
        self.logging.start()
        # Chunk fallback semantics are tested without the always-on episode view.
        self.episodes = patch.object(config, 'STORE_EPISODES', False)
        self.episodes.start()

    def tearDown(self):
        self.logging.stop()
        self.episodes.stop()
        self.st.conn.close()

    def req(self, text, rid='r', timestamp=1684972800000):
        return schemas.AddRequest(request_id=rid, user_id='u', session_id='s',
                                  messages=[schemas.Message(role='user', content=text, timestamp=timestamp)])

    def fact(self, text='Alice lives in Paris.', value='Paris'):
        return dict(content=text, type='profile', entities=['Alice'], keywords=[],
                    retrieval_key='Where does Alice live?',
                    state=dict(subject='Alice', attribute='primary_residence', value=value),
                    evidence=[dict(message_index=0, quote=text)], time_expression=None,
                    triples=[dict(subject='Alice', relation='lives_in', object=value)])

    async def test_wrong_source_index_preserves_episode(self):
        req = self.req('Where does Alice live?')
        req.messages.append(schemas.Message(role='assistant', content='Alice lives in Paris.'))
        with patch.object(llm, 'complete_json', AsyncMock(return_value={'facts': [self.fact()]})):
            result = await add._extract(self.st, req)
        self.assertEqual(result['facts'][0]['type'], 'episode')
        self.assertEqual(result['facts'][0]['sensitivity'], 'sensitive')
        self.assertIn('Alice lives in Paris.', result['facts'][0]['content'])
        amu_id = await add._persist_fact(
            self.st, req, result['facts'][0], (await embed(['fallback episode']))[0])
        self.assertEqual(
            self.st.get_amus_by_ids([amu_id], include_sensitive=True)[0]['sensitivity'],
            'sensitive')
        self.assertEqual(self.st.get_amus_by_ids([amu_id]), [])

    async def test_semantic_rejection_does_not_store_wrong_graph(self):
        fact = self.fact()
        fact['triples'][0]['object'] = 'Berlin'
        responses = [{'facts': [fact]}, {'valid': False, 'reason': 'triple contradicts fact'}]
        with patch.object(config, 'FAKE', False), patch.object(llm, 'complete_json', AsyncMock(side_effect=responses)):
            result = await add._extract(self.st, self.req(fact['content']))
        self.assertEqual(result['facts'][0]['type'], 'episode')
        self.assertEqual(result['facts'][0]['sensitivity'], 'sensitive')
        self.assertNotIn('triples', result['facts'][0])

    async def test_verified_evidence_keeps_original_quote(self):
        fact = self.fact()
        with patch.object(config, 'FAKE', False), patch.object(llm, 'complete_json', AsyncMock(
                side_effect=[{'facts': [fact]}, {'valid': True, 'reason': 'supported'}])):
            result = await add._extract(self.st, self.req(fact['content']))
        self.assertEqual(result['facts'][0]['evidence'][0]['request_id'], 'r')
        self.assertEqual(result['facts'][0]['evidence'][0]['quote'], fact['content'])

    async def test_missing_evidence_cannot_use_batch_as_proof(self):
        fact = self.fact()
        del fact['evidence']
        with patch.object(llm, 'complete_json', AsyncMock(return_value={'facts': [fact]})):
            result = await add._extract(self.st, self.req(fact['content']))
        self.assertEqual(result['facts'][0]['type'], 'episode')

    async def insert_state(self):
        return self.st.insert_amu(user_id='u', session_id='s', content='Alice lives in Paris.',
                                  type='profile', valid_from='2023-05-01',
                                  state=self.fact()['state'], embedding=(await embed(['Paris']))[0])

    async def test_matching_state_supersedes_atomically(self):
        target = await self.insert_state()
        req = self.req('Alice lives in Berlin.')
        self.st.save_messages(req)
        fact = self.fact(req.messages[0].content, 'Berlin')
        fact['temporal'] = {'start': '2023-06-01T00:00:00Z', 'precision': 'day'}
        with patch.object(add, '_govern_one', AsyncMock(return_value=('SUPERSEDE', {'target_id': target}))):
            aid = await add._persist_fact(self.st, req, fact, (await embed(['Berlin']))[0])
        self.assertEqual(self.st.get_amus_by_ids([aid])[0]['supersedes'], target)
        predecessor = self.st.conn.execute(
            'SELECT valid_to,superseded_by FROM amu WHERE id=?', (target,)).fetchone()
        self.assertIsNotNone(predecessor['valid_to'])
        self.assertEqual(predecessor['superseded_by'], aid)

    async def test_failed_add_rolls_back_valid_state_transition(self):
        target = await self.insert_state()
        req = self.req('Alice lives in Berlin.')
        fact = self.fact(req.messages[0].content, 'Berlin')
        fact['_sources'] = [0]
        before = list(self.st.conn.iterdump())
        with patch.object(add, '_extract', AsyncMock(return_value={'facts': [fact]})), patch.object(
                add, '_govern_one', AsyncMock(return_value=('SUPERSEDE', {'target_id': target}))), patch.object(
                add, '_update_summary', AsyncMock(side_effect=RuntimeError('summary failed'))):
            with self.assertRaises(RuntimeError):
                await add.run_add(self.st, req)
        self.assertEqual(before, list(self.st.conn.iterdump()))

    async def test_update_cannot_bypass_state_transition_guard(self):
        target = await self.insert_state()
        fact = self.fact('Alice lives in Berlin.', 'Berlin')
        with patch.object(add, '_govern_one', AsyncMock(return_value=('UPDATE', {
                'target_id': target, 'merged_content': fact['content']}))):
            aid = await add._persist_fact(self.st, self.req(fact['content']), fact, (await embed(['Berlin']))[0])
        self.assertIsNotNone(aid)
        self.assertEqual(self.st.get_amus_by_ids([target])[0]['content'], 'Alice lives in Paris.')

    async def test_unrelated_event_never_closes_old_event(self):
        target = self.st.insert_amu(user_id='u', session_id='s', content='Caroline attended a workshop.', type='event')
        fact = dict(content='Caroline attended an adoption meeting.', type='event',
                    state=dict(subject='Caroline', attribute='activity', value='adoption meeting'))
        with patch.object(add, '_govern_one', AsyncMock(return_value=('SUPERSEDE', {'target_id': target}))):
            aid = await add._persist_fact(self.st, self.req(fact['content']), fact, (await embed(['meeting']))[0])
        self.assertIsNone(self.st.get_amus_by_ids([aid])[0]['supersedes'])
        self.assertEqual(len(self.st.get_amus('u')), 2)

    async def test_different_state_attribute_cannot_supersede(self):
        old, new = self.fact(), self.fact(value='Berlin')
        new['state']['attribute'] = 'workplace'
        self.assertFalse(integrity.may_supersede(old, new))

    async def test_backdated_state_preserves_both(self):
        target = await self.insert_state()
        fact = self.fact('Alice lives in Berlin.', 'Berlin')
        fact['event_time'] = '2023-04-01T00:00:00+00:00'
        with patch.object(add, '_govern_one', AsyncMock(return_value=('SUPERSEDE', {'target_id': target}))):
            aid = await add._persist_fact(self.st, self.req(fact['content']), fact, (await embed(['Berlin']))[0])
        self.assertIsNone(self.st.get_amus_by_ids([aid])[0]['supersedes'])
        self.assertEqual(len(self.st.get_amus('u')), 2)

    async def test_invalid_interval_rejected_without_mutation(self):
        target = await self.insert_state()
        before = list(self.st.conn.iterdump())
        with self.assertRaises(ValueError):
            self.st.close_validity(target, '2023-04-01')
        self.assertEqual(before, list(self.st.conn.iterdump()))
        with self.assertRaises(ValueError):
            self.st.insert_amu(user_id='u', session_id='s', content='bad', valid_from='2023-06-26', valid_to='2023-06-01')
        self.assertEqual(before, list(self.st.conn.iterdump()))

    async def test_month_range_recall_and_answer_metadata(self):
        temporal = integrity.resolve_time('next month', '2023-08-28T15:19:00Z')
        aid = self.st.insert_amu(user_id='u', session_id='s', content='Talent show next month.', type='event', temporal=temporal)
        found = self.st.temporal_search('u', {'from': '2023-09-15', 'to': '2023-09-15'}, 10)
        self.assertEqual([a['id'] for a in found], [aid])
        self.assertIn('precision: month', _time_prefix(found[0]))
        self.assertEqual(self.st.temporal_search('u', {'from': '2023-10-01', 'to': '2023-10-02'}, 10), [])

    async def test_graph_edges_keep_fact_time_range(self):
        temporal = integrity.resolve_time('2023-09', None)
        aid = self.st.insert_amu(user_id='u', session_id='s', content='Alice visited Paris.',
                                 temporal=temporal)
        self.st.insert_triple('u', 'Alice', 'visited', 'Paris', aid,
                              temporal['start'], temporal['end'])
        edge = self.st.triples_for_user('u')[0]
        self.assertEqual(edge['valid_from'], temporal['start'])
        self.assertEqual(edge['valid_to'], temporal['end'])

    def test_relative_dates_and_precision(self):
        for reference, expression, expected in [
                ('2023-05-25', 'last Saturday', '2023-05-20'),
                ('2023-06-27', 'last Friday', '2023-06-23'),
                ('2023-10-22', 'last Friday', '2023-10-20'),
                ('2023-07-12', 'two days ago', '2023-07-10')]:
            self.assertTrue(integrity.resolve_time(expression, reference)['start'].startswith(expected))
        month = integrity.resolve_time('next month', '2023-08-28')
        self.assertTrue(month['start'].startswith('2023-09-01'))
        self.assertTrue(month['end'].startswith('2023-09-30'))
        self.assertEqual(month['precision'], 'month')
        self.assertEqual(integrity.resolve_time('a few weeks ago', '2023-08-28')['precision'], 'unknown')
        self.assertIsNone(integrity.resolve_time('yesterday', None)['start'])
        self.assertEqual(integrity.resolve_time('three years ago', '2023-06-09')['precision'], 'year')
        self.assertTrue(integrity.resolve_time('three years ago', '2023-06-09')['start'].startswith('2020-01-01'))

    async def test_wrong_time_expression_not_accepted(self):
        fact = self.fact()
        fact['time_expression'] = 'yesterday'
        with patch.object(llm, 'complete_json', AsyncMock(return_value={'facts': [fact]})):
            result = await add._extract(self.st, self.req(fact['content']))
        self.assertEqual(result['facts'][0]['type'], 'episode')

    async def test_formatting_only_quote_differences_are_accepted(self):
        source = 'Well\u2026 I\u2019m moving to  Paris \u2013 next week!'
        fact = self.fact('Alice is moving to Paris.')
        fact['evidence'] = [dict(message_index=0, quote="well... i'm moving to Paris - next week")]
        with patch.object(llm, 'complete_json', AsyncMock(return_value={'facts': [fact]})):
            result = await add._extract(self.st, self.req(source))
        self.assertEqual(result['facts'][0]['type'], 'profile')
        self.assertEqual(result['facts'][0]['_sources'], [0])
        self.assertEqual(result['facts'][0]['evidence'][0]['quote'],
                         "well... i'm moving to Paris - next week")

    async def test_paraphrased_quote_still_rejected(self):
        fact = self.fact('Alice lives in Paris.')
        fact['evidence'] = [dict(message_index=0, quote='Alice resides in Paris')]
        with patch.object(llm, 'complete_json', AsyncMock(return_value={'facts': [fact]})):
            result = await add._extract(self.st, self.req('Alice lives in Paris.'))
        self.assertEqual(result['facts'][0]['type'], 'episode')
        self.assertEqual(result['facts'][0]['sensitivity'], 'sensitive')

    async def test_bad_quote_only_shadows_its_cited_message(self):
        req = self.req('Alice lives in Paris.')
        req.messages.append(schemas.Message(role='user', content='Bob likes tea.'))
        req.messages.append(schemas.Message(role='user', content='Carol plays chess.'))
        good = self.fact('Bob likes tea.', value='tea')
        good['evidence'] = [dict(message_index=1, quote='Bob likes tea.')]
        bad = self.fact('Alice lives in Paris.')
        bad['evidence'] = [dict(message_index=0, quote='Alice resides in Paris')]
        with patch.object(llm, 'complete_json', AsyncMock(return_value={'facts': [bad, good]})):
            result = await add._extract(self.st, req)
        kinds = {f['type']: f for f in result['facts']}
        self.assertEqual(kinds['profile']['content'], 'Bob likes tea.')
        self.assertEqual(next(f for f in result['facts'] if f['type'] == 'episode' and f['_sources'] == [0])['_sources'], [0])
        self.assertNotIn('Carol', next(f for f in result['facts'] if f['_sources'] == [0])['content'])

    def test_cited_indices_follow_quote_to_real_message(self):
        batch = [schemas.Message(role='user', content='Where does Alice live?'),
                 schemas.Message(role='assistant', content='Alice lives in Paris.')]
        fact = dict(evidence=[dict(message_index=0, quote='Alice lives in Paris.')])
        self.assertEqual(integrity.cited_indices(fact, batch), [0, 1])
        self.assertEqual(integrity.cited_indices(dict(evidence=[dict(message_index=9, quote='x')]), batch), [])
        self.assertEqual(integrity.cited_indices(dict(), batch), [])

    def test_find_time_expressions(self):
        self.assertEqual(integrity.find_time_expressions(
            'I went to the group Yesterday and it was great'), ['yesterday'])
        self.assertEqual(integrity.find_time_expressions(
            'we camped last weekend and last week too'), ['last weekend', 'last week'])
        self.assertEqual(integrity.find_time_expressions('a few weeks ago; last night'), [])
        self.assertEqual(integrity.find_time_expressions('我们上周三见面'), ['上周三'])

    async def test_null_time_expression_backfilled_from_own_quote(self):
        text = 'I attended an LGBTQ support group yesterday.'
        fact = self.fact('Alice attended an LGBTQ support group.')
        fact.update(type='event', state=None, triples=[],
                    evidence=[dict(message_index=0, quote=text)])
        with patch.object(llm, 'complete_json', AsyncMock(return_value={'facts': [fact]})):
            result = await add._extract(self.st, self.req(text, timestamp=1683554160000))
        saved = result['facts'][0]
        self.assertEqual(saved['type'], 'event')
        self.assertEqual(saved['time_expression'], 'yesterday')
        self.assertEqual(saved['temporal']['precision'], 'day')
        self.assertTrue(saved['temporal']['start'].startswith('2023-05-07'))

    async def test_ambiguous_quote_times_are_not_backfilled(self):
        text = 'I painted this last year and framed it yesterday.'
        fact = self.fact('Alice painted a picture.')
        fact.update(type='event', state=None, triples=[],
                    evidence=[dict(message_index=0, quote=text)])
        with patch.object(llm, 'complete_json', AsyncMock(return_value={'facts': [fact]})):
            result = await add._extract(self.st, self.req(text))
        self.assertIsNone(result['facts'][0]['time_expression'])
        self.assertEqual(result['facts'][0]['temporal']['precision'], 'unknown')

    def test_quote_in_normalization(self):
        self.assertTrue(integrity.quote_in('\u201cLast Friday\u201d', 'we met last friday.'))
        self.assertTrue(integrity.quote_in('周五 见', '我们周五 见面'))
        self.assertFalse(integrity.quote_in('...', 'anything'))
        self.assertFalse(integrity.quote_in('Friday', 'we met last Thursday'))

    async def test_time_in_cited_message_expands_quote_without_mutation(self):
        text = 'Yesterday Alice moved to Paris.'
        fact = self.fact('Alice moved to Paris.')
        fact['time_expression'] = 'Yesterday'
        with patch.object(llm, 'complete_json', AsyncMock(return_value={'facts': [fact]})):
            result = await add._extract(self.st, self.req(text))
        saved = result['facts'][0]
        self.assertNotEqual(saved['type'], 'episode')
        self.assertEqual(saved['evidence'][0]['quote'], text)
        self.assertEqual(fact['evidence'][0]['quote'], 'Alice moved to Paris.')
        self.assertIsNotNone(saved['temporal']['start'])

    async def test_one_bad_time_preserves_other_facts(self):
        good = self.fact()
        bad = self.fact('Alice enjoys pottery.')
        bad['time_expression'] = 'yesterday'
        bad['evidence'][0]['message_index'] = 1
        req = self.req(good['content'])
        req.messages.append(schemas.Message(role='user', content=bad['content']))
        with patch.object(llm, 'complete_json', AsyncMock(return_value={'facts': [bad, good]})):
            result = await add._extract(self.st, req)
        self.assertEqual([f['type'] for f in result['facts']], ['profile', 'episode'])
        self.assertEqual(result['facts'][0]['content'], good['content'])
        self.assertEqual(result['facts'][1]['_sources'], [1])

    async def test_time_cannot_be_borrowed_from_uncited_message(self):
        fact = self.fact()
        fact['time_expression'] = 'Yesterday'
        req = self.req(fact['content'])
        req.messages.append(schemas.Message(role='user', content='Yesterday Bob left.'))
        with patch.object(llm, 'complete_json', AsyncMock(return_value={'facts': [fact]})):
            result = await add._extract(self.st, req)
        self.assertEqual(result['facts'][0]['type'], 'episode')

    async def test_semantic_rejection_isolated_to_one_fact(self):
        good = self.fact()
        bad = self.fact('Alice enjoys pottery.')
        bad['evidence'][0]['message_index'] = 1
        req = self.req(good['content'])
        req.messages.append(schemas.Message(role='user', content=bad['content']))
        responses = [{'facts': [good, bad]}, {'valid': False},
                     {'valid': True}, {'valid': False, 'reason': 'unsupported triple'}]
        with patch.object(add, '_segments', AsyncMock(return_value=[[0, 1]])), patch.object(config, 'FAKE', False), patch.object(
                llm, 'complete_json', AsyncMock(side_effect=responses)):
            result = await add._extract(self.st, req)
        self.assertEqual([f['type'] for f in result['facts']], ['profile', 'episode'])
        self.assertEqual(result['facts'][1]['_sources'], [1])

    async def test_partial_failure_uses_request_indices_in_later_batch(self):
        good = self.fact()
        bad = self.fact('Alice enjoys pottery.')
        bad['time_expression'] = 'yesterday'
        req = self.req(good['content'])
        req.messages.append(schemas.Message(role='user', content=bad['content']))
        with patch.object(config, 'EXTRACT_BATCH_MESSAGES', 1), patch.object(
                llm, 'complete_json', AsyncMock(side_effect=[{'facts': [good]}, {'facts': [bad]}])):
            result = await add._extract(self.st, req)
        self.assertEqual(result['facts'][1]['_sources'], [1])
        self.assertIn('pottery', result['facts'][1]['content'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
