"""Regression coverage for dropped option evidence and whole-list rerank fallback."""
import copy
import json
import time
import unittest
from memory_connection.app.config import Settings
from memory_connection.app.event_builder import build_events, valid_spans
from memory_connection.app.ranking import rank_candidates
from memory_connection.app.rerank_documents import documents
from memory_connection.app.references import reference_packet, decode_references
from memory_connection.app.schemas import AddRequest, SearchRequest, Span
from memory_connection.app.service import MemoryService
from memory_connection.app.store import Store
from memory_connection.app.verification import verify_candidates
from memory_connection.tests.test_pipeline import FixtureProvider


class EmptyRecallTests(unittest.IsolatedAsyncioTestCase):
    async def test_excess_valid_adjacent_refs_preserve_complete_evidence(self):
        event = build_events(AddRequest(request_id='r', user_id='u', session_id='s', messages=[
            {'role': 'user', 'content': ' '.join(f'Fact {i}.' for i in range(12))}]))[0]
        _, aliases, refs = reference_packet([event])
        row = {'event_id': 'e0', 'verdict': 'partial', 'evidence_refs': list(refs), 'reason': 'premises'}
        decoded = decode_references({'decisions': [row]}, aliases, refs, [event])['decisions'][0]
        self.assertEqual(decoded['evidence_spans'][0]['quote'], event['text'])
        self.assertTrue(valid_spans([Span(**s) for s in decoded['evidence_spans']], event))
        row['evidence_refs'].append('invented')
        invalid = decode_references({'decisions': [row]}, aliases, refs, [event])['decisions'][0]
        self.assertEqual(invalid, {'event_id': event['event_id']})

    async def test_excess_refs_cannot_import_other_event_evidence(self):
        events = build_events(AddRequest(request_id='r', user_id='u', session_id='s', messages=[
            {'role': 'user', 'content': 'Local fact.'},
            {'role': 'assistant', 'content': ' '.join(f'Other {i}.' for i in range(12))}]))
        _, aliases, refs = reference_packet(events)
        row = {'event_id': 'e0', 'verdict': 'relevant', 'evidence_refs': list(refs)}
        result = decode_references({'decisions': [row]}, aliases, refs, events)['decisions'][0]
        self.assertEqual(result, {'event_id': events[0]['event_id']})

    async def test_long_candidate_does_not_disable_other_candidates_reranking(self):
        events = build_events(AddRequest(request_id='r', user_id='u', session_id='s', messages=[
            {'role': 'user', 'content': '无关背景。'*1800},
            {'role': 'user', 'content': 'I volunteer at the chess club.'}]))
        before = copy.deepcopy(events)
        seen = []
        class Reranker:
            async def rerank(self, query, batch):
                self_query = 'chess club' in query
                assert self_query
                assert all(len(d.encode()) <= 600 for d in batch)
                seen.extend(batch)
                return [.99 if 'chess club' in d else .01 for d in batch]
        ranked, metrics, errors = await rank_candidates(Reranker(),
            {'query': 'What should I bring?', 'options': ['A. My chess club has a gathering.', 'B. beach picnic']},
            [{'event': e, 'score': 1, 'paths': []} for e in events],
            Settings(rerank_model='test', rerank_document_bytes=600, rerank_request_bytes=2400), time.monotonic()+5)
        self.assertFalse(errors)
        self.assertGreater(metrics['rerank_calls'], 1)
        self.assertEqual(ranked[0]['event']['event_id'], events[1]['event_id'])
        self.assertEqual(events, before)
        self.assertGreater(len(seen), 2)

    async def test_unicode_fragments_keep_all_original_text_and_roles(self):
        events = build_events(AddRequest(request_id='r', user_id='u', session_id='s', messages=[
            {'role': 'user', 'content': 'I do NOT own a car. 中文🙂\n'*50},
            {'role': 'assistant', 'content': 'You could rent a car. '*50}]))
        parts = [json.loads(d) for d in documents(events[1], 300)]
        for role, expected in [('user', events[0]['text']), ('assistant', events[1]['text'])]:
            subset = [p for p in parts if p['role'] == role]
            self.assertEqual(''.join(p['text'] for p in subset), expected)
            self.assertTrue(all(expected[p['start']:p['end']] == p['text'] for p in subset))
        self.assertTrue(all(len(json.dumps(p, ensure_ascii=False).encode()) <= 300 for p in parts))

    async def test_byte_split_verification_does_not_drop_all_later_evidence(self):
        events = build_events(AddRequest(request_id='r', user_id='u', session_id='s', messages=[
            {'role': 'user', 'content': 'original ' + str(i)} for i in range(4)]))
        class OnePerBatch(FixtureProvider):
            def verification_size(self, query, originals):
                return 200 * len(originals)
            async def verify(self, query, originals):
                result = await super().verify(query, originals)
                for row in result['decisions']:
                    if row['event_id'] != events[-1]['event_id']:
                        row.update(verdict='irrelevant', evidence_spans=[])
                return result
        decisions, errors, calls = await verify_candidates(OnePerBatch(), {},
            [{'event': e} for e in events], Settings(batch_bytes=256), time.monotonic()+5)
        self.assertEqual(calls, 4)
        self.assertFalse(errors)
        self.assertEqual(decisions[events[-1]['event_id']]['verdict'], 'relevant')
        self.assertTrue(valid_spans([Span(**s) for s in decisions[events[-1]['event_id']]['evidence_spans']], events[-1]))

    async def test_explicit_small_batch_budget_still_reports_omitted_candidates(self):
        events = build_events(AddRequest(request_id='r', user_id='u', session_id='s', messages=[
            {'role': 'user', 'content': str(i)} for i in range(3)]))
        class OnePerBatch(FixtureProvider):
            def verification_size(self, query, originals):
                return 200 * len(originals)
        _, errors, calls = await verify_candidates(OnePerBatch(), {}, [{'event': e} for e in events],
            Settings(batch_bytes=256, max_batches=2), time.monotonic()+5)
        self.assertEqual(calls, 2)
        self.assertEqual(errors[0]['reason'], 'batch_budget')

    async def test_option_route_survives_expansion_failure_without_inventing_evidence(self):
        class Provider(FixtureProvider):
            async def embed(self, texts):
                return [[1., 0., 0.] if 'chess' in t else [0., 1., 0.] for t in texts]
        provider = Provider()
        provider.fail_query = True
        store = Store(':memory:', provider.space_id)
        service = MemoryService(store, provider, Settings(memory_expansion=False, route_limit=1))
        try:
            await service.add(AddRequest(request_id='r', user_id='u', session_id='s', messages=[
                {'role': 'user', 'content': 'I organize chess tournaments.'}]))
            response = await service.search(SearchRequest(user_id='u', query='What should I bring?',
                options=['A. Bring snacks for chess players.', 'B. Bring beach towels.'], debug=True))
            self.assertEqual(response['data'][0]['content'], 'I organize chess tournaments.')
            self.assertTrue(any(v['facet'] == 'option:0' for v in response['trace']['query_views']))
            sources = [s['content'] for p in provider.seen for e in p['events'] for s in e['sources']]
            self.assertNotIn('A. Bring snacks for chess players.', sources)
        finally:
            store.close()


if __name__ == '__main__':
    unittest.main()
