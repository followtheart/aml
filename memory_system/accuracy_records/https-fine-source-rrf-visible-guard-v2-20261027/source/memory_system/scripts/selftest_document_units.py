"""Pasted documents stay whole: detection, Add-side episode, Search-side packing."""
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from listwise_fixture import from_scores
from app import add_pipeline as add, config, document_units, schemas, search_pipeline as search, store

RULES = "\n".join([
    "# Response policy for the support bot",
    "1. Always greet the customer by name when the ticket includes one.",
    "2. Never promise a refund before the warranty check in step 4 is complete.",
    "3. Output every reply in the format: SUMMARY / ACTION / NEXT STEP.",
    "4. Run the warranty check: purchase date must be within 24 months.",
    "5. If the warranty check fails, offer the paid repair option and stop.",
    "6. Escalate to a human agent when the customer mentions legal action.",
    "7. Do not disclose internal ticket identifiers to the customer.",
    "8. Close the ticket only after the customer confirms resolution.",
])
DOCUMENT = RULES + "\n\nAppendix: the paid repair option costs 49 EUR plus parts and takes about ten days. " * 12


def request(rid, *texts, session='s'):
    return schemas.AddRequest(request_id=rid, user_id='u', session_id=session, messages=[
        schemas.Message(role='user', content=t, timestamp=1684972800000 + i) for i, t in enumerate(texts)])


async def score_all(prompt, *args, **kwargs):
    ids = re.findall(r'^\d+:', prompt, re.M)
    return from_scores([0.9 for _ in ids])


class DetectionTests(unittest.TestCase):
    def test_structured_or_directive_text_is_a_document(self):
        self.assertTrue(document_units.is_document_like(DOCUMENT))
        self.assertTrue(document_units.is_document_like("```python\n" + "x = 1\n" * 200 + "```"))
        self.assertTrue(document_units.is_document_like(
            "The policy is strict. " * 100 + "You must never share keys. You should always rotate them. "
            "Output must follow the required format."))
        # Long but directive-free prose without structure stays conversational.
        self.assertFalse(document_units.is_document_like("The weather was lovely that afternoon. " * 60))

    def test_ordinary_chat_is_not_a_document(self):
        self.assertFalse(document_units.is_document_like("I started photography in March 2024."))
        self.assertFalse(document_units.is_document_like("We talked about my trip to Lisbon for a while. " * 30))
        self.assertFalse(document_units.is_document_like(None))
        with patch.object(config, 'DOCUMENT_UNITS_ENABLED', False):
            self.assertFalse(document_units.is_document_like(DOCUMENT))


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')

    def tearDown(self):
        self.st.conn.close()

    async def test_single_document_message_keeps_a_verbatim_episode(self):
        await add.run_add(self.st, request('r1', DOCUMENT))
        episodes = [a for a in self.st.get_amus('u') if a['type'] == 'episode']
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]['retrieval_key'], document_units.RETRIEVAL_KEY)
        self.assertIn(RULES, episodes[0]['content'])
        self.assertEqual(episodes[0]['compressed_content'], episodes[0]['content'])
        # An ordinary single message is still fully represented by its facts.
        await add.run_add(self.st, request('r2', 'Alice plays tennis with Bob.', session='s2'))
        plain = [a for a in self.st.get_amus('u') if a['type'] == 'episode'
                 and a['retrieval_key'] != document_units.RETRIEVAL_KEY]
        self.assertEqual(plain, [])

    async def test_search_packs_the_whole_document_but_ranks_an_excerpt(self):
        await add.run_add(self.st, request('r1', DOCUMENT))
        req = schemas.SearchRequest(user_id='u', query='What format must every reply use and when do we escalate?')
        with patch.object(config, 'SEARCH_ITEM_MAX_BYTES', 1200), patch.object(config, 'CE_MAX_DOCUMENT_BYTES', 1200), \
                patch.object(search, '_understand', AsyncMock(return_value={'intent': 'document', 'entities': []})), \
                patch.object(search.llm, 'complete_json', side_effect=score_all):
            response = await search.run_search(self.st, req)
        episode = next(item for item in response.data if item.memory_type == 'episode')
        for line in RULES.splitlines()[1:]:
            self.assertIn(line, episode.content)
        self.assertNotIn('[conversation excerpts', episode.content)
        trace_rank = next(c for c in response.coverage_manifest['included_ids'] if c == episode.id)
        self.assertEqual(trace_rank, episode.id)
        self.assertEqual(response.coverage_manifest['budget_source'], 'intent_tier')
        self.assertEqual(response.coverage_manifest['evidence_token_budget'], config.EVIDENCE_BUDGET_BY_INTENT['document'])

    async def test_rank_text_is_bounded_for_documents(self):
        await add.run_add(self.st, request('r1', DOCUMENT))
        episode = next(a for a in self.st.get_amus('u') if a['type'] == 'episode')
        req = schemas.SearchRequest(user_id='u', query='warranty check')
        plan = {'intent': 'document', '_coverage_requirements': []}
        with patch.object(config, 'SEARCH_ITEM_MAX_BYTES', 1200), patch.object(config, 'CE_MAX_DOCUMENT_BYTES', 1200):
            prepared = search._prepare_candidates(self.st, req, plan, [episode])
        self.assertEqual(len(prepared), 1)
        self.assertTrue(prepared[0]['_document'])
        self.assertLessEqual(len(prepared[0]['_rank_text'].encode('utf-8')), 1200)
        self.assertGreater(len(prepared[0]['_packet_item']['content']), 1200)
        self.assertIn('warranty check', prepared[0]['_rank_text'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
