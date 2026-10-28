"""Exact source segmentation, attribution boundaries and runtime ID decoding."""
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context, budget, config, listwise_recovery as recovery, llm, recovery_spans as spans


class AbbreviationParts(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.req = SimpleNamespace(query='What affects travel plans?', options=[])

    def built(self, text, role='user'):
        cards = [dict(candidate_id='one', sources=[dict(source_id='s0', role=role, text=text)])]
        return spans.build(self.req, cards, abbreviations=True)

    def positive(self, parts, kind='user_context'):
        return dict(c0=dict(useful=True, target_id='question', evidence_kind=kind, citation_parts=parts))

    def test_exact_intervals_and_stable_ids(self):
        cases = [
            ('I follow U.S. automakers. I also cycle.', ['I follow U.S. automakers. ', 'I also cycle.'], [0, 2]),
            ('I leave at 5 p.m. since it gets cold. I take a coat.', ['I leave at 5 p.m. since it gets cold. ', 'I take a coat.'], [0, 2]),
            ('I consulted Dr. Chen. I rest now.', ['I consulted Dr. Chen. ', 'I rest now.'], [0, 2]),
            ('I bring supplies, e.g. Water and food. I pack lightly.', ['I bring supplies, e.g. Water and food. ', 'I pack lightly.'], [0, 2]),
            ('I visit the U.S. My brother lives abroad.', ['I visit the U.S. ', 'My brother lives abroad.'], [0, 1]),
            ('I visit the U.S.\nmy brother stays home.', ['I visit the U.S.\n', 'my brother stays home.'], [0, 1]),
            ('I leave at 5 p.m.\r\nsince it is late.', ['I leave at 5 p.m.\r\n', 'since it is late.'], [0, 1]),
            ('I paid 3.14 dollars. 我喜欢旅行。\n她不喜欢。', ['I paid 3.14 dollars. ', '我喜欢旅行。\n', '她不喜欢。'], [0, 1, 2]),
        ]
        for text, expected, ids in cases:
            with self.subTest(text=text):
                result = spans.source_parts(text, abbreviations=True)
                self.assertEqual([text[a:b] for _, a, b in result], expected)
                self.assertEqual([i for i, _, _ in result], ids)
                self.assertEqual(''.join(text[a:b] for _, a, b in result), text)
                self.assertEqual(result[0][1], 0)
                self.assertEqual(result[-1][2], len(text))

    def test_no_cross_sentence_attribution_or_role_bypass(self):
        cases = [
            ('Alex wrote: I collect U.S. stamps.', 'user', 's0.p0', 'not_actual_user_context'),
            ('Imagine I collected U.S. stamps.', 'user', 's0.p0', 'not_actual_user_context'),
            ('If I collect U.S. stamps, I will need an album.', 'user', 's0.p0', 'not_actual_user_context'),
            ('I collect U.S. stamps.', 'assistant', 's0.p0', 'not_user_source'),
            ('I travel to the U.S. My brother collects stamps.', 'user', 's0.p1', 'third_party_subject'),
            ('I collect U.S. stamps. The weather is warm.', 'user', 's0.p2', 'no_user_assertion'),
        ]
        for text, role, part, error in cases:
            with self.subTest(text=text, role=role):
                ids, judgments = spans.decode_partial(self.positive([part]), self.built(text, role))
                self.assertFalse(ids)
                self.assertIn(error, judgments[0]['validation_errors'])

    def test_owned_whole_quote_and_removed_parts(self):
        text = 'I do not buy U.S. stocks because I cannot afford them. I save cash.'
        built = self.built(text)
        ids, judgments = spans.decode_partial(self.positive(['s0.p0']), built)
        self.assertEqual(ids, {'one'})
        self.assertEqual(judgments[0]['citations'][0]['quote'], text.split(' I save')[0])
        for missing in ('s0.p1', 's9.p0'):
            ids, judgments = spans.decode_partial(self.positive([missing]), built)
            self.assertFalse(ids)
            self.assertEqual(judgments[0]['validation_errors'], ['span_reference_error'])
        # Later unchanged IDs must still decode to the same source text.
        ids, judgments = spans.decode_partial(self.positive(['s0.p2']), built)
        self.assertEqual(judgments[0]['citations'][0]['quote'], 'I save cash.')

    async def test_runtime_request_and_decoder_use_identical_parts_and_budget(self):
        text = 'I leave at 5 p.m. because I need to rest. I avoid late travel.'
        source = dict(role='user', content=text, request_id='fixture', message_index=0)
        row = dict(id='one', content='Travel context', _rank_text=answer_context.with_evidence('', [source]),
                   _packet_item=dict(sources=[source], personal_evidence=False))
        original = copy.deepcopy(row)
        async def model(prompt, **kwargs):
            budget.current.get().before_call()
            self.assertNotIn('s0.p1', kwargs['schema']['properties']['c0']['properties']['citation_parts']['items']['enum'])
            return self.positive(['s0.p0'])
        with patch.multiple(config, LISTWISE_SPAN_REFS=True, LISTWISE_ABBREVIATION_PARTS=True), \
                patch.object(llm, 'complete_json', model), budget.scope(seconds=30, calls=1, tokens=64000) as limits:
            selected, trace = await recovery._recover_batch(self.req, [row], [], {0})
        self.assertEqual(selected, [original])
        self.assertEqual(row, original)
        self.assertEqual(trace['segmentation_policy'], 'abbreviation_aware_v1')
        self.assertEqual(trace['restored_ids'], ['one'])
        self.assertEqual(trace['judgments'][0]['citations'][0]['quote'], 'I leave at 5 p.m. because I need to rest.')
        self.assertEqual((limits.calls, limits.reserved_calls, limits.reserved_tokens), (1, 0, 0))
        cards = [dict(candidate_id='one', sources=recovery.sources_for(row))]
        with patch.multiple(config, LISTWISE_SPAN_REFS=True, LISTWISE_ABBREVIATION_PARTS=False):
            before = recovery.request_for(self.req, cards)
            with patch.object(config, 'LISTWISE_ABBREVIATION_PARTS', True):
                after = recovery.request_for(self.req, cards)
        self.assertLess(after['reservation'], before['reservation'])
        self.assertEqual(before['system'], after['system'])


if __name__ == '__main__':
    unittest.main()
