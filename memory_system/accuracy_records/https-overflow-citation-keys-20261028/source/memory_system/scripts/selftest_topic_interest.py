"""Topic retrieval must preserve source attribution and factual boundaries."""
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context, budget, config, llm, listwise_recovery as recovery, recovery_spans as spans
from probe_topic_interest import GUIDANCE


class TopicInterestTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.req = SimpleNamespace(query='Which outdoor activity could suit warm weather?', options=['A. Try a local trail.'])
        self.text = 'Why do shaded paths feel cooler than nearby open fields?'
        self.cards = [dict(candidate_id='one', sources=[dict(source_id='s0', role='user', text=self.text)])]

    def result(self, kind='topic_interest', part='s0.p0'):
        return dict(c0=dict(useful=True, target_id='question', evidence_kind=kind, citation_parts=[part]))

    def test_request_matches_live_diagnostic_without_changing_sources_or_schema(self):
        self.assertEqual(GUIDANCE, recovery.TOPIC_INTEREST_GUIDANCE)
        original = copy.deepcopy(self.cards)
        for span_refs in (False, True):
            with patch.multiple(config, LISTWISE_SPAN_REFS=span_refs, LISTWISE_TOPIC_INTEREST=False):
                before = recovery.request_for(self.req, self.cards)
                with patch.object(config, 'LISTWISE_TOPIC_INTEREST', True):
                    after = recovery.request_for(self.req, self.cards)
            for key in ('prompt', 'schema', 'max_tokens'):
                self.assertEqual(before[key], after[key])
            self.assertEqual(after['system'], before['system'] + GUIDANCE)
            self.assertEqual(after['reservation'] - before['reservation'], len(GUIDANCE.encode()))
        self.assertEqual(original, self.cards)

    def test_curiosity_does_not_certify_personal_circumstances(self):
        built = spans.build(self.req, self.cards)
        self.assertEqual(spans.decode_partial(self.result(), built)[0], {'one'})
        kept, judgments = spans.decode_partial(self.result('user_context'), built)
        self.assertFalse(kept)
        self.assertIn('no_user_assertion', judgments[0]['validation_errors'])

    def test_wrong_role_other_speaker_and_wrong_owner_stay_rejected(self):
        for role, text, error in (
                ('assistant', self.text, 'not_user_source'),
                ('user', 'Alex wrote: ' + self.text, 'not_user_interest')):
            cards = copy.deepcopy(self.cards)
            cards[0]['sources'][0].update(role=role, text=text)
            kept, judgments = spans.decode_partial(self.result(), spans.build(self.req, cards))
            self.assertFalse(kept)
            self.assertIn(error, judgments[0]['validation_errors'])
        kept, judgments = spans.decode_partial(self.result(part='s8.p0'), spans.build(self.req, self.cards))
        self.assertFalse(kept)
        self.assertIn('span_reference_error', judgments[0]['validation_errors'])

    async def test_entry_restores_original_unit_without_factual_upgrade(self):
        source = dict(request_id='r', message_index=0, source_event_id='s', role='user', content=self.text)
        row = dict(id='one', content='A question about paths.',
                   _rank_text=answer_context.with_evidence('', [source]),
                   _packet_item=dict(sources=[source], personal_evidence=False), _supported_coverage_ids=[])
        original = copy.deepcopy(row)
        async def model(prompt, **kwargs):
            budget.current.get().before_call()
            self.assertTrue(kwargs['system'].endswith(GUIDANCE))
            return self.result()
        with patch.multiple(config, LISTWISE_SPAN_REFS=True, LISTWISE_TOPIC_INTEREST=True), \
                patch.object(llm, 'complete_json', model):
            with budget.scope(seconds=30, calls=1, tokens=40000) as limits:
                selected, trace = await recovery.recover(self.req, [row], [], {0})
        self.assertEqual(selected, [original])
        self.assertEqual(row, original)
        self.assertEqual(trace['topic_interest_policy'], 'question_topic_separate_from_premise_v1')
        self.assertEqual(trace['restored_ids'], ['one'])
        self.assertEqual((limits.calls, limits.reserved_calls, limits.reserved_tokens), (1, 0, 0))


if __name__ == '__main__':
    unittest.main()
