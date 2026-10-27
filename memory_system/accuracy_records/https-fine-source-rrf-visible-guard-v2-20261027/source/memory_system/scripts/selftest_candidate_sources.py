"""Candidate-local source layout preserves evidence and original rejection rules."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from probe_candidate_sources import proposal, inline_request, decode, recovery, spans, config


class CandidateSources(unittest.TestCase):
    def setUp(self):
        self.req = SimpleNamespace(query='What helps loneliness?', options=['A. Join a group.'])
        self.source = dict(source_id='u', role='user', request_id='r', message_index=0,
            text='Please edit this fictional story: Leon feels lonely.\nHe has no friends.\n')
        self.cards = [dict(candidate_id='one', sources=[self.source]),
            dict(candidate_id='two', sources=[dict(source_id='a', role='assistant',
                text='Meeting friends can help with loneliness.', request_id='r', message_index=1)])]

    def request(self, cards):
        with patch.multiple(config, LISTWISE_SPAN_REFS=True, LISTWISE_ROLE_BOUND_REFS=False,
                            LISTWISE_TRANSFORM_CONTEXT=False, LISTWISE_TOPIC_INTEREST=False,
                            LISTWISE_ABBREVIATION_PARTS=False):
            return recovery.request_for(self.req, cards)

    def test_exact_evidence_and_contract(self):
        before = self.request(self.cards)
        frozen = copy.deepcopy(before)
        after = proposal(before)
        old = json.loads(before['prompt'].rsplit('\n', 1)[1])
        new = json.loads(after['prompt'].rsplit('\n', 1)[1])
        self.assertNotEqual(before, after)
        self.assertEqual(before, frozen)
        self.assertEqual(old['targets'], new['targets'])
        self.assertEqual(new['candidates'][0]['sources'][0], old['sources'][0])
        self.assertEqual(''.join(p['text'] for p in new['candidates'][0]['sources'][0]['parts']), self.source['text'])
        self.assertEqual(before['prompt'].rsplit('\n', 1)[0], after['prompt'].rsplit('\n', 1)[0])
        for key in ('schema', 'system', 'max_tokens'):
            self.assertEqual(before[key], after[key])
        self.assertLess(after['reservation'], before['reservation'])

    def test_shared_source_identity_and_budget_fallback(self):
        shared = dict(self.source, source_id='different-local-id')
        cards = [self.cards[0], dict(candidate_id='two', sources=[shared])]
        before = self.request(cards)
        raw = inline_request(before)
        data = json.loads(raw['prompt'].rsplit('\n', 1)[1])
        self.assertEqual(data['candidates'][0]['sources'], data['candidates'][1]['sources'])
        self.assertGreater(raw['reservation'], before['reservation'])
        self.assertEqual(proposal(before), before)

    def test_owned_citation_and_fiction_checks_unchanged(self):
        built = spans.build(self.req, self.cards)
        def payload(kind, parts):
            return dict(c0=dict(useful=True, target_id='question', evidence_kind=kind,
                citation_parts=parts), c1=dict(useful=False, target_id='', evidence_kind='none', citation_parts=[]))
        for p in (payload('third_party_context', ['s0.p0']),
                  payload('third_party_context', ['s1.p0']),
                  payload('user_context', ['s0.p0'])):
            self.assertEqual(decode(p, built, False), decode(p, built, True))
        self.assertEqual(decode(payload('third_party_context', ['s0.p0']), built, True)[0], {'one'})
        self.assertFalse(decode(payload('third_party_context', ['s1.p0']), built, True)[1][0]['valid'])
        self.assertFalse(decode(payload('user_context', ['s0.p0']), built, True)[1][0]['valid'])


if __name__ == '__main__':
    unittest.main()
