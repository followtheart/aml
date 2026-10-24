"""The diagnostic wire label must preserve evidence and rejection behavior."""
import copy
import json
from types import SimpleNamespace
import unittest

from probe_context_label import OLD, NEW, proposal, decode, recovery, spans, config
from unittest.mock import patch


class ContextLabel(unittest.TestCase):
    def setUp(self):
        self.req = SimpleNamespace(query='What could help with loneliness?', options=['A. Find a group.'])
        self.cards = [dict(candidate_id='one', sources=[dict(source_id='u', role='user',
            text='Please edit this fictional story: Leon feels lonely. third_party_context is a label.',
            request_id='r', message_index=0)]), dict(candidate_id='two', sources=[dict(source_id='a',
            role='assistant', text='Meeting friends can help with loneliness.', request_id='r', message_index=1)])]
        self.built = spans.build(self.req, self.cards)
        with patch.multiple(config, LISTWISE_SPAN_REFS=True, LISTWISE_ROLE_BOUND_REFS=False,
                            LISTWISE_TRANSFORM_CONTEXT=False, LISTWISE_TOPIC_INTEREST=False,
                            LISTWISE_ABBREVIATION_PARTS=False):
            self.request = recovery.request_for(self.req, self.cards)

    def payload(self, kind=NEW, parts=None):
        return dict(c0=dict(useful=True, target_id='question', evidence_kind=kind,
                            citation_parts=['s0.p0'] if parts is None else parts),
                    c1=dict(useful=False, target_id='', evidence_kind='none', citation_parts=[]))

    def test_payload_and_contract_preserved(self):
        original = copy.deepcopy(self.request)
        after = proposal(self.request)
        self.assertEqual(self.request, original)
        self.assertEqual(after['prompt'].rsplit('\n', 1)[1], self.request['prompt'].rsplit('\n', 1)[1])
        self.assertIn(OLD, json.loads(after['prompt'].rsplit('\n', 1)[1])['sources'][0]['parts'][1]['text'])
        schema = copy.deepcopy(after['schema'])
        for decision in schema['properties'].values():
            decision['properties']['evidence_kind']['enum'] = [OLD if k == NEW else k
                for k in decision['properties']['evidence_kind']['enum']]
        self.assertEqual(schema, self.request['schema'])
        self.assertLessEqual(after['reservation'], self.request['reservation'])

    def test_context_alias_is_exactly_original_nonpersonal_validation(self):
        old = decode(self.payload(OLD), self.built, False)
        new = decode(self.payload(NEW), self.built, True)
        self.assertEqual(old, new)
        self.assertEqual(new[0], {'one'})
        self.assertEqual(new[1][0]['evidence_kind'], OLD)

    def test_bad_ownership_not_salvaged(self):
        restored, judgments = decode(self.payload(parts=['s1.p0']), self.built, True)
        self.assertFalse(restored)
        self.assertFalse(judgments[0]['valid'])

    def test_fiction_does_not_become_user_fact(self):
        restored, judgments = decode(self.payload('user_context'), self.built, True)
        self.assertFalse(restored)
        self.assertFalse(judgments[0]['valid'])

    def test_old_wire_label_and_extra_fields_rejected(self):
        for payload in (self.payload(OLD), dict(self.payload(), c9=self.payload()['c0'])):
            try:
                restored, judgments = decode(payload, self.built, True)
            except ValueError:
                continue
            self.assertFalse(restored)
            self.assertFalse(judgments[0]['valid'])


if __name__ == '__main__':
    unittest.main()
