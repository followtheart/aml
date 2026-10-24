"""Reasoning text cannot bypass citation validation or become evidence."""
import copy
from types import SimpleNamespace
import unittest
from probe_review_connections import decode, spans


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        self.built = spans.build(SimpleNamespace(query='What might make a walk pleasant?', options=[]), [
            dict(candidate_id='one', sources=[dict(source_id='s0', role='user', text='Why are shaded paths cool?')])])
        self.answer = dict(c0=dict(relevance_reason='The question concerns shade, which may inform a walking route.',
                                  useful=True, target_id='question', evidence_kind='topic_interest', citation_parts=['s0.p0']))

    def test_reason_removed_only_from_validation_copy(self):
        before = copy.deepcopy(self.answer)
        kept, judged, reasons = decode(self.answer, self.built)
        self.assertEqual(kept, {'one'})
        self.assertEqual(self.answer, before)
        self.assertNotIn('relevance_reason', judged[0])
        self.assertEqual(reasons['c0'], before['c0']['relevance_reason'])
        self.assertEqual(judged[0]['citations'], [dict(source_id='s0', quote='Why are shaded paths cool?')])

    def test_missing_empty_or_overlong_reason_is_not_validated_positive(self):
        for value in (None, '', '   ', 'x'*241, 1):
            answer = copy.deepcopy(self.answer)
            if value is None:
                answer['c0'].pop('relevance_reason')
            else:
                answer['c0']['relevance_reason'] = value
            with self.subTest(value=value):
                kept, judged, _ = decode(answer, self.built)
                self.assertFalse(kept)
                self.assertFalse(judged[0]['valid'])

    def test_reason_does_not_fix_wrong_owner_or_certify_personal_fact(self):
        answer = copy.deepcopy(self.answer)
        answer['c0']['citation_parts'] = ['s9.p0']
        kept, judged, _ = decode(answer, self.built)
        self.assertFalse(kept)
        self.assertIn('span_reference_error', judged[0]['validation_errors'])
        answer = copy.deepcopy(self.answer)
        answer['c0']['evidence_kind'] = 'user_context'
        kept, judged, _ = decode(answer, self.built)
        self.assertFalse(kept)
        self.assertIn('no_user_assertion', judged[0]['validation_errors'])


if __name__ == '__main__':
    unittest.main()
