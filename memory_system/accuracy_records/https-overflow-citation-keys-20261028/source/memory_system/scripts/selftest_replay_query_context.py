"""Frozen replay preserves provider-parse failures and their repair path."""
import unittest
from unittest.mock import patch

import selftest_answer_choice  # Offline provider setup.
from app import answer_choice, llm, metrics
from replay_query_context import recorded_response


class ReplayFailureTests(unittest.IsolatedAsyncioTestCase):
    def test_recorded_payload_is_not_mutated(self):
        call = dict(response={'options': []})
        value = recorded_response(call)
        value['options'].append('change')
        self.assertEqual(call['response']['options'], [])

    def test_unknown_failure_is_not_silently_replaced(self):
        with self.assertRaises(LookupError):
            recorded_response(dict(status='error', error_type='UnrecordedFailure'))

    async def test_structured_failure_reaches_existing_bounded_repair_path(self):
        call = dict(failure_phase='provider_parse', error_type='LLMError',
                    error_detail='Provider returned an invalid structured response')

        async def response(*args, **kwargs):
            return recorded_response(call)

        diagnostics = {}
        with patch.object(llm, 'complete_json', response):
            with self.assertRaisesRegex(ValueError, 'Malformed structured judgment'):
                await answer_choice._judge('prompt', {}, lambda r: r,
                                           'eval.choice_support', diagnostics, attempts=1)
        recorded = diagnostics['answer_calls'][0]
        self.assertEqual(recorded['failure_phase'], 'provider_parse')
        self.assertEqual(recorded['error_type'], 'LLMError')
        self.assertEqual(recorded['error_detail'], call['error_detail'])


if __name__ == '__main__':
    unittest.main()
