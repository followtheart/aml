"""Current input can be cited without becoming durable memory or trusted truth."""
import copy
import unittest
from unittest.mock import AsyncMock, patch

from selftest_answer_choice import memory, seal
from app import answer_choice as ac, choice_witness, config


class CurrentQuestionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')).start()
        patch.object(config, 'CHOICE_CURRENT_QUESTION_SOURCE', True).start()

    def cards(self, question):
        sources, constraints, diagnostics = {}, {}, {}
        ac.add_current_question(sources, constraints, question, diagnostics)
        return sources, constraints, diagnostics

    def test_exact_question_can_anchor_a_claim(self):
        question = 'I own a canoe. How should I store it?'
        sources, _, diagnostics = self.cards(question)
        citation = ac.Citation(source_id='q0', quote='I own a canoe.',
                               basis='self_report', subject='current_user')
        checked = ac._check_citation(citation, 'you own a canoe', sources, 'ownership')
        self.assertTrue(checked['anchor'])
        self.assertEqual(sources['q0']['text'], question)
        self.assertFalse(diagnostics['answer_current_question_source']['persisted'])
        self.assertNotIn('entailment_verified', checked)

    def test_hypothetical_and_third_party_are_not_self_reports(self):
        for question in ['Imagine I own a canoe. How should I store it?',
                         'My friend said: I own a canoe. How should they store it?']:
            sources, _, _ = self.cards(question)
            citation = ac.Citation(source_id='q0', quote='I own a canoe.',
                                   basis='self_report', subject='current_user')
            checked = ac._check_citation(citation, 'you own a canoe', sources, 'ownership')
            self.assertFalse(checked['anchor'])

    def test_forget_request_is_a_constraint_not_positive_evidence(self):
        sources, constraints, diagnostics = self.cards('Please forget that I own a canoe.')
        self.assertEqual(sources, {})
        self.assertEqual(constraints['q0']['source_scope'], 'current_request')
        self.assertEqual(diagnostics['answer_current_question_source']['status'], 'constraint')

    def test_default_path_unchanged(self):
        with patch.object(config, 'CHOICE_CURRENT_QUESTION_SOURCE', False):
            self.assertEqual(self.cards('I own a canoe.'), ({}, {}, {}))

    async def test_reader_sees_question_but_not_options_or_gold_as_source(self):
        packet = seal(memory('I enjoy quiet lakes.'))
        before = copy.deepcopy(packet)
        qa = dict(question='I own a canoe. How should I store it?',
                  options=['A. SECRET_OPTION', 'B. OTHER_OPTION'], answer='SECRET_GOLD')
        stop = AsyncMock(side_effect=RuntimeError('stop after source assembly'))
        with patch.object(choice_witness, 'select', stop):
            with self.assertRaisesRegex(RuntimeError, 'stop after source assembly'):
                await ac._answer(qa, packet, {})
        sources = stop.call_args.args[2]
        self.assertEqual(sources['q0']['text'], qa['question'])
        self.assertNotIn('SECRET_', str(sources))
        self.assertEqual(packet, before)


if __name__ == '__main__':
    unittest.main()
