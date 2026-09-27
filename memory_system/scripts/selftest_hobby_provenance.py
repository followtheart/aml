"""A topical question does not establish the user's claimed hobby."""
import unittest

from app import answer_choice


class HobbyProvenanceTests(unittest.TestCase):
    def check(self, quote, claim, *, declared=None):
        source = dict(id='s0', role='user', text=quote)
        if declared:
            source['declared'] = declared
        citation = answer_choice.Citation.model_validate(dict(
            source_id='s0', quote=quote, basis='self_report', subject='current_user'))
        return answer_choice._check_citation(citation, claim, {'s0': source}, 'interest')

    def test_abstract_questions_do_not_establish_a_hobby(self):
        claim = 'so guests get a sense of both your hobby and personality.'
        for quote in (
            'Why do some people display personal mementos in their living rooms?',
            'Could identity markers influence how visitors perceive the homeowner?',
        ):
            with self.subTest(quote=quote):
                result = self.check(quote, claim)
                self.assertFalse(result['valid'])
                self.assertIn('no_user_assertion', result['validation_errors'])

    def test_first_party_hobby_remains_valid(self):
        claim = 'your hobby of collecting model trains'
        self.assertTrue(self.check('My hobby is collecting model trains.', claim)['valid'])
        self.assertTrue(self.check('"hobbies_interests": ["Collecting model trains"]',
                                   claim, declared='persona')['valid'])

    def test_topical_interest_still_uses_questions(self):
        result = self.check('Why are model trains so popular?',
                            'you are interested in model trains')
        self.assertTrue(result['valid'])
        self.assertEqual(result['basis'], 'topic_interest')


if __name__ == '__main__':
    unittest.main()
