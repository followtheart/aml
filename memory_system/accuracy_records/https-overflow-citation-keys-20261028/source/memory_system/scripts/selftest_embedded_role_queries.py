"""Regression boundaries for omitted roles in option evidence queries."""
import unittest
from app import retrieval_queries

class EmbeddedRoleQueries(unittest.TestCase):
    def query(self, option, planned):
        specs = retrieval_queries.build('What could I try next?', [option],
                                       {'option_queries': [planned]})
        return next(s['text'] for s in specs if s['option_index'] == 0)

    def test_embedded_role_survives_advice_rewrite(self):
        option = ('A. Choose a project that reflects your background as a botanist'
                  '—then invite friends to join.')
        self.assertEqual(self.query(option, 'evidence for choosing projects'),
                         'your background as a botanist')

    def test_following_condition_is_not_discarded(self):
        option = 'A. Draw on your work as an engineer—if you become one.'
        self.assertEqual(self.query(option, 'engineering work'), option[3:])

    def test_negative_and_fictional_scope_is_not_discarded(self):
        for option in ['A. Do not rely on your work as a physician.',
                       'A. Your work as a physician was fictional.']:
            with self.subTest(option=option):
                self.assertEqual(self.query(option, 'project suggestions'), option[3:])

    def test_long_conditional_does_not_lose_qualifier(self):
        option = ('A. Imagine your work as an engineer. ' + 'Consider a project. ' * 20)
        planned = 'Imagine your work as an engineer'
        self.assertEqual(self.query(option, planned), planned)

    def test_already_represented_role_is_not_rewritten(self):
        self.assertEqual(self.query('A. Draw on your work as an engineer.',
                                    'your work as an engineer'), 'your work as an engineer')

    def test_nonpersonal_as_phrase_is_unchanged(self):
        self.assertEqual(self.query('A. Use a vase as a centerpiece.',
                                    'vase centerpiece'), 'vase centerpiece')

    def test_existing_initial_premise_is_preserved(self):
        option = 'A. Since you own a telescope, you could visit an observatory.'
        self.assertEqual(self.query(option, 'visiting an observatory'),
                         'Since you own a telescope')

if __name__ == '__main__':
    unittest.main()
