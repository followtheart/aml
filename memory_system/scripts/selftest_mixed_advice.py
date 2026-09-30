"""Routing preserves evidence and keeps ordinary prior facts on the original path."""
import copy
import unittest
from app import mixed_advice


class MixedAdviceTests(unittest.TestCase):
    def check(self, text, kind='premise'):
        return dict(claim_id='X:0', check_type=kind, claim=text,
                    sources=[dict(id='s0', quote='An original source quotation.')])

    def test_advice_with_embedded_personal_fact(self):
        for text in ['Use your experience as a teacher to plan the visit.',
                     'You could draw on your previous engineering work.',
                     'Try your own equipment, which you bought last year.']:
            with self.subTest(text=text):
                self.assertEqual(mixed_advice.mixed_advice_checks([self.check(text)]), ['X:0'])

    def test_no_change_to_claim_or_sources(self):
        checks = [self.check('Use your equipment, although you have never tried this activity.')]
        saved = copy.deepcopy(checks)
        self.assertEqual(mixed_advice.mixed_advice_checks(checks), ['X:0'])
        self.assertEqual(checks, saved)

    def test_ordinary_facts_and_hypotheses_keep_original_route(self):
        for text in ['Since you own a camera, try taking a photo.',
                     'Your former work involved engineering.',
                     'If you are an art enthusiast, consider a gallery.',
                     'Use of your equipment is restricted.',
                     'Consider a relaxing hobby.']:
            with self.subTest(text=text):
                self.assertEqual(mixed_advice.mixed_advice_checks([self.check(text)]), [])

    def test_only_premise_checks_trigger(self):
        self.assertEqual(mixed_advice.mixed_advice_checks([
            self.check('Use your professional expertise.', kind='option')]), [])


if __name__ == '__main__':
    unittest.main()
