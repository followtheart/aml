"""Conditional-interest grammar boundaries; full-option verification stays required."""
import unittest
from app import answer_choice as ac

class ConditionalNounTests(unittest.TestCase):
    def entry(self, prefix):
        option = 'A. ' + prefix + ' consider a local event.'
        claim = dict(text=prefix, option_span=dict(start=3, end=3+len(prefix)),
                     status='unsupported', validation_errors=[], reason='no_source', citations=[])
        return dict(letter='A', kind='personal', status='unsupported', option=option,
                    claims=[claim], validation_errors=[], validation_status='valid', primary_claim=0)

    def test_conditional_enthusiast_is_not_an_asserted_history(self):
        e = self.entry('If you’re a music and theatre enthusiast,')
        out = ac.normalize_nonasserted_scopes([e], {})[0]
        self.assertEqual(out['kind'], 'generic')
        self.assertEqual(out['claims'], [])
        self.assertEqual(len(e['claims']), 1)

    def test_people_met_are_not_the_users_professional_role(self):
        e = self.entry('If you enjoy networking with professionals,')
        self.assertEqual(ac.normalize_nonasserted_scopes([e], {})[0]['claims'], [])

    def test_presupposed_facts_remain(self):
        for prefix in ['If you enjoy your professional work,',
                       'If you’re a professional photography enthusiast,',
                       'If you’re a culture enthusiast with your own collection,',
                       'Since you are a culture enthusiast,',
                       'If you enjoy networking again,']:
            with self.subTest(prefix=prefix):
                e = self.entry(prefix)
                self.assertEqual(ac.normalize_nonasserted_scopes([e], {})[0]['claims'], e['claims'])

    def test_complete_option_is_still_verified(self):
        e = self.entry('If you’re a music enthusiast,')
        e['option'] += ' Continue your daily professional performances.'
        entries = ac.normalize_nonasserted_scopes([e], {})
        checks = ac.entailment_checks(entries, {}, [e['option']])
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]['option'], e['option'])
        out = ac.validate_entailments({'checks':[{'claim_id':'A:option','entailed':False}]}, entries, checks)
        self.assertEqual(ac.eligible_choices(out, set()), [])

if __name__ == '__main__':
    unittest.main()
