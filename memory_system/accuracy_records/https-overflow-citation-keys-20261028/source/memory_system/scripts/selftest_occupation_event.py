"""Job metadata must not become evidence of unrelated life events."""
import unittest

from app import answer_choice


class OccupationEventTests(unittest.TestCase):
    def check(self, text, claim, premise_type, *, persona=True):
        source = dict(id='s0', text=text, role='user',
                      declared='persona' if persona else 'user')
        citation = answer_choice.Citation.model_validate(dict(
            source_id='s0', quote=text, basis='self_report', subject='current_user'))
        return answer_choice._check_citation(citation, claim, {'s0': source}, premise_type)

    def test_job_metadata_does_not_prove_a_specific_event(self):
        quote = ('"occupation": {\\n    "title": "High School Social Studies Teacher",'
                 '\\n    "employer": "Public School District",\\n')
        result = self.check(quote, 'After I mentored a student who left school unexpectedly,',
                            'experience')
        self.assertFalse(result['valid'])
        self.assertIn('occupation_profile_does_not_prove_event', result['validation_errors'])

    def test_occupation_and_actual_event_sources_remain_valid(self):
        job = ('"occupation": {\\n    "title": "High School Social Studies Teacher",'
               '\\n    "employer": "Public School District",\\n')
        self.assertTrue(self.check(job, 'I am a high school social studies teacher',
                                   'occupation')['valid'])
        self.assertTrue(self.check(job, 'After teaching in a school district,',
                                   'experience')['valid'])
        event = 'I mentored a student who left school unexpectedly.'
        self.assertTrue(self.check(event, 'I mentored a student who left school unexpectedly',
                                   'experience', persona=False)['valid'])
        self.assertTrue(self.check('"history": "I mentored a student who left school unexpectedly"',
                                   'I mentored a student who left school unexpectedly',
                                   'experience')['valid'])


if __name__ == '__main__':
    unittest.main()
