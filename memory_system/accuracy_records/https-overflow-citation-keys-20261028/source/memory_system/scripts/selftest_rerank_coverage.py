"""Offline source-local coverage witnesses and compact CE query contracts."""
import copy
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import search_coverage


def source(text, *, role='user', index=0, **extra):
    return dict(request_id='request', message_index=index, source_event_id=f'event-{index}',
                role=role, content=text, **extra)


def requirement(text, rid='option:0', origin='option_premise'):
    return dict(id=rid, text=text, origin=origin)


class WitnessTests(unittest.TestCase):
    def annotate(self, text, sources=(), *, summary='', **extra):
        row = dict(id='candidate', content=summary, _rank_text=summary, **extra)
        return search_coverage.annotate(row, [requirement(text)], sources)

    def test_exact_visible_self_statement_has_locatable_witness(self):
        text = 'I enjoy cycling along the river trail.'
        row = self.annotate('Since you enjoy cycling along the river trail', [source(text)])
        self.assertEqual(row.get('_supported_coverage_ids'), ['option:0'])
        witness = row['_coverage_witnesses']['option:0'][0]
        self.assertEqual(witness['request_id'], 'request')
        self.assertEqual(witness['message_index'], 0)
        self.assertEqual(witness['source_event_id'], 'event-0')
        self.assertEqual(witness['role'], 'user')
        self.assertEqual(text[witness['span']['start']:witness['span']['end']], witness['content'])
        self.assertIn('not entailment', witness['basis'])

    def test_student_source_matches_inflections_and_synonymous_forms(self):
        query = 'experience responding to students sharing troubling personal circumstances'
        for text in ['One of my students shared troubling personal circumstances with me.',
                     'One of my pupils disclosed distressing personal circumstances to me.']:
            with self.subTest(text=text):
                row = self.annotate(query, [source(text)])
                self.assertEqual(row.get('_supported_coverage_ids'), ['option:0'])

    def test_adjacent_antecedent_and_student_disclosure_form_one_local_witness(self):
        text = ("I had a conversation with one of my students this morning. "
                "During our discussion, they shared troubling personal circumstances that I'm not entirely sure how to address.")
        row = self.annotate('experience with being present for students sharing troubling matters', [source(text)])
        self.assertEqual(row.get('_supported_coverage_ids'), ['option:0'])
        witness = row['_coverage_witnesses']['option:0'][0]
        self.assertIn('students', witness['content'])
        self.assertIn('not entirely sure how', witness['content'])

    def test_uncertain_existence_is_not_a_personal_fact_witness(self):
        row = self.annotate('Since you have mild asthma', [source('I am not sure whether I have mild asthma.')])
        self.assertEqual(row.get('_supported_coverage_ids'), [])

    def test_other_speaker_and_fictional_frame_cannot_lend_first_person(self):
        for text in ['Claire said, "I enjoy cycling."',
                     'Imagine the following fictional diary. I enjoy cycling.',
                     'My friend said,\n"I enjoy cycling."']:
            with self.subTest(text=text):
                row = self.annotate('Since you enjoy cycling', [source(text)])
                self.assertEqual(row.get('_supported_coverage_ids'), [])

    def test_negation_cannot_move_to_a_different_activity(self):
        for text in ['I do not enjoy cycling, but I enjoy swimming.',
                     'I do not enjoy cycling. I enjoy swimming.']:
            with self.subTest(text=text):
                row = self.annotate('Since you do not enjoy swimming', [source(text)])
                self.assertEqual(row.get('_supported_coverage_ids'), [])

    def test_generic_social_words_do_not_witness_student_circumstances(self):
        query = 'experience responding to students sharing troubling personal circumstances'
        text = ('I enjoy sharing an experience on social media and responding to comments. '
                'Being present for friends online is useful.')
        row = self.annotate(query, [source(text)], summary=query)
        self.assertIn('option:0', row['_coverage_ids'])
        self.assertEqual(row.get('_supported_coverage_ids'), [])
        self.assertEqual(row.get('_coverage_witnesses'), {})

    def test_keywords_scattered_across_long_source_are_only_soft_coverage(self):
        query = 'gardening vegetables herbs balcony'
        text = ('I read about gardening. ' + 'A separate topic is music. ' * 30 +
                'I bought vegetables. ' + 'A separate topic is architecture. ' * 30 +
                'I studied herbs. ' + 'A separate topic is economics. ' * 30 +
                'I visited a balcony.')
        row = self.annotate(query, [source(text)])
        self.assertIn('option:0', row['_coverage_ids'])
        self.assertEqual(row.get('_supported_coverage_ids'), [])

    def test_assistant_explanation_is_not_personal_hard_support(self):
        row = self.annotate('Since you keep a well stocked spice rack',
                            [source('You keep a well stocked spice rack.', role='assistant')])
        self.assertIn('option:0', row['_coverage_ids'])
        self.assertEqual(row.get('_supported_coverage_ids'), [])

    def test_third_party_conditional_and_negated_claim_are_not_own_positive_claim(self):
        texts = ["My friend keeps a well stocked spice rack.",
                 'If I kept a well stocked spice rack, I could cook more.',
                 'I do not keep a well stocked spice rack.',
                 'Claire wrote: "I keep a well stocked spice rack."']
        for text in texts:
            with self.subTest(text=text):
                row = self.annotate('Since you keep a well stocked spice rack', [source(text)])
                self.assertEqual(row.get('_supported_coverage_ids'), [])

    def test_explicit_third_party_stays_matched_to_that_party_and_predicate(self):
        query = 'Since your daughter enjoys cycling'
        self.assertEqual(self.annotate(query, [source('My daughter enjoys cycling.')]).get(
            '_supported_coverage_ids'), ['option:0'])
        for text in ['My friend enjoys cycling.', 'My daughter read an article about cycling.']:
            self.assertEqual(self.annotate(query, [source(text)]).get('_supported_coverage_ids'), [])
        row = self.annotate('Since your daughter enjoys cycling along the river trail',
                            [source('My daughter read about cycling along the river trail.')])
        self.assertEqual(row.get('_supported_coverage_ids'), [])

    def test_matching_explicit_negation_can_have_a_witness(self):
        row = self.annotate('Since you do not eat meat', [source('I do not eat meat.')])
        self.assertEqual(row.get('_supported_coverage_ids'), ['option:0'])
        self.assertIn('not', row['_coverage_witnesses']['option:0'][0]['content'])

    def test_negated_ownership_and_preference_are_not_dropped_by_subject_guard(self):
        for query, text in [('Since you do not have a vegetable garden', 'I do not have a vegetable garden.'),
                            ('Since you do not enjoy cycling', 'I do not enjoy cycling.')]:
            with self.subTest(query=query):
                row = self.annotate(query, [source(text)])
                self.assertEqual(row.get('_supported_coverage_ids'), ['option:0'])

    def test_malformed_original_span_cannot_produce_a_false_locator(self):
        row = self.annotate('cycling along the river trail', [source(
            'I enjoy cycling along the river trail.',
            content_span=dict(start=100, end=102, original_length=102))])
        self.assertEqual(row.get('_supported_coverage_ids'), [])

    def test_past_claim_retains_time_and_does_not_become_current_habit(self):
        text = 'I used to enjoy cycling along the river trail.'
        current = self.annotate('Since you enjoy cycling along the river trail', [source(text)])
        historical = self.annotate('Since you used to enjoy cycling along the river trail', [source(text)])
        self.assertEqual(current.get('_supported_coverage_ids'), [])
        self.assertEqual(historical.get('_supported_coverage_ids'), ['option:0'])
        self.assertIn('used to', historical['_coverage_witnesses']['option:0'][0]['content'])

    def test_first_person_question_is_not_automatically_third_party(self):
        row = self.annotate('keeping an eye on your cholesterol', [source(
            'Why did my cholesterol change between checkups even though my diet stayed the same?')])
        self.assertEqual(row.get('_supported_coverage_ids'), ['option:0'])

    def test_visible_excerpt_coordinates_refer_to_original_source(self):
        visible = 'I enjoy cycling along the river trail.'
        row = self.annotate('cycling along the river trail', [source(visible,
            content_span=dict(start=117, end=117+len(visible), original_length=900))])
        witness = row.get('_coverage_witnesses', {}).get('option:0', [{}])[0]
        self.assertGreaterEqual(witness.get('span', {}).get('start', -1), 117)
        self.assertLessEqual(witness['span']['end'], 117+len(visible))
        self.assertEqual(witness['content'], visible[witness['span']['start']-117:witness['span']['end']-117])

    def test_summary_and_hidden_source_cannot_create_hard_support(self):
        for sources in [[], [source('', content_omitted='unit_budget')],
                        [dict(role='user', content='I enjoy cycling along the river trail.')]]:
            with self.subTest(sources=sources):
                row = self.annotate('cycling along the river trail', sources,
                                    summary='The user enjoys cycling along the river trail.')
                self.assertIn('option:0', row['_coverage_ids'])
                self.assertEqual(row.get('_supported_coverage_ids'), [])

    def test_reannotation_clears_obsolete_hard_witnesses(self):
        reqs = [requirement('cycling along the river trail')]
        row = dict(id='candidate', content='cycling along the river trail')
        search_coverage.annotate(row, reqs, [source('I enjoy cycling along the river trail.')])
        self.assertEqual(row.get('_supported_coverage_ids'), ['option:0'])
        search_coverage.annotate(row, reqs, [])
        self.assertEqual(row.get('_supported_coverage_ids'), [])
        self.assertEqual(row.get('_coverage_witnesses'), {})

    def test_witness_count_and_content_are_bounded(self):
        row = self.annotate('cycling along the river trail',
            [source('I enjoy cycling along the river trail.', index=i) for i in range(100)])
        witnesses = row.get('_coverage_witnesses', {}).get('option:0', [])
        self.assertGreater(len(witnesses), 0)
        self.assertLessEqual(len(witnesses), 2)
        self.assertTrue(all(len(w['content']) <= 480 for w in witnesses))

    def test_chinese_self_statement_keeps_historical_qualifier(self):
        row = self.annotate('過去有輕微氣喘', [source('我過去有輕微氣喘。')])
        self.assertEqual(row.get('_supported_coverage_ids'), ['option:0'])


class QueryTests(unittest.TestCase):
    def test_explicit_approach_advice_keeps_interest_and_qualifications(self):
        value = self.build(options=[
            'A. Given your interest in birdwatching, a good approach is to visit a wetland.',
            'B. Since you enjoy hiking, but no longer climb steep trails, one useful strategy would be to choose a flat route.'])
        self.assertIn('interest in birdwatching', value)
        self.assertIn('but no longer climb steep trails', value)
        self.assertNotIn('wetland', value)
        self.assertNotIn('flat route', value)

    def test_approach_without_personal_prefix_stays_generic(self):
        self.assertEqual(self.build(options=[
            'A. A good approach is to compare several maps before traveling.']), 'q')

    def build(self, query='q', options=None, plan=None):
        module = importlib.import_module('app.rerank_query')
        return module.build(SimpleNamespace(query=query, options=options or []), plan or {})

    def test_no_auxiliary_context_returns_original_query_exactly(self):
        self.assertEqual(self.build(query='q'), 'q')
        self.assertEqual(self.build(query='保留原問句？\n'), '保留原問句？\n')

    def test_option_advice_is_removed_and_premises_remain_hypotheses(self):
        value = self.build(query='Which outdoor activity fits?', options=[
            'A. Since you enjoy cycling, you might ride before sunrise and pack a picnic.',
            'B. Since your daughter had mild asthma in childhood, you could check the weather.',
            'C. If you have a small garden, you could install colorful planters.',
            'D. Try walking in a nearby park and taking a water bottle.'])
        self.assertTrue(value.startswith('Which outdoor activity fits?'))
        for wanted in ['enjoy cycling', 'your daughter', 'in childhood', 'If you have a small garden']:
            self.assertIn(wanted, value)
        for advice in ['before sunrise', 'pack a picnic', 'check the weather', 'colorful planters', 'water bottle']:
            self.assertNotIn(advice, value)
        self.assertIn('not established facts', value)
        self.assertIn('negation', value)

    def test_negation_is_preserved_in_option_premise(self):
        value = self.build(options=['A. Since you do not eat meat, you could try lentil soup.'])
        self.assertIn('do not eat meat', value)
        self.assertNotIn('lentil soup', value)

    def test_planner_cannot_replace_option_premise_with_its_advice(self):
        value = self.build(options=['A. Since you cycle along the river, you could buy a titanium bike.'],
            plan={'option_queries':['buy a titanium bike']})
        self.assertIn('cycle along the river', value)
        self.assertNotIn('titanium', value)

    def test_ambiguous_advice_tail_is_omitted_instead_of_sent_whole(self):
        value = self.build(options=[
            'A. Since you enjoy cycling, a morning ride can be refreshing.',
            'B. Since you enjoy cooking, but not baking, I would recommend a soup workshop.'])
        self.assertNotIn('morning ride', value)
        self.assertNotIn('soup workshop', value)
        self.assertIn('not baking', value)

    def test_recommendation_without_clause_punctuation_is_not_copied(self):
        value = self.build(options=['A. Since you enjoy cycling you could take a sunrise ride.'])
        self.assertNotIn('sunrise ride', value)

    def test_qualifying_user_clause_is_not_an_advice_boundary(self):
        value = self.build(options=[
            'A. Since you have asthma, you no longer have symptoms, consider a light walk.',
            'B. Since you enjoy cycling, you stopped last year because of injury, consider a short walk.'])
        self.assertIn('you no longer have symptoms', value)
        self.assertIn('you stopped last year because of injury', value)
        self.assertNotIn('consider a light walk', value)
        self.assertNotIn('consider a short walk', value)

    def test_invalid_optional_plan_fields_do_not_break_plain_query(self):
        self.assertEqual(self.build(plan={'sub_queries': None}), 'q')

    def test_subquestions_are_bounded_deduplicated_and_do_not_mutate_plan(self):
        plan = {'_query_specs': [dict(origin='sub_queries', text='Who owns the bicycle?')]*10 +
                [dict(origin='expanded_queries', text='Unrelated recommendation')]}
        original = copy.deepcopy(plan)
        value = self.build(query='Who repaired it?', plan=plan)
        self.assertEqual(value.count('Who owns the bicycle?'), 1)
        self.assertNotIn('Unrelated recommendation', value)
        self.assertEqual(plan, original)

    def test_total_auxiliary_text_is_bounded_without_rewriting_question(self):
        question = 'What activity fits my schedule?'
        options = [f'{i}. Since you enjoy ' + 'unusual creative activities ' * 200 +
                   ', you could buy many things.' for i in range(40)]
        value = self.build(query=question, options=options)
        self.assertTrue(value.startswith(question))
        self.assertLessEqual(len(value.encode('utf-8')), len(question.encode('utf-8'))+1800)
        self.assertNotIn('buy many things', value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
