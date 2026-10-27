"""Check final-selection guidance routing without changing earlier decisions."""
import copy
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice as gate, config
from probe_choice_selection import BASELINE, CANDIDATE

if os.environ.get('ANSWER_CANDIDATE'):
    spec = importlib.util.spec_from_file_location('app.answer_candidate', os.environ['ANSWER_CANDIDATE'])
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)


class ContextSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_verified_eligible_personal_options_change_guidance(self):
        for status in ['generic', 'supported', 'partial', 'unsupported']:
            with self.subTest(status=status):
                options = ['A. Try a quiet walk.', 'B. Try a new book.', 'C. Use your canoe.']
                entries = [dict(letter=letter, option=option, kind='generic', status='generic',
                                claims=[], validation_errors=[], validation_status='valid')
                           for letter, option in zip('ABC', options)]
                entries[-1].update(kind='generic' if status == 'generic' else 'personal', status=status)
                eligible = gate.eligible_choices(entries, set())
                expected = CANDIDATE if status in ('supported', 'partial') else BASELINE

                async def choose(prompt, schema, validator, stage, diagnostics, **kwargs):
                    self.assertEqual(stage, 'eval.choice_select')
                    self.assertTrue(prompt.endswith(expected))
                    self.assertEqual(schema['properties']['answer']['enum'], eligible)
                    return validator({'answer': 'A'})

                with patch.object(gate, 'build_catalog', return_value=({}, {})), \
                     patch.object(gate.choice_witness, 'select', AsyncMock(return_value={})), \
                     patch.object(gate, 'assess_support', AsyncMock(return_value=copy.deepcopy(entries))), \
                     patch.object(gate, 'recover_missing_citations', AsyncMock(return_value=copy.deepcopy(entries))), \
                     patch.object(gate, 'entailment_checks', return_value=[]), \
                     patch.object(gate, 'constraint_pairs', return_value=[]), \
                     patch.object(gate, '_judge', choose):
                    self.assertEqual(await gate._answer(dict(question='How can I relax?', options=options), [], {}), 'A')

    async def test_independent_review_keeps_eligible_set_and_hides_first_answer(self):
        options = ['A. Try a quiet walk.', 'B. Since you own a canoe, paddle today.']
        sources = {'s0': dict(id='s0', role='user', text='I own a canoe.')}
        entries = [
            dict(letter='A', option=options[0], kind='generic', status='generic', claims=[],
                 validation_errors=[], validation_status='valid'),
            dict(letter='B', option=options[1], kind='personal', status='supported',
                 claims=[dict(text='you own a canoe', status='supported', entailment_verified=True,
                              validation_errors=[], citations=[
                                  dict(source_id='s0', quote='I own a canoe.', valid=True, anchor=True)])],
                 validation_errors=[], validation_status='valid')]
        before = copy.deepcopy((sources, entries))
        calls = []

        async def choose(prompt, schema, validator, stage, diagnostics, **kwargs):
            calls.append(stage)
            self.assertEqual(schema['properties']['answer']['enum'], ['A', 'B'])
            if stage == 'eval.choice_select':
                return validator({'answer': 'A'})
            self.assertEqual(stage, 'eval.choice_select_review')
            self.assertIn('No provisional answer is supplied', prompt)
            self.assertNotIn('First-pass eligible answer:', prompt)
            self.assertIn('I own a canoe.', prompt)
            return validator({'answer': 'B'})

        diagnostics = {}
        with patch.object(config, 'CHOICE_INDEPENDENT_SELECTION_REVIEW', True), \
             patch.object(gate, 'build_catalog', return_value=(sources, {})), \
             patch.object(gate.choice_witness, 'select', AsyncMock(return_value={})), \
             patch.object(gate, 'assess_support', AsyncMock(return_value=copy.deepcopy(entries))), \
             patch.object(gate, 'recover_missing_citations', AsyncMock(return_value=copy.deepcopy(entries))), \
             patch.object(gate, 'entailment_checks', return_value=[]), \
             patch.object(gate, 'constraint_pairs', return_value=[]), \
             patch.object(gate, '_judge', choose):
            result = await gate._answer(dict(question='What can I do outdoors?', options=options), [], diagnostics)
        self.assertEqual(result, 'B')
        self.assertEqual((sources, entries), before)
        self.assertEqual(calls, ['eval.choice_select', 'eval.choice_select_review'])
        self.assertEqual(diagnostics['answer_selection_review']['mode'], 'independent')
        self.assertEqual(diagnostics['answer_selection_review']['first'], 'A')

    async def test_structured_coverage_locally_selects_highest_request_fit(self):
        options = ['A. Provide the requested timeline.', 'B. Offer unrelated general tips.']
        entries = [dict(letter=letter, option=option, kind='generic', status='generic',
                        claims=[], validation_errors=[], validation_status='valid')
                   for letter, option in zip('AB', options)]
        calls = []

        async def choose(prompt, schema, validator, stage, diagnostics, **kwargs):
            calls.append(stage)
            self.assertEqual(stage, 'eval.choice_select')
            self.assertEqual(schema['properties']['answer']['enum'], ['A', 'B'])
            self.assertEqual(schema['$defs']['OptionCoverage']['properties']['letter']['enum'], ['A', 'B'])
            self.assertEqual(schema['$defs']['OptionCoverage']['properties']['met']['items']['enum'],
                             ['C1', 'C2', 'C3', 'C4', 'C5'])
            result = validator(dict(answer='B',
                criteria=[dict(id='C1', description='Provide a timeline', priority=3)],
                coverage=[dict(letter='A', met=['C1'], missed=[]),
                          dict(letter='B', met=[], missed=['C1'])]))
            self.assertEqual(result['answer'], 'A')
            return result

        diagnostics = {}
        with patch.object(config, 'CHOICE_STRUCTURED_REQUEST_SELECTION', True), \
             patch.object(gate, 'build_catalog', return_value=({}, {})), \
             patch.object(gate.choice_witness, 'select', AsyncMock(return_value={})), \
             patch.object(gate, 'assess_support', AsyncMock(return_value=copy.deepcopy(entries))), \
             patch.object(gate, 'recover_missing_citations', AsyncMock(return_value=copy.deepcopy(entries))), \
             patch.object(gate, 'entailment_checks', return_value=[]), \
             patch.object(gate, 'constraint_pairs', return_value=[]), \
             patch.object(gate, '_judge', choose):
            result = await gate._answer(dict(question='Provide a timeline.', options=options), [], diagnostics)
        self.assertEqual(result, 'A')
        self.assertEqual(calls, ['eval.choice_select'])
        self.assertEqual(diagnostics['answer_selection_rubric']['scores'], {'A': 3, 'B': -3})
        self.assertEqual(diagnostics['answer_eligible_options'], ['A', 'B'])

    def test_structured_coverage_requires_every_option_and_criterion_once(self):
        eligible = ['A', 'B']
        valid = dict(answer='B', criteria=[dict(id='C1', description='Match task', priority=2)],
                     coverage=[dict(letter='A', met=[], missed=['C1']),
                               dict(letter='B', met=['C1'], missed=[])])
        self.assertEqual(gate.validate_structured_request_selection(valid, eligible)['answer'], 'B')
        malformed = [
            {**valid, 'coverage': valid['coverage'][:1]},
            {**valid, 'coverage': [valid['coverage'][0], valid['coverage'][0]]},
            {**valid, 'coverage': [dict(letter='A', met=['C1'], missed=['C1']),
                                   valid['coverage'][1]]},
            {**valid, 'criteria': [dict(id='C2', description='Gap in numbering', priority=2)]},
            {**valid, 'coverage': [dict(letter='A', met=['C9'], missed=[]),
                                   valid['coverage'][1]]},
        ]
        for payload in malformed:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                gate.validate_structured_request_selection(payload, eligible)

    def test_structured_schema_constrains_coverage_to_criterion_identifiers(self):
        schema = gate.structured_request_selection_schema(['A', 'C'])
        self.assertEqual(schema['properties']['answer']['enum'], ['A', 'C'])
        self.assertEqual(schema['$defs']['OptionCoverage']['properties']['letter']['enum'], ['A', 'C'])
        for field in ('met', 'missed'):
            self.assertEqual(schema['$defs']['OptionCoverage']['properties'][field]['items']['enum'],
                             ['C1', 'C2', 'C3', 'C4', 'C5'])

    def test_structured_exact_ties_use_only_model_choice_among_maxima(self):
        payload = dict(answer='B', criteria=[dict(id='C1', description='Task', priority=1)],
                       coverage=[dict(letter='A', met=[], missed=['C1']),
                                 dict(letter='B', met=[], missed=['C1']),
                                 dict(letter='C', met=['C1'], missed=[])])
        self.assertEqual(gate.validate_structured_request_selection(payload, ['A', 'B', 'C'])['answer'], 'C')
        payload['coverage'][2] = dict(letter='C', met=[], missed=['C1'])
        result = gate.validate_structured_request_selection(payload, ['A', 'B', 'C'])
        self.assertEqual(result['tied_leaders'], ['A', 'B', 'C'])
        self.assertEqual(result['answer'], 'B')


class ReviewPromptTests(unittest.TestCase):
    def test_legacy_prompt_is_byte_equivalent(self):
        context = [dict(letter='B', source_id='s0', excerpt='Original text.')]
        import json
        expected = ('original prompt\nPotential first-party context matches (use as relevance context only):\n'
                    + json.dumps(context, ensure_ascii=False)
                    + '\nFocused relevance review: original focus Return only an eligible answer letter.'
                    + '\nFirst-pass eligible answer: A. This provisional answer is not evidence and has no '
                    'priority over the other eligible options. Compare all eligible options against the '
                    'current request and supplied sources, correcting the provisional answer whenever '
                    'another option fits better.')
        with patch.object(config, 'CHOICE_INDEPENDENT_SELECTION_REVIEW', False):
            self.assertEqual(gate.selection_review_prompt('original prompt', context, 'original focus', 'A'),
                             expected)

    def test_independent_prompt_does_not_depend_on_initial_letter_or_focus(self):
        context = [dict(source_id='s0', excerpt='I never owned a canoe.')]
        before = copy.deepcopy(context)
        with patch.object(config, 'CHOICE_INDEPENDENT_SELECTION_REVIEW', True):
            first = gate.selection_review_prompt('same sources and eligible choices', context, 'biased A', 'A')
            second = gate.selection_review_prompt('same sources and eligible choices', context, 'biased B', 'B')
        self.assertEqual(first, second)
        self.assertIn('I never owned a canoe.', first)
        self.assertNotIn('biased', first)
        self.assertEqual(context, before)

    def test_direct_request_guidance_is_opt_in_and_avoids_personalization_bias(self):
        with patch.object(config, 'CHOICE_DIRECT_REQUEST_SELECTION', False), \
             patch.object(config, 'CHOICE_STRUCTURED_REQUEST_SELECTION', False):
            generic = gate.direct_request_selection_instruction(False)
            personal = gate.direct_request_selection_instruction(True)
        self.assertTrue(generic.startswith('Generic options have no personal premise'))
        self.assertIn('specifically relevant, established user detail', personal)
        with patch.object(config, 'CHOICE_DIRECT_REQUEST_SELECTION', True):
            candidate = gate.direct_request_selection_instruction(True)
        self.assertIn('explicit deliverables, constraints, and intended outcome', candidate)
        self.assertIn('personal context as useful only when it materially helps', candidate)
        self.assertIn('unchanged eligible options', candidate)
        self.assertNotIn('prefer the one that uses', candidate)
        with patch.object(config, 'CHOICE_DIRECT_REQUEST_SELECTION', False), \
             patch.object(config, 'CHOICE_STRUCTURED_REQUEST_SELECTION', True):
            structured = gate.direct_request_selection_instruction(True)
        self.assertEqual(structured, candidate)


if __name__ == '__main__':
    unittest.main()
