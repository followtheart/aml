"""Complete-option scope checks cannot promote unsupported personal facts."""
import copy
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice as gate, config, llm


class TypedScopeTests(unittest.TestCase):
    def fixture(self):
        option = 'A. Try different frames so guests get a sense of your personal style.'
        payload = dict(options=[dict(letter='A', kind='personal', claims=[
            dict(text='your personal style', status='unsupported', reason='no_source', citations=[])])])
        entries = gate.validate_assessments(payload, [option], {})
        with patch.object(config, 'CHOICE_TYPED_SCOPE_CONSISTENCY', True):
            checks = gate.entailment_checks(entries, {}, [option])
        scope = {}
        verdicts = gate._typed_verdicts(dict(checks=[
            dict(claim_id=c['claim_id'], verdict='no_personal_premise') for c in checks]),
            checks, scope=scope)
        entries = gate.validate_entailments(verdicts, entries, checks)
        return entries, checks, scope, verdicts

    def normalize(self, fixture):
        return gate.normalize_typed_scopes(*fixture, {})

    def test_full_scope_agreement_removes_extraction_not_evidence(self):
        fixture = self.fixture()
        before = copy.deepcopy(fixture)
        entries = self.normalize(fixture)
        self.assertEqual(fixture, before)
        self.assertEqual(entries[0]['status'], 'generic')
        self.assertEqual(entries[0]['claims'], [])
        self.assertEqual(entries[0]['removed_claims'][0]['text'], 'your personal style')
        self.assertEqual(gate.eligible_choices(entries, set()), ['A'])
        self.assertEqual(gate.eligible_choices(entries, {'A'}), [])

    def test_uncited_claim_receives_full_option_context(self):
        entries, checks, _, _ = self.fixture()
        premise = next(c for c in checks if c['check_type'] == 'premise')
        self.assertEqual(premise['sources'], [])
        self.assertEqual(premise['option_context'], entries[0]['option'])

    def test_disabled_checks_keep_original_shape(self):
        entries, _, _, _ = self.fixture()
        entries[0]['claims'][0]['validation_errors'] = []
        with patch.object(config, 'CHOICE_TYPED_SCOPE_CONSISTENCY', False):
            checks = gate.entailment_checks(entries, {}, [entries[0]['option']])
        self.assertEqual(len(checks), 1)
        self.assertNotIn('option_context', checks[0])

    def test_whole_option_cannot_overrule_asserted_personal_claim(self):
        fixture = self.fixture()
        fixture[2]['A:0'] = 'unsupported_personal'
        self.assertEqual(self.normalize(fixture)[0]['status'], 'unsupported')

    def test_all_premises_required_even_when_whole_option_passes(self):
        entries, checks, scope, verdicts = self.fixture()
        del scope['A:0']
        with self.assertRaises(ValueError):
            gate.normalize_typed_scopes(entries, checks, scope, verdicts, {})

    def test_option_support_is_not_scope_reclassification(self):
        fixture = self.fixture()
        fixture[2]['A:option'] = 'personal_supported'
        self.assertEqual(self.normalize(fixture)[0]['status'], 'unsupported')

    def test_positive_typed_verdict_cannot_create_a_missing_anchor(self):
        entries, checks, _, _ = self.fixture()
        scopes = {}
        verdicts = gate._typed_verdicts(dict(checks=[
            dict(claim_id=c['claim_id'], verdict='personal_supported') for c in checks]),
            checks, scope=scopes)
        entries = gate.validate_entailments(verdicts, entries, checks)
        normalized = self.normalize((entries, checks, scopes, verdicts))
        self.assertEqual(normalized[0]['status'], 'unsupported')
        self.assertEqual(gate.eligible_choices(normalized, set()), [])

    def test_later_semantic_rejection_is_preserved(self):
        fixture = self.fixture()
        fixture[3]['checks'][0]['entailed'] = False
        self.assertEqual(self.normalize(fixture)[0]['status'], 'unsupported')

    def test_later_personal_premise_approval_prevents_erasure(self):
        fixture = self.fixture()
        fixture[3]['checks'][1]['entailed'] = True
        self.assertEqual(self.normalize(fixture)[0]['status'], 'unsupported')

    def test_invalid_citation_and_contradiction_not_erased(self):
        for field, value in [('dropped_citations', [{'errors': ['quote_not_in_source']}]),
                             ('reason', 'contradicted'), ('validation_errors', ['invalid_span'])]:
            with self.subTest(field=field):
                fixture = self.fixture()
                fixture[0][0]['claims'][0][field] = value
                self.assertEqual(self.normalize(fixture)[0]['status'], 'unsupported')

    def test_uncertain_event_child_prevents_reclassification(self):
        fixture = self.fixture()
        fixture[1].append(dict(claim_id='A:0:event:0', check_type='premise'))
        fixture[2]['A:0:event:0'] = 'unsupported_personal'
        fixture[3]['checks'].append(dict(claim_id='A:0:event:0', entailed=False))
        self.assertEqual(self.normalize(fixture)[0]['status'], 'unsupported')

    def test_unknown_final_check_rejected(self):
        fixture = self.fixture()
        fixture[3]['checks'].append(dict(claim_id='other', entailed=False))
        with self.assertRaises(ValueError):
            self.normalize(fixture)

    def test_unanimous_model_labels_cannot_erase_real_assertions(self):
        examples = [
            ('Since you enjoy documentaries, watch this film.', 'you enjoy documentaries'),
            ('Given your love of sailing, visit the marina.', 'your love of sailing'),
            ('You could display your stamp collection.', 'your stamp collection'),
            ('Lean on your close professional connections.', 'your close professional connections'),
            ('Because you visited the museum, share your memories.', 'you visited the museum'),
        ]
        for option, claim in examples:
            with self.subTest(option=option):
                entries = gate.validate_assessments(dict(options=[dict(letter='A', kind='personal',
                    claims=[dict(text=claim, status='unsupported', reason='no_source', citations=[])])]),
                    ['A. ' + option], {})
                with patch.object(config, 'CHOICE_TYPED_SCOPE_CONSISTENCY', True):
                    checks = gate.entailment_checks(entries, {}, ['A. ' + option])
                scopes = {}
                verdicts = gate._typed_verdicts(dict(checks=[dict(
                    claim_id=c['claim_id'], verdict='no_personal_premise') for c in checks]),
                    checks, scope=scopes)
                entries = gate.validate_entailments(verdicts, entries, checks)
                normalized = self.normalize((entries, checks, scopes, verdicts))
                self.assertEqual(normalized[0]['status'], 'unsupported')
                self.assertEqual(gate.eligible_choices(normalized, set()), [])

    def test_unlisted_possession_outside_effect_prevents_reclassification(self):
        option = 'A. Try using your camera so guests get a sense of your personal style.'
        entries = gate.validate_assessments(dict(options=[dict(letter='A', kind='personal',
            claims=[dict(text='your personal style', status='unsupported',
                         reason='no_source', citations=[])])]), [option], {})
        with patch.object(config, 'CHOICE_TYPED_SCOPE_CONSISTENCY', True):
            checks = gate.entailment_checks(entries, {}, [option])
        scopes = {}
        verdicts = gate._typed_verdicts(dict(checks=[dict(
            claim_id=c['claim_id'], verdict='no_personal_premise') for c in checks]),
            checks, scope=scopes)
        entries = gate.validate_entailments(verdicts, entries, checks)
        normalized = self.normalize((entries, checks, scopes, verdicts))
        self.assertEqual(normalized[0]['status'], 'unsupported')

    def test_generic_identity_prefix_does_not_hide_a_specific_qualifier(self):
        for suffix in [' in a particular country', ' as a former athlete', ' and political views']:
            with self.subTest(suffix=suffix):
                option = 'A. Try different frames so guests get a sense of your personal style' + suffix + '.'
                claim = dict(text='your personal style', validation_errors=[], reason='no_source',
                    option_span=dict(start=option.index('your personal style'),
                                     end=option.index('your personal style') + len('your personal style')))
                self.assertIsNone(gate._nonasserted_scope_reason(claim, option, complete_effect=True))


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_scope_agreement_is_wired_before_selection(self):
        qa = dict(question='How could I decorate a workspace?', options=[
            'A. Try different frames so guests get a sense of your personal style.',
            'B. Take a break.'])
        stages = []

        async def respond(prompt, **kwargs):
            stage = kwargs['stage']
            stages.append(stage)
            if stage == 'eval.choice_support':
                return dict(options=[
                    dict(letter='A', kind='personal', claims=[dict(
                        text='your personal style', status='unsupported', reason='no_source', citations=[])]),
                    dict(letter='B', kind='generic', claims=[])])
            if stage == 'eval.choice_premise_scope':
                return dict(claims=[dict(claim_id='A:0', personal_fact=True)])
            if stage == 'eval.choice_entailment':
                checks = json.loads(re.search(r'<checks>\s*(.*?)\s*</checks>', prompt, re.S)[1])
                self.assertEqual(next(c for c in checks if c['claim_id'] == 'A:0')['option_context'],
                                 qa['options'][0])
                return dict(checks=[dict(claim_id=c['claim_id'], verdict='no_personal_premise')
                                    for c in checks])
            if stage == 'eval.choice_select':
                self.assertEqual(kwargs['schema']['properties']['answer']['enum'], ['A', 'B'])
                return dict(answer='A')
            raise AssertionError('Unexpected provider stage: ' + stage)

        diagnostics = {}
        with patch.object(config, 'FAKE', False), \
                patch.object(config, 'CHOICE_TYPED_SCOPE_CONSISTENCY', True), \
                patch.object(config, 'CHOICE_ENTAILMENT_REVIEW', False), \
                patch.object(llm, 'complete_json', AsyncMock(side_effect=respond)), \
                patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
            prediction = await gate._answer(qa, [], diagnostics)
        self.assertEqual(prediction, 'A')
        self.assertEqual(diagnostics['choice_alignment'][0]['status'], 'generic')
        self.assertEqual(diagnostics['answer_typed_scope_consistency']['removed'][0]['claim_id'], 'A:0')
        self.assertEqual(stages, ['eval.choice_support', 'eval.choice_premise_scope',
                                  'eval.choice_entailment', 'eval.choice_select'])


if __name__ == '__main__':
    unittest.main()
