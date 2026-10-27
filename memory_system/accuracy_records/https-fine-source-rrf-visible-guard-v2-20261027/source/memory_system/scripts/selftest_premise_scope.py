"""Offline tests for the undeployed premise-scope candidate."""
import copy
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import answer_choice as gate, budget

path = os.environ.get('SCOPE_CANDIDATE', str(ROOT / 'runs/iteration-20260924-13/candidate/premise_scope.py'))
spec = importlib.util.spec_from_file_location('scope_candidate', path)
scope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scope)


def entry(text='You could try walking every evening.'):
    return dict(letter='A', option='A. ' + text, kind='personal', status='unsupported', primary_claim=0,
        validation_status='valid', validation_errors=[], warnings=[], removed_claims=[], claims=[dict(
            text=text, status='unsupported', proposed_status='unsupported', premise_type='unknown',
            reason='no_source', citations=[], dropped_citations=[], validation_errors=[])])


def judge_for(payload):
    async def judge(prompt, schema, validator, stage, diagnostics, **kwargs):
        assert stage == 'eval.choice_premise_scope' and kwargs['attempts'] == 1
        return validator(payload)
    return judge


class ScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_index_mapping_preserves_verified_secondary_claim(self):
        original = [entry('You could walk. You own a canoe.')]
        original[0]['claims'][0]['text'] = 'You could walk.'
        verified = copy.deepcopy(original[0]['claims'][0])
        verified.update(text='You own a canoe.', status='supported', proposed_status='supported',
                        citations=[{'source_id': 's0', 'valid': True, 'anchor': True}])
        original[0]['claims'].append(verified)
        diagnostics = {}
        with patch.object(gate, '_judge', judge_for({'claims': [{'claim_id': 'A:0', 'personal_fact': False}]})):
            changed = await scope.reclassify_uncited(original, diagnostics)
        self.assertEqual(changed[0]['claims'], [verified])
        self.assertEqual(changed[0]['primary_claim'], 0)
        self.assertEqual(changed[0]['kind'], 'personal')
        self.assertEqual(diagnostics['answer_premise_scope']['claim_index_map'], {'A:0': None, 'A:1': 'A:0'})
        self.assertEqual(len(original[0]['claims']), 2)

    async def test_advice_removal_still_requires_whole_option_verification(self):
        original = [entry()]
        snapshot = copy.deepcopy(original)
        with patch.object(gate, '_judge', judge_for({'claims': [{'claim_id': 'A:0', 'personal_fact': False}]})):
            changed = await scope.reclassify_uncited(original, {})
        self.assertEqual(original, snapshot)
        self.assertEqual(changed[0]['kind'], 'generic')
        self.assertIsNone(changed[0]['primary_claim'])
        checks = gate.entailment_checks(changed, {}, [original[0]['option']])
        self.assertEqual([c['claim_id'] for c in checks], ['A:option'])
        rejected = gate.validate_entailments({'checks': [{'claim_id': 'A:option', 'entailed': False}]}, changed, checks)
        self.assertEqual(gate.eligible_choices(rejected, set()), [])

    async def test_real_personal_fact_stays_unsupported(self):
        original = [entry('Since you already own a canoe, try the lake.')]
        with patch.object(gate, '_judge', judge_for({'claims': [{'claim_id': 'A:0', 'personal_fact': True}]})):
            changed = await scope.reclassify_uncited(original, {})
        self.assertEqual(changed, original)

    async def test_ineligible_claims_and_unresolved_entries_are_untouched(self):
        variants = [dict(status='supported'), dict(citations=[{'source_id': 's0'}]),
                    dict(dropped_citations=[{'source_id': 's0'}]), dict(validation_errors=['bad_quote']),
                    dict(reason='contradicted')]
        for changes in variants:
            with self.subTest(changes=changes):
                original = [entry()]
                original[0]['claims'][0].update(changes)
                judge = AsyncMock()
                with patch.object(gate, '_judge', judge):
                    self.assertIs(await scope.reclassify_uncited(original, {}), original)
                judge.assert_not_called()
        original = [entry()]
        original[0]['validation_status'] = 'unresolved'
        with patch.object(gate, '_judge', AsyncMock()) as judge:
            self.assertIs(await scope.reclassify_uncited(original, {}), original)
            judge.assert_not_called()

    async def test_malformed_ids_and_timeout_keep_original(self):
        for payload in [{'claims': []}, {'claims': [{'claim_id': 'B:0', 'personal_fact': False}]},
                        {'claims': [{'claim_id': 'A:0', 'personal_fact': False}] * 2},
                        {'claims': [{'claim_id': 'A:0', 'personal_fact': 'false'}]}]:
            original, diagnostics = [entry()], {}
            with patch.object(gate, '_judge', judge_for(payload)):
                self.assertIs(await scope.reclassify_uncited(original, diagnostics), original)
            self.assertEqual(diagnostics['answer_premise_scope']['status'], 'kept_original')
        with patch.object(gate, '_judge', AsyncMock(side_effect=TimeoutError)):
            original = [entry()]
            self.assertIs(await scope.reclassify_uncited(original, {}), original)

    async def test_budget_reserves_later_stages(self):
        for limits in [dict(seconds=240, calls=5, tokens=128000),
                       dict(seconds=240, calls=20, tokens=1000), dict(seconds=30, calls=20, tokens=128000)]:
            original, diagnostics = [entry()], {}
            with budget.scope(**limits), patch.object(gate, '_judge', AsyncMock()) as judge:
                self.assertIs(await scope.reclassify_uncited(original, diagnostics), original)
                judge.assert_not_called()
            self.assertEqual(diagnostics['answer_premise_scope']['status'], 'skipped_budget')


if __name__ == '__main__':
    unittest.main()
