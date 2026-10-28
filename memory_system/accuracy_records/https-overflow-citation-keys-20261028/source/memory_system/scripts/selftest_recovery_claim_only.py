"""Premise-only retrieval must preserve source validation and verification gates."""
import asyncio
import copy
import json
import unittest
from unittest.mock import patch

from selftest_answer_choice import memory, seal
from app import answer_choice as ac, config
from probe_recovery_claim_only import focus


class RecoveryTests(unittest.TestCase):
    def run_recovery(self, enabled, quote=None):
        source = 'I brought a large potted palm into my office.'
        sources, _ = ac.build_catalog(seal(memory(source)))
        sid = next(iter(sources))
        claim = dict(text='You already own an indoor plant.', status='unsupported',
                     reason='no_source', premise_type='ownership', citations=[],
                     dropped_citations=[], validation_errors=[])
        entries = [dict(letter='C', option='C. Buy a peace lily because you already own an indoor plant.',
                        kind='personal', status='unsupported', primary_claim=0,
                        validation_status='valid', validation_errors=[], claims=[claim])]
        prompts = []

        async def judge(prompt, schema, validate, *args, **kwargs):
            prompts.append(prompt)
            return validate(dict(matches=[dict(claim_id='C:0', premise_type='ownership',
                supported=True, citations=[dict(source_id=sid, quote=source if quote is None else quote,
                                               basis='self_report', subject='current_user')])]))

        diagnostics = {}
        with patch.object(config, 'CHOICE_RECOVERY_CLAIM_ONLY', enabled), \
             patch.object(config, 'CHOICE_SOURCE_SCOPE', False), \
             patch.object(ac, '_judge', side_effect=judge):
            result = asyncio.run(ac.recover_missing_citations(copy.deepcopy(entries), sources,
                dict(question='What could brighten the room?'), diagnostics))
        return prompts[0], result, entries, diagnostics

    def test_runtime_prompt_matches_measured_intervention_exactly(self):
        original, _, _, _ = self.run_recovery(False)
        focused, _, _, _ = self.run_recovery(True)
        self.assertEqual(focused, focus(original))
        self.assertIn('I brought a large potted palm into my office.', focused)

    def test_recovered_ownership_still_requires_independent_verification(self):
        _, result, original, diagnostics = self.run_recovery(True)
        claim = result[0]['claims'][0]
        self.assertEqual(diagnostics['answer_citation_recovery']['attached'], ['C:0'])
        self.assertEqual(claim['citation_recovery'], 'awaiting_entailment')
        self.assertEqual(claim['status'], 'unsupported')
        self.assertEqual(result[0]['status'], 'unsupported')
        self.assertEqual(claim['text'], original[0]['claims'][0]['text'])

    def test_invented_quote_cannot_be_attached(self):
        _, result, original, diagnostics = self.run_recovery(True, 'I have owned orchids for ten years.')
        self.assertEqual(result, original)
        self.assertEqual(diagnostics['answer_citation_recovery']['attached'], [])
        self.assertEqual(diagnostics['answer_citation_recovery']['rejected'], ['C:0'])


if __name__ == '__main__':
    unittest.main()
