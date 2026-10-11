"""A sibling's formatting error must not erase an already valid premise."""
import copy
import unittest
from unittest.mock import AsyncMock, patch

from selftest_answer_choice import memory, seal
from app import answer_choice as ac, config, llm


class RepairPremiseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')).start()
        patch.object(config, 'CHOICE_PRESERVE_REPAIR_PREMISES', True).start()
        self.options = ['A. Since you are interested in astronomy and enjoy detailed books, try a new guide.',
                        'B. Take a walk.']
        self.sources, _ = ac.build_catalog(seal(memory('Why do distant stars have different colors?')))
        self.original = dict(options=[dict(letter='A', kind='personal', claims=[
            dict(text='you are interested in astronomy', status='unsupported', premise_type='interest',
                 reason='no_source', citations=[]),
            dict(text='you enjoy detailed books', status='unsupported', premise_type='interest',
                 reason='no_source', citations=[])]), dict(letter='B', kind='generic', claims=[])])

    async def run_repair(self, enabled=True, repaired=None):
        replacement = repaired or dict(options=[dict(letter='A', kind='generic', claims=[])])
        model = AsyncMock(side_effect=[copy.deepcopy(self.original), replacement])
        diagnostics = {}
        prompt = 'Options:\n' + '\n'.join(self.options)
        with patch.object(config, 'CHOICE_PRESERVE_REPAIR_PREMISES', enabled), patch.object(llm, 'complete_json', model):
            entries = await ac.assess_support(prompt, ac.Assessments.model_json_schema(),
                                              self.options, self.sources, diagnostics)
        return entries, diagnostics, model

    async def test_generic_repair_keeps_valid_causal_clause_but_not_bad_sibling(self):
        entries, diagnostics, model = await self.run_repair()
        self.assertEqual(model.await_count, 2)
        self.assertEqual(entries[0]['kind'], 'personal')
        self.assertEqual([c['text'] for c in entries[0]['claims']], ['you are interested in astronomy'])
        self.assertEqual(entries[0]['claims'][0]['citations'], [])
        self.assertEqual(entries[0]['status'], 'unsupported')
        self.assertEqual(entries[1]['status'], 'generic')
        self.assertEqual(ac.eligible_choices(entries, set()), ['B'])
        self.assertEqual(diagnostics['answer_repair_preserved_premises'][0]['letter'], 'A')

    async def test_disabled_mode_preserves_existing_behavior(self):
        entries, diagnostics, _ = await self.run_repair(False)
        self.assertEqual(entries[0]['kind'], 'generic')
        self.assertEqual(entries[0]['claims'], [])
        self.assertNotIn('answer_repair_preserved_premises', diagnostics)

    async def test_repaired_enclosing_clause_is_not_duplicated(self):
        repaired = dict(options=[dict(letter='A', kind='personal', claims=[dict(
            text='Since you are interested in astronomy and enjoy detailed books',
            status='unsupported', premise_type='interest', reason='no_source', citations=[])])])
        entries, diagnostics, _ = await self.run_repair(repaired=repaired)
        self.assertEqual(len(entries[0]['claims']), 1)
        self.assertNotIn('answer_repair_preserved_premises', diagnostics)

    async def test_preservation_does_not_invent_a_source_or_entailed_verdict(self):
        entries, _, _ = await self.run_repair()
        checks = ac.entailment_checks(entries, self.sources, self.options)
        self.assertTrue(any(c['claim_id'] == 'A:option' for c in checks))
        self.assertFalse(any(c.get('entailment_verified') for c in entries[0]['claims']))
        # Subsequent semantic scope classification remains free to reject the
        # earlier claim; the preservation step never seals it as a fact.
        self.assertEqual(entries[0]['claims'][0]['proposed_status'], 'unsupported')

    def test_conditional_or_noncausal_advice_is_not_frozen(self):
        for option in ['A. If you are interested in astronomy, try a guide.',
                       'A. Since you might enjoy astronomy, try a guide.',
                       'A. Try a guide if you are interested in astronomy.']:
            claim = dict(text='you are interested in astronomy', status='unsupported',
                         validation_errors=[], option_span=dict(start=6, end=37), citations=[])
            original = dict(option=option, claims=[claim])
            repaired = dict(letter='A', option=option, kind='generic', claims=[], validation_errors=[])
            self.assertEqual(ac.preserve_repair_premises(original, repaired, {}), repaired)


if __name__ == '__main__':
    unittest.main()
