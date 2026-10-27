"""Repair retention preserves candidates, not truth or semantic scope."""
import copy
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['AML_FAKE'] = '1'
_read_text = Path.read_text
with patch.object(Path, 'read_text', lambda path, *a, **kw:
                  '' if path.name == '.env' else _read_text(path, *a, **kw)):
    from app import answer_choice as ac, config, llm


class ValidClaimRepairTests(unittest.IsolatedAsyncioTestCase):
    async def repair(self, enabled):
        options = ['A. Verify your license number TEST-24680, then use a secure channel.',
                   'B. Use a placeholder.']
        original = dict(options=[dict(letter='A', kind='personal', claims=[
            dict(text='your license number TEST-24680', status='unsupported',
                 premise_type='unknown', reason='no_source', citations=[]),
            dict(text='this paraphrase is absent', status='unsupported',
                 premise_type='unknown', reason='no_source', citations=[])]),
            dict(letter='B', kind='generic', claims=[])])
        replacement = dict(options=[dict(letter='A', kind='generic', claims=[])])
        diagnostics = {}
        with patch.object(config, 'CHOICE_REPAIR_VALID_CLAIMS', enabled), \
             patch.object(config, 'CHOICE_PRESERVE_REPAIR_PREMISES', False), \
             patch.object(llm, 'complete_json', AsyncMock(side_effect=[original, replacement])):
            entries = await ac.assess_support('Options:\n'+'\n'.join(options),
                ac.Assessments.model_json_schema(), options, {}, diagnostics)
        return entries, diagnostics

    async def test_valid_uncited_claim_survives_invalid_sibling(self):
        entries, diagnostics = await self.repair(True)
        self.assertEqual([c['text'] for c in entries[0]['claims']], ['your license number TEST-24680'])
        self.assertEqual(entries[0]['status'], 'unsupported')
        self.assertEqual(entries[0]['claims'][0]['citations'], [])
        self.assertEqual(entries[1]['kind'], 'generic')
        self.assertIn('answer_repair_preserved_premises', diagnostics)
        self.assertEqual(ac.eligible_choices(entries, set()), ['B'])

    async def test_default_off_preserves_old_behavior(self):
        entries, diagnostics = await self.repair(False)
        self.assertEqual(entries[0]['kind'], 'generic')
        self.assertNotIn('answer_repair_preserved_premises', diagnostics)

    async def test_later_scope_can_remove_retained_claim(self):
        entries, _ = await self.repair(True)
        with patch.object(ac, '_judge', AsyncMock(return_value={'A:0': False})):
            scoped = await ac.reclassify_uncited(copy.deepcopy(entries), {})
        self.assertEqual(scoped[0]['claims'], [])
        self.assertEqual(scoped[0]['kind'], 'generic')


if __name__ == '__main__':
    unittest.main()
