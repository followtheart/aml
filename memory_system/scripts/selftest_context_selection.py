"""Check final-selection guidance routing without changing earlier decisions."""
import copy
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice as gate
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


if __name__ == '__main__':
    unittest.main()
