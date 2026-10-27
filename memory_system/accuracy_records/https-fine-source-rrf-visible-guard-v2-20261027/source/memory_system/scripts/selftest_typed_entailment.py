"""Native typed verdict checks; no legacy response conversion."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice as gate
if os.environ.get('ANSWER_CANDIDATE'):
    spec = importlib.util.spec_from_file_location('app.typed_test', os.environ['ANSWER_CANDIDATE'])
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

class TypedTests(unittest.TestCase):
    def test_mapping(self):
        for kind in ('option', 'premise'):
            for verdict in ('personal_supported', 'no_personal_premise', 'unsupported_personal'):
                with self.subTest(kind=kind, verdict=verdict):
                    result = gate._typed_verdicts({'checks': [dict(claim_id='A', verdict=verdict)]},
                                                  [dict(claim_id='A', check_type=kind)])
                    self.assertEqual(result['checks'][0]['entailed'], verdict == 'personal_supported' or
                                     (kind == 'option' and verdict == 'no_personal_premise'))

    def test_invalid_payloads_rejected(self):
        for rows in ([], [dict(claim_id='B', verdict='personal_supported')],
                     [dict(claim_id='A', verdict='personal_supported')] * 2,
                     [dict(claim_id='A', verdict='invented')], [dict(claim_id='A', entailed=True)],
                     [dict(claim_id='A', verdict='personal_supported', extra=True)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                gate._typed_verdicts(dict(checks=rows), [dict(claim_id='A', check_type='option')])

    def test_whole_option_cannot_supply_missing_anchor(self):
        options = ['A. Since you own a canoe, paddle today.']
        entries = gate.validate_assessments(dict(options=[dict(letter='A', kind='personal', claims=[
            dict(text='you own a canoe', status='unsupported', premise_type='ownership',
                 reason='no_source', citations=[])])]), options, {})
        checks = [dict(claim_id='A:option', check_type='option', option=options[0])]
        for verdict in ('personal_supported', 'no_personal_premise'):
            result = gate.validate_entailments(gate._typed_verdicts(dict(checks=[
                dict(claim_id='A:option', verdict=verdict)]), checks), copy.deepcopy(entries), checks)
            self.assertEqual(result[0]['status'], 'unsupported')

    def test_generic_approval_and_hidden_premise_rejection(self):
        options = ['A. Take a walk.']
        entries = gate.validate_assessments(dict(options=[dict(letter='A', kind='generic', claims=[])]), options, {})
        checks = [dict(claim_id='A:option', check_type='option', option=options[0])]
        for verdict, status in [('no_personal_premise', 'generic'), ('unsupported_personal', 'unsupported')]:
            result = gate.validate_entailments(gate._typed_verdicts(dict(checks=[
                dict(claim_id='A:option', verdict=verdict)]), checks), copy.deepcopy(entries), checks)
            self.assertEqual(result[0]['status'], status)

    def test_legacy_review_still_accepts_boolean(self):
        payload = dict(checks=[dict(claim_id='A:option', entailed=False)])
        self.assertEqual(gate._verdicts(payload, [dict(claim_id='A:option')]), payload)

if __name__ == '__main__':
    unittest.main()
