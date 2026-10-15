"""Failed reader arms must stay in the denominator and expose missing metadata."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from summarize_recovery_before_scope import summarize


class SummaryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.prefix = dict(stage='eval.choice_support', prompt_sha256='support', response={}, status='ok')
        original = dict(qa_id='q', arm='window', packet_hash='packet', diagnostics=dict(answer_calls=[
            self.prefix, dict(stage='eval.choice_premise_scope')]))
        snapshot = (json.dumps(original) + '\n').encode()
        (self.path / 'reader-snapshot.jsonl').write_bytes(snapshot)
        (self.path / 'plan.json').write_text(json.dumps(dict(questions=['q'], excluded=[], repetitions=1,
            snapshot_sha256=hashlib.sha256(snapshot).hexdigest())))
        self.rows = [dict(qa_id='q', enabled=enabled, repetition=1, packet_hash='packet',
            prediction='A', score=1, live_calls=[], diagnostics=dict(answer_calls=[copy.deepcopy(self.prefix)],
            answer_status='ok', answer_eligible_options=['A'],
            **({'answer_recovery_order': 'before_scope'} if enabled else {}))) for enabled in (False, True)]

    def report(self, partial=False):
        (self.path / 'q.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        return summarize(self.path, partial)

    def test_complete_pair(self):
        report = self.report()
        self.assertEqual((report['complete_pairs'], report['control'], report['treatment']), (1, 1, 1))
        self.assertEqual(report['unverified_orders'], [])

    def test_early_failure_kept_in_score_and_reported(self):
        row = self.rows[1]
        row.update(prediction='', score=0)
        row['diagnostics'].pop('answer_recovery_order')
        row['diagnostics'].pop('answer_eligible_options')
        row['diagnostics'].update(answer_status='error', error_stage='answer')
        report = self.report()
        self.assertEqual((report['complete_pairs'], report['treatment']), (1, 0))
        self.assertEqual(report['losses'], ['q'])
        self.assertEqual(len(report['errors']), 1)
        self.assertEqual(len(report['unverified_orders']), 1)

    def test_successful_wrong_order_rejected(self):
        self.rows[1]['diagnostics'].pop('answer_recovery_order')
        with self.assertRaisesRegex(ValueError, 'recovery order'):
            self.report()

    def test_prefix_drift_rejected_even_on_error(self):
        self.rows[1]['diagnostics']['answer_calls'][0]['response'] = {'changed': True}
        self.rows[1]['diagnostics']['answer_status'] = 'error'
        with self.assertRaisesRegex(ValueError, 'prefix drift'):
            self.report()

    def test_duplicate_arm_rejected(self):
        self.rows.append(copy.deepcopy(self.rows[1]))
        with self.assertRaisesRegex(ValueError, 'Duplicate arm'):
            self.report()

    def test_partial_pair_not_counted_as_complete(self):
        self.rows.pop()
        report = self.report(partial=True)
        self.assertTrue(report['partial'])
        self.assertEqual(report['complete_pairs'], 0)


if __name__ == '__main__':
    unittest.main()
