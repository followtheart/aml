"""Do not report incomplete or mismatched HTTPS runs as a complete comparison."""
import json
from pathlib import Path
import tempfile
import unittest

from compare_https_experiments import compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def run_fixture(self, name, scores, *, expected=None, terminal=True, dataset_hash='same'):
        directory = self.root / name
        path = directory / 'completed/data/results/personamem-v2-32k.jsonl'
        path.parent.mkdir(parents=True)
        rows = [dict(dataset='test', conversation_id='conversation', qa_id=str(qid),
                     transport='https', fake=False, run_id=name, protocol='proxy',
                     prediction='A', score=score, answer_source_catalog=[],
                     answer_eligible_options=['A', 'B']) for qid, score in scores]
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (directory / 'output.log').write_text(f'protocol=proxy qa={len(rows) if expected is None else expected}\n')
        if terminal:
            (directory / 'exit-code').write_text('0\n')
        (directory / 'funnel').mkdir()
        (directory / 'funnel/https-summary.json').write_text(json.dumps(dict(prepared_sha256=dataset_hash)))
        return directory

    def test_complete_pairs_count_wins_and_losses(self):
        a = self.run_fixture('a', [(0, 0), (1, 1)])
        b = self.run_fixture('b', [(0, 1), (1, 0)])
        result = compare(a, b)
        self.assertFalse(result['partial'])
        self.assertEqual((result['wins'], result['losses']), (1, 1))
        self.assertEqual(result['matched_questions'], 2)

    def test_active_run_requires_explicit_partial_mode(self):
        a = self.run_fixture('a', [(0, 0), (1, 1)])
        b = self.run_fixture('b', [(0, 1)], expected=2, terminal=False)
        with self.assertRaisesRegex(ValueError, 'still active'):
            compare(a, b)
        result = compare(a, b, allow_partial=True)
        self.assertTrue(result['partial'])
        self.assertEqual(result['matched_questions'], 1)
        self.assertEqual(result['matched_control_score'], 0)

    def test_stale_audit_counts_are_not_combined_with_new_results(self):
        a = self.run_fixture('a', [(0, 1), (1, 0)])
        b = self.run_fixture('b', [(0, 1), (1, 0)], expected=3, terminal=False)
        path = b / 'funnel/https-summary.json'
        audit = dict(run_id='b', count=1, score=1, prepared_sha256='same',
                     wrong_stages={'S6:selection': 0})
        path.write_text(json.dumps(audit))
        result = compare(a, b, allow_partial=True)
        self.assertIsNone(result['treatment']['wrong_stages'])
        self.assertFalse(result['treatment']['audit_current'])
        audit.update(count=2, wrong_stages={'S6:selection': 1})
        path.write_text(json.dumps(audit))
        result = compare(a, b, allow_partial=True)
        self.assertTrue(result['treatment']['audit_current'])
        self.assertEqual(result['treatment']['wrong_stages'], {'S6:selection': 1})

    def test_audit_from_another_experiment_is_rejected(self):
        a = self.run_fixture('a', [(0, 1)])
        b = self.run_fixture('b', [(0, 1)])
        (b / 'funnel/https-summary.json').write_text(json.dumps(dict(run_id='unrelated')))
        with self.assertRaisesRegex(ValueError, 'different experiment'):
            compare(a, b)

    def test_terminal_incomplete_run_is_not_complete(self):
        a = self.run_fixture('a', [(0, 1)])
        b = self.run_fixture('b', [(0, 1)], expected=2)
        with self.assertRaisesRegex(ValueError, 'complete planned evaluation'):
            compare(a, b)

    def test_duplicate_questions_are_rejected(self):
        a = self.run_fixture('a', [(0, 1)])
        b = self.run_fixture('b', [(0, 1), (0, 1)])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            compare(a, b)

    def test_different_question_sets_or_dataset_cannot_be_pooled(self):
        a = self.run_fixture('a', [(0, 1)])
        b = self.run_fixture('b', [(1, 1)])
        with self.assertRaisesRegex(ValueError, 'different question sets'):
            compare(a, b)
        c = self.run_fixture('c', [(0, 1)], dataset_hash='different')
        with self.assertRaisesRegex(ValueError, 'datasets differ'):
            compare(a, c)


if __name__ == '__main__':
    unittest.main()
