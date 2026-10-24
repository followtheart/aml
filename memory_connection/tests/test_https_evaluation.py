"""Reject comparisons that quietly change the historical task."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from memory_connection.scripts.evaluate_https import plan


class HTTPSPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'prepared.jsonl'
        self.qa = dict(id='q', question='question', options=['A. a', 'B. b'])
        self.write(self.data, [dict(conversation_id='c', qa=[self.qa])])
        (self.root / 'isolated-manifest.json').write_text(json.dumps(dict(data=str(self.data),
            data_sha256=hashlib.sha256(self.data.read_bytes()).hexdigest())))
        self.results = self.root / 'completed/data/results/personamem-v2-32k.jsonl'
        self.results.parent.mkdir(parents=True)
        self.row = dict(conversation_id='c', qa_id='q', user_id='u')
        self.write(self.results, [self.row])
        self.requests = self.root / 'requests.jsonl'
        self.rows = [dict(path='add', payload=dict(user_id='u', messages=[{'role': 'user', 'content': 'fact'}])),
                     dict(path='search', payload=dict(user_id='u', query='question', options=self.qa['options']))]
        self.write(self.requests, self.rows)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, rows):
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_full_cohort_keeps_ingestion_and_question_identity(self):
        _, _, rows, groups = plan(self.root)
        self.assertEqual(rows, [self.row])
        self.assertEqual(groups['u']['adds'][0], self.rows[0]['payload'])
        self.assertEqual(groups['u']['questions'][0]['qa'], self.qa)

    def test_modified_input_is_rejected(self):
        self.data.write_text(self.data.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            plan(self.root)

    def test_changed_search_task_is_rejected(self):
        self.rows[1]['payload']['query'] = 'different question'
        self.write(self.requests, self.rows)
        with self.assertRaisesRegex(ValueError, 'identities differ'):
            plan(self.root)

    def test_partial_history_is_rejected(self):
        self.write(self.requests, self.rows[1:])
        with self.assertRaisesRegex(ValueError, 'Missing full-history'):
            plan(self.root)

    def test_duplicate_question_is_rejected(self):
        self.write(self.results, [self.row, self.row])
        self.write(self.requests, self.rows + self.rows[1:])
        with self.assertRaisesRegex(ValueError, 'duplicate baseline'):
            plan(self.root)


if __name__ == '__main__':
    unittest.main()
