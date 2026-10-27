"""Offline audit labels must not transfer source evidence between users."""
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch


with patch.object(sys, 'argv', ['audit_experiment_funnel.py', '--run', '.']):
    audit = runpy.run_path(str(Path(__file__).with_name('audit_experiment_funnel.py')),
                          run_name='audit_test')
gates = runpy.run_path(str(Path(__file__).with_name('audit_answer_gates.py')),
                      run_name='gate_test')


class ProbeIdentityTests(unittest.TestCase):
    def source(self, user, session=0, text='original'):
        return dict(request_id=f'{user}:session:{session}:chunk:1',
                    message_index=4, content=text)

    def test_same_chunk_label_belongs_to_distinct_users(self):
        first = self.source('local:dataset:first', text='first user')
        second = self.source('local:dataset:second', text='second user')
        index = audit['probe_source_index']([first, second])
        self.assertIs(audit['probe_source'](index, 'local:dataset:first', 'S1:4'), first)
        self.assertIs(audit['probe_source'](index, 'local:dataset:second', 'S1:4'), second)

    def test_missing_user_never_borrows_another_users_source(self):
        index = audit['probe_source_index']([self.source('local:dataset:first')])
        with self.assertRaisesRegex(ValueError, 'missing probe source'):
            audit['probe_source'](index, 'local:dataset:other', 'S1:4')

    def test_reused_labels_within_user_are_explicitly_ambiguous(self):
        index = audit['probe_source_index']([
            self.source('local:dataset:first', session=0),
            self.source('local:dataset:first', session=1)])
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            audit['probe_source'](index, 'local:dataset:first', 'S1:4')

    def test_same_question_id_is_scoped_to_conversation(self):
        rows = [dict(conversation_id='first', qa_id='0', probes=['first source']),
                dict(conversation_id='second', qa_id='0', probes=['second source'])]
        index = gates['indexed_probes'](rows, [])
        self.assertEqual(index[('first', '0')], ['first source'])
        self.assertEqual(index[('second', '0')], ['second source'])

    def test_unambiguous_legacy_audit_still_loads(self):
        rows = [dict(qa_id='0', probes=['original'])]
        results = [dict(conversation_id='first', qa_id='0')]
        self.assertEqual(gates['indexed_probes'](rows, results), {('first', '0'): ['original']})

    def test_legacy_ambiguous_or_duplicate_question_is_rejected(self):
        rows = [dict(qa_id='0', probes=[])]
        results = [dict(conversation_id='first', qa_id='0'),
                   dict(conversation_id='second', qa_id='0')]
        with self.assertRaisesRegex(ValueError, 'ambiguous question'):
            gates['indexed_probes'](rows, results)
        rows = [dict(conversation_id='first', qa_id='0', probes=[])] * 2
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            gates['indexed_probes'](rows, [])


if __name__ == '__main__':
    unittest.main()
