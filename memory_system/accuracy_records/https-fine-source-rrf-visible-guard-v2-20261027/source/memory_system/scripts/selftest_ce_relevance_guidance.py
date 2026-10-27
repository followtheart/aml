"""Guard against changing the question or silently adding/dropping clauses."""
import copy
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import probe_ce_relevance_guidance as probe


class GuidanceChecks(unittest.TestCase):
    def test_batches_preserve_utf8_long_records_blank_lines_and_final_record(self):
        rows = [{'text': '中文🙂' * 30}, {'text': 'last record without newline'}]
        payload = ('\n \n' + '\n\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)).encode()
        class BinaryStream(io.BytesIO):
            def __iter__(self):
                raise AssertionError('File iteration forbidden')
            def readline(self, *args):
                raise AssertionError('Line reads forbidden')
            def read(self, size=-1):
                if size != 7:
                    raise AssertionError('Must use fixed-size reads')
                return super().read(size)
        path = SimpleNamespace(open=lambda mode: BinaryStream(payload))
        actual = [r for batch in probe.json_batches(path, batch_bytes=7) for r in batch]
        self.assertEqual(actual, rows)

    def test_malformed_record_fails_instead_of_silently_skipping(self):
        path = SimpleNamespace(open=lambda mode: io.BytesIO(b'{"ok": 1}\n{broken}\n'))
        with self.assertRaises(json.JSONDecodeError):
            list(probe.json_batches(path, batch_bytes=4))

    def test_no_auxiliary_is_unchanged(self):
        question = '原问题：' + probe.OLD_SENTENCE
        self.assertEqual(probe.replace_guidance(question, question), question)

    def test_question_literal_negation_roles_and_all_clauses_preserved(self):
        question = 'Why? ' + probe.OLD_SENTENCE + '第三方\n'
        clauses = '\nPremise 1: You no longer cycle\nSubquestion: Does her sister cycle?'
        old = question + '\n\n' + probe.rerank_query._GUIDANCE + clauses
        new = probe.replace_guidance(question, old)
        self.assertTrue(new.startswith(question + '\n\n' + probe.NEW_SENTENCE))
        self.assertEqual(new[len(question + '\n\n' + probe.NEW_SENTENCE):],
                         old[len(question + '\n\n' + probe.OLD_SENTENCE):])
        self.assertTrue(new.endswith(clauses))

    def test_shorter_guidance_does_not_admit_previously_excluded_clause(self):
        req = SimpleNamespace(query='Question', options=['Since you cycle, you could walk.'])
        plan = {'_query_specs': [{'origin': 'sub_queries', 'text': 'Explore another topic'}]}
        first = '\nPremise 1: Since you cycle'
        second = '\nSubquestion: Explore another topic'
        limit = len(('\n\n' + probe.rerank_query._GUIDANCE + first + second).encode()) - 1
        with patch.object(probe.rerank_query, 'MAX_AUXILIARY_BYTES', limit):
            original = probe.rerank_query._base_query(req, plan)
            candidate = probe.replace_guidance(req.query, original)
        self.assertNotIn(second, original)
        self.assertNotIn(second, candidate)
        self.assertIn(first, candidate)

    def test_unexpected_header_rejected(self):
        with self.assertRaises(ValueError):
            probe.replace_guidance('Question', 'Question\n\nother header\nPremise 1: x')

    def test_historical_query_mismatch_rejected(self):
        trace = dict(query='Question', options=[], plan={}, query_specs=[],
                     cascade={'cross_encoder': {'query': 'Different question'}})
        original = copy.deepcopy(trace)
        with self.assertRaises(ValueError):
            probe.queries_for(trace)
        self.assertEqual(trace, original)


if __name__ == '__main__':
    unittest.main()
