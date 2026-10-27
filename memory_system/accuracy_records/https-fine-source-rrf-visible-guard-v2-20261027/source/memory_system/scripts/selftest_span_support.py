"""Check compact support requests and per-option repair isolation offline."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice as gate

if os.environ.get('ANSWER_CANDIDATE'):
    spec = importlib.util.spec_from_file_location('app.answer_candidate', os.environ['ANSWER_CANDIDATE'])
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)


class SpanSupportTests(unittest.IsolatedAsyncioTestCase):
    async def test_compact_prompt_and_scoped_repair(self):
        options = ['A. Since you own a canoe, try the lake.', 'B. Try a quiet walk.']
        prompt = ('Question: Help me unwind.\nOptions:\n' + '\n'.join(options) +
                  '\n<sources>original source cards</sources>\nCORE TASK\nconflicting free-text example')
        seen = []

        async def respond(request, *, schema, **kwargs):
            seen.append((request, schema))
            claim = schema['$defs']['Claim']
            self.assertIn('original source cards', request)
            self.assertIn('Question: Help me unwind.', request)
            letters = schema['$defs']['Assessment']['properties']['letter']['enum']
            if len(seen) == 1:
                self.assertEqual(letters, ['A', 'B'])
                self.assertIn('text', claim['properties'])
                self.assertIn('conflicting free-text example', request)
                return {'options': [dict(letter='A', kind='personal', claims=[
                    dict(text='you possess a canoe', status='unsupported', reason='no_source',
                         premise_type='ownership', citations=[])]),
                    dict(letter='B', kind='generic', claims=[])]}
            self.assertEqual(letters, ['A'])
            self.assertNotIn('text', claim['properties'])
            self.assertIn('span_id', claim['required'])
            self.assertNotIn('conflicting free-text example', request)
            self.assertIn('source_id,quote,basis,subject', request)
            ids = claim['properties']['span_id']['enum']
            self.assertTrue(all(x.startswith('A:') for x in ids))
            return {'options': [dict(letter='A', kind='personal', claims=[
                dict(span_id='A:1', status='unsupported', reason='no_source',
                     premise_type='ownership', citations=[])])]}

        diagnostics = {}
        with patch.object(gate.llm, 'complete_json', respond):
            entries = await gate.assess_support(prompt, gate.Assessments.model_json_schema(),
                                               options, {}, diagnostics)
        self.assertEqual(len(seen), 2)
        self.assertEqual(entries[0]['claims'][0]['text'], 'Since you own a canoe,')
        self.assertEqual(entries[0]['status'], 'unsupported')
        self.assertEqual(entries[1]['status'], 'generic')
        self.assertEqual(diagnostics['answer_support_validation']['unresolved_options'], [])


if __name__ == '__main__':
    unittest.main()
