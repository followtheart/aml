"""Scope checks for cited claims must preserve support and claim mappings."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['AML_FAKE'] = '1'
_read = Path.read_text
with patch.object(Path, 'read_text', lambda p, *a, **kw:
                  '' if p.name == '.env' else _read(p, *a, **kw)):
    from app import answer_choice as ac, config


def entry():
    texts = ['The tournament offers exciting matches.', 'You already own a violin.']
    return dict(letter='A', option='A. ' + ' '.join(texts), kind='personal',
        status='supported', validation_status='valid', validation_errors=[], primary_claim=0,
        claims=[dict(text=t, status='supported', reason='none', premise_type='unknown',
            citations=[dict(source_id='s0', quote='unrelated private source', valid=True)],
            validation_errors=[], dropped_citations=[]) for t in texts])


class CitedScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_does_not_reclassify(self):
        original = entry()
        with patch.object(config, 'CHOICE_CITED_SCOPE', False), \
             patch.object(ac, '_judge', AsyncMock()) as judge:
            self.assertEqual(await ac.reclassify_uncited([original], {}), [original])
            judge.assert_not_called()

    async def test_scope_only_and_retained_support(self):
        original = entry()
        async def classify(prompt, schema, validate, *a, **kw):
            payload = json.loads(prompt.rsplit('\n', 1)[1])
            self.assertEqual({x['claim_id'] for x in payload}, {'A:0', 'A:1'})
            self.assertNotIn('unrelated private source', prompt)
            return validate({'claims': [{'claim_id': 'A:0', 'personal_fact': False},
                                         {'claim_id': 'A:1', 'personal_fact': True}]})
        diagnostics = {}
        with patch.object(config, 'CHOICE_CITED_SCOPE', True), \
             patch.object(ac, '_judge', side_effect=classify):
            updated = await ac.reclassify_uncited([copy.deepcopy(original)], diagnostics)
        self.assertEqual(updated[0]['claims'], [original['claims'][1]])
        self.assertEqual(diagnostics['answer_premise_scope']['claim_index_map'],
                         {'A:0': None, 'A:1': 'A:0'})

    async def test_failed_scope_preserves_original(self):
        original = entry()
        with patch.object(config, 'CHOICE_CITED_SCOPE', True), \
             patch.object(ac, '_judge', AsyncMock(side_effect=ValueError('invalid'))):
            self.assertEqual(await ac.reclassify_uncited([original], {}), [original])


if __name__ == '__main__':
    unittest.main()
