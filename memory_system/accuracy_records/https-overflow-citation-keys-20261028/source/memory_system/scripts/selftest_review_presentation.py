"""Structured renaming preserves exact citation ownership and validation."""
import copy
from types import SimpleNamespace
import unittest

from probe_review_presentation import presentation, request_for, spans, variants


class PresentationTests(unittest.TestCase):
    def setUp(self):
        req = SimpleNamespace(query='Where could I take a walk?', options=[])
        assistant = dict(source_id='s0', role='assistant', text='A shaded route may feel cooler.', request_id='r', message_index=1)
        user = dict(source_id='s0', role='user', text='Why are wooded trails cooler?', request_id='r', message_index=0)
        unrelated = dict(source_id='s0', role='user', text='Why do telescopes need mirrors?', request_id='r', message_index=2)
        self.cards = [dict(candidate_id='redundant', sources=[assistant]),
                      dict(candidate_id='target', sources=[user, dict(assistant, source_id='s1')]),
                      dict(candidate_id='other', sources=[unrelated])]
        self.old = spans.build(req, self.cards)
        self.new = spans.build(req, self.cards[1:])

    def test_composed_transform_exactly_reproduces_current_request_and_no_mutation(self):
        original = copy.deepcopy(self.old)
        result = variants(self.old, self.new)
        self.assertEqual(request_for(result['current']), request_for(self.new))
        self.assertEqual(self.old, original)
        self.assertEqual(len(result), 6)

    def test_every_variant_decodes_to_same_original_quote_and_local_owner(self):
        for name, built in variants(self.old, self.new).items():
            output = {key: dict(useful=False, target_id='', evidence_kind='none', citation_parts=[])
                      for key in built['candidate_map']}
            key = next(k for k, mid in built['candidate_map'].items() if mid == 'target')
            pid = next(pid for pid in built['owners'][key] if built['spans'][pid]['quote'] == 'Why are wooded trails cooler?')
            output[key] = dict(useful=True, target_id='question', evidence_kind='topic_interest', citation_parts=[pid])
            with self.subTest(name=name):
                selected, judgments = spans.decode_partial(output, built)
                self.assertEqual(selected, {'target'})
                judged = next(j for j in judgments if j['candidate_id'] == 'target')
                self.assertEqual(judged['citations'], [dict(source_id='s0', quote='Why are wooded trails cooler?')])
                wrong = next(pid for pid, span in built['spans'].items() if span['quote'] == 'Why do telescopes need mirrors?')
                output[key]['citation_parts'] = [wrong]
                selected, judgments = spans.decode_partial(output, built)
                self.assertFalse(selected)
                judged = next(j for j in judgments if j['candidate_id'] == 'target')
                self.assertIn('span_reference_error', judged['validation_errors'])

    def test_invalid_permutations_and_source_removal_rejected(self):
        with self.assertRaises(ValueError):
            presentation(self.old, source_order=['s0'])
        with self.assertRaises(ValueError):
            presentation(self.old, source_names={'s0':'s0','s1':'s0','s2':'s2'})
        changed = copy.deepcopy(self.new)
        changed['prompt'] = changed['prompt'].replace('wooded trails', 'city streets')
        with self.assertRaises(ValueError):
            variants(self.old, changed)


if __name__ == '__main__':
    unittest.main()
