"""Check reader-scoped lifecycle priority without altering source units."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, answer_context, evidence_packet, schemas
from fine_direct_lifecycle import prioritize


def candidate(mid, text, role='user', *, request_id=None, visible=True, body='Conversation excerpt:'):
    sources = [dict(request_id=request_id or mid, message_index=0, role=role, timestamp=1700000000000,
                    content=text)]
    content = answer_context.with_evidence(body, sources) if visible else body
    return dict(id=mid, _ce_score=.5, _packet_item=dict(id=mid, content=content, sources=sources,
                _evidence_body=body, memory_type='episode'))


def order(rows, rules=(), budget=32000):
    return prioritize(rows, rules, evidence_packet, schemas, answer_choice,
                      top_k=50, byte_budget=budget)


class LifecyclePriorityTests(unittest.TestCase):
    def test_stable_priority_and_input_immutability(self):
        rows = [candidate('a', 'I enjoy art.'), candidate('b', 'Please forget that I enjoy skiing.'),
                candidate('c', 'I enjoy music.'), candidate('d', 'Please forget that I play baseball.')]
        before = copy.deepcopy(rows)
        result, details = order(rows)
        self.assertEqual([r['id'] for r in result], ['b', 'd', 'a', 'c'])
        self.assertEqual(details['prioritized_ids'], ['b', 'd'])
        self.assertEqual(rows, before)

    def test_only_first_owner_of_identical_constraint_moves(self):
        text = 'Please forget that I enjoy skiing.'
        rows = [candidate('a', 'I enjoy art.'), candidate('b', text, request_id='shared'),
                candidate('c', text, request_id='shared')]
        result, details = order(rows)
        self.assertEqual([r['id'] for r in result], ['b', 'a', 'c'])
        self.assertEqual(len(details['expected_constraint_hashes']), 1)

    def test_packed_rule_prefix_already_covers_constraint(self):
        text = 'Please forget that I enjoy skiing.'
        rule = candidate('rule', text, request_id='shared')
        rule['_packet_item'].update(memory_type='rule', is_constraint=True)
        rows = [candidate('a', 'I enjoy art.'), candidate('b', text, request_id='shared')]
        result, details = order(rows, [rule])
        self.assertEqual([r['id'] for r in result], ['a', 'b'])
        self.assertEqual(len(details['seeded_constraint_hashes']), 1)

    def test_unpacked_rule_does_not_hide_fine_constraint(self):
        text = 'Please forget that I enjoy skiing.'
        rule = candidate('rule', text, request_id='shared', body='Long body. ' * 100)
        rule['_packet_item'].update(memory_type='rule', is_constraint=True)
        rows = [candidate('a', 'I enjoy art.'), candidate('b', text, request_id='shared')]
        result, details = order(rows, [rule], budget=256)
        self.assertEqual([r['id'] for r in result], ['b', 'a'])
        self.assertEqual(details['seeded_constraint_hashes'], [])

    def test_assistant_attribution_hypothetical_and_negation_are_not_promoted(self):
        examples = [('assistant', 'Please forget that I enjoy skiing.'),
                    ('user', 'My friend said: Please forget that I enjoy skiing.'),
                    ('user', 'my friend said: Please forget that I enjoy skiing.'),
                    ('user', 'MY FRIEND SAID: Please forget that I enjoy skiing.'),
                    ('user', 'Imagine a fictional story. Please forget that I enjoy skiing.'),
                    ('user', 'If I move abroad, please forget that I enjoy skiing.'),
                    ('user', 'Please do not forget that I enjoy skiing.')]
        for role, text in examples:
            with self.subTest(role=role, text=text):
                rows = [candidate('a', 'I enjoy art.'), candidate('b', text, role)]
                self.assertEqual(order(rows)[1]['prioritized_ids'], [])

    def test_attribution_exclusion_is_reported_without_relabeling_reader(self):
        rows = [candidate('a', 'My friend said: Please forget that I enjoy skiing.')]
        before = copy.deepcopy(rows)
        _, details = order(rows)
        self.assertEqual(len(details['observations'][0]['attribution_excluded_hashes']), 1)
        self.assertEqual(rows, before)

    def test_hidden_source_and_unsourced_summary_are_not_promoted(self):
        rows = [candidate('a', 'I enjoy art.'),
                candidate('b', 'Please forget that I enjoy skiing.', visible=False),
                candidate('c', 'Ordinary source.', body='Please forget that I enjoy skiing.')]
        self.assertEqual(order(rows)[1]['prioritized_ids'], [])


if __name__ == '__main__':
    unittest.main()
