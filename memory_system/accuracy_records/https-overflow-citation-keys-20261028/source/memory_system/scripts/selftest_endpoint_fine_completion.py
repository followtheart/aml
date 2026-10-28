import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, evidence_packet, schemas
from endpoint_fine_completion import complete, without_digest
from selftest_fine_direct_lifecycle import candidate


def baseline(rows):
    items = [dict(row['_packet_item'], score=.5, score_kind='cross_encoder') for row in rows]
    packed, _, _ = evidence_packet.pack_ranked(items, 50, 32000)
    return [schemas.SearchItem.model_validate(item).model_dump() for item in packed]


def run(packed, rows, groups=(), **kwargs):
    return complete(packed, rows, groups, evidence_packet, schemas, answer_choice,
                    top_k=kwargs.get('top_k', 50), byte_budget=kwargs.get('byte_budget', 32000))


class CompletionTests(unittest.TestCase):
    def test_prefix_source_labels_and_inputs_immutable(self):
        packed = baseline([candidate('a', 'Please forget that I play baseball.')])
        rows = [candidate('b', 'I like music.', body='[inferred; not stated by the user] Topic: music.')]
        before = copy.deepcopy((packed, rows))
        result, digest, trace = run(packed, rows)
        self.assertEqual((packed, rows), before)
        self.assertEqual(without_digest(result[0]), without_digest(packed[0]))
        self.assertEqual(result[1]['content'], rows[0]['_packet_item']['content'])
        self.assertEqual(result[1]['sources'], rows[0]['_packet_item']['sources'])
        self.assertEqual(evidence_packet.digest(result), digest)
        self.assertEqual(trace['appended_ids'], ['b'])

    def test_only_visible_novel_user_text_admitted(self):
        packed = baseline([candidate('a', 'I like music.', request_id='shared')])
        rows = [candidate('b', 'I like music.', request_id='shared'),
                candidate('c', 'I like art.', role='assistant'),
                candidate('d', 'I like books.', visible=False), candidate('e', 'I like hiking.')]
        self.assertEqual(run(packed, rows)[2]['appended_ids'], ['e'])

    def test_atomic_group_keeps_assistant_dependency(self):
        rows = [candidate('a', 'I like music.'), candidate('b', 'Try piano.', role='assistant')]
        result, _, trace = run([], rows, [['a', 'b']])
        self.assertEqual(trace['appended_ids'], ['a', 'b'])
        self.assertEqual(len(result), 2)
        self.assertEqual(run([], rows, [['a', 'b']], top_k=1)[2]['appended_ids'], [])

    def test_byte_boundary_is_whole_unit(self):
        packed = baseline([candidate('a', 'I like music.')])
        rows = [candidate('b', 'I like art.')]
        used = run(packed, rows)[2]['total_bytes']
        self.assertEqual(run(packed, rows, byte_budget=used)[2]['appended_ids'], ['b'])
        self.assertEqual(run(packed, rows, byte_budget=used-1)[2]['appended_ids'], [])

    def test_partial_old_group_member_blocks_new_group(self):
        rows = [candidate('a', 'I like music.'), candidate('b', 'I like art.')]
        packed = baseline([candidate('a', 'I like music.', body='Shorter body')])
        self.assertEqual(run(packed, rows, [['a', 'b']])[2]['decisions'][0]['reason'], 'unavailable_dependency')

    def test_invalid_score_group_or_baseline_fails(self):
        row = candidate('a', 'I like music.')
        with self.assertRaises(ValueError):
            run([], [row], [['a', 'missing']])
        row['_ce_score'] = float('nan')
        with self.assertRaises(ValueError):
            run([], [row])
        with self.assertRaises(ValueError):
            run(baseline([candidate('b', 'I like music.')]), [], top_k=0)


if __name__ == '__main__':
    unittest.main()
