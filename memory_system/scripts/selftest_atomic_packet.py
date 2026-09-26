"""Atomic groups retain unique evidence despite duplicate episode wrappers."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import evidence_packet, answer_context

if os.environ.get('PACKET_CANDIDATE'):
    spec = importlib.util.spec_from_file_location('app.packet_candidate', os.environ['PACKET_CANDIDATE'])
    evidence_packet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence_packet)


class AtomicPacketTests(unittest.TestCase):
    def items(self):
        def item(mid, index, text, kind='fact'):
            source = dict(request_id='r', message_index=index, content=text, role='user')
            return dict(id=mid, content=answer_context.with_evidence(mid, [source]),
                        sources=[source], memory_type=kind)
        return [item('unique', 0, 'An old leg injury limits my cycling.'),
                item('fact', 1, 'Watching television helps me unwind.'),
                item('episode', 1, 'Watching television helps me unwind.', 'episode')]

    def test_duplicate_wrapper_does_not_drop_unique_group_source(self):
        items = self.items()
        packed, digest, manifest = evidence_packet.pack_ranked(items, 3, 4096, groups=[['unique','fact','episode']])
        self.assertEqual([m['id'] for m in packed], ['unique','fact','episode'])
        self.assertIn('An old leg injury', answer_context.build(packed))
        self.assertEqual(answer_context.build(packed).count('Watching television helps'), 1)
        self.assertEqual(evidence_packet.digest(packed), digest)
        self.assertEqual(manifest['evidence_groups'], [['unique','fact','episode']])
        self.assertTrue(all(not any(k.startswith('_') for k in m) for m in packed))
        self.assertIn('content', items[2]['sources'][0])

    def test_top_k_still_rejects_whole_group(self):
        packed, _, manifest = evidence_packet.pack_ranked(self.items(), 2, 4096, groups=[['unique','fact','episode']])
        self.assertEqual(packed, [])
        self.assertTrue(all(m['reason']=='atomic_group_top_k' for m in manifest['omitted']))

    def test_byte_budget_still_rejects_whole_group(self):
        packed, _, manifest = evidence_packet.pack_ranked(self.items(), 3, 100, groups=[['unique','fact','episode']])
        self.assertEqual(packed, [])
        self.assertTrue(all(m['reason']=='atomic_group_token_budget' for m in manifest['omitted']))

    def test_ungrouped_redundant_episode_is_still_removed(self):
        packed, _, manifest = evidence_packet.pack_ranked(self.items(), 3, 4096)
        self.assertEqual([m['id'] for m in packed], ['unique','fact'])
        self.assertTrue(any(m['id']=='episode' for m in manifest['omitted']))

    def test_intervening_duplicate_preserves_group_and_order(self):
        items = self.items()
        packed, _, manifest = evidence_packet.pack_ranked(items, 3, 4096, groups=[['unique','episode']])
        self.assertEqual([m['id'] for m in packed], ['unique','fact','episode'])
        self.assertEqual(manifest['evidence_groups'], [['unique','episode']])
        self.assertEqual(answer_context.build(packed).count('Watching television helps'), 1)

if __name__ == '__main__':
    unittest.main()
