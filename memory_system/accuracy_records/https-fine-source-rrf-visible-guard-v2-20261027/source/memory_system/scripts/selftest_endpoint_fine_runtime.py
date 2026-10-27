import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, endpoint_fine_completion, evidence_packet, schemas, search_pipeline
import selftest_endpoint_fine_completion as offline_tests
from selftest_fine_direct_lifecycle import candidate


class RuntimeCompletionTests(offline_tests.CompletionTests):
    def setUp(self):
        self.patch = patch.object(offline_tests, 'complete', endpoint_fine_completion.complete)
        self.patch.start()
        self.addCleanup(self.patch.stop)


class AttachmentTests(unittest.TestCase):
    def fixture(self):
        rows = [candidate('a', 'Please forget that I play baseball.'), candidate('b', 'I like music.')]
        packed, digest, manifest = evidence_packet.pack_ranked([rows[0]['_packet_item']], 50, 32000)
        req = schemas.SearchRequest(user_id='test', query='What do I like?', top_k=50)
        plan = dict(_cascade=dict(fine=dict(selected_ids=['a', 'b'], selection={})))
        return req, plan, rows, packed, digest, manifest

    def test_default_off_is_exact_noop(self):
        args = self.fixture()
        with patch.object(config, 'PACKET_ENDPOINT_FINE_COMPLETION', False), patch.object(
                endpoint_fine_completion, 'attach', side_effect=AssertionError('disabled')):
            result = search_pipeline._append_fine_completion(*args)
        for original, after in zip(args[-3:], result):
            self.assertIs(original, after)

    def test_enabled_preserves_prefix_and_updates_real_manifest(self):
        args = self.fixture()
        before = copy.deepcopy(args[1:])
        with patch.object(config, 'PACKET_ENDPOINT_FINE_COMPLETION', True):
            result, digest, manifest = search_pipeline._append_fine_completion(*args)
        self.assertEqual(args[1:], before)
        self.assertEqual(manifest['included_ids'], ['a', 'b'])
        self.assertEqual(manifest['token_upper_bound'], sum(len(item['content'].encode()) for item in result)+1)
        self.assertEqual(manifest['evidence_count']+manifest['constraint_count'], len(result))
        self.assertEqual(evidence_packet.digest(result), digest)
        self.assertEqual(manifest['endpoint_fine_completion']['base_packet_hash'], args[-2])

    def test_stale_fine_is_not_resurrected(self):
        args = self.fixture()
        args[2][1]['view_status'] = 'stale'
        with patch.object(config, 'PACKET_ENDPOINT_FINE_COMPLETION', True):
            result, _, manifest = search_pipeline._append_fine_completion(*args)
        self.assertEqual(result, args[3])
        self.assertEqual(manifest['endpoint_fine_completion']['appended_ids'], [])

    def test_degraded_fine_or_low_deadline_preserves_baseline(self):
        args = self.fixture()
        with patch.object(config, 'PACKET_ENDPOINT_FINE_COMPLETION', True):
            with budget.scope(seconds=.1):
                result = search_pipeline._append_fine_completion(*args)
            self.assertEqual(result[0], args[3])
            self.assertEqual(result[2]['endpoint_fine_completion']['status'], 'budget_skipped')
            args[2][1]['_ce_score'] = None
            result = search_pipeline._append_fine_completion(*args)
            self.assertEqual(result[0], args[3])
            self.assertEqual(result[2]['endpoint_fine_completion']['status'], 'missing_ce_score')


if __name__ == '__main__':
    unittest.main()
