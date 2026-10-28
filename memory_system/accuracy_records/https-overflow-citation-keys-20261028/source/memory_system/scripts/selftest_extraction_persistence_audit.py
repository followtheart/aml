"""Prevent source overlap or stale snapshots from being reported as fact survival."""
import unittest
from audit_extraction_coverage import committed_matches


class PersistenceAuditTests(unittest.TestCase):
    def match(self, memories, indices=(1,)):
        return committed_matches(dict(request_id='r', user_id='u', memories=memories),
                                 [dict(content='I collect pottery.', _sources=list(indices))])[0]

    def memory(self, **overrides):
        return dict(dict(id='m', user_id='u', type='fact', content='I collect pottery.',
                         sources=[dict(request_id='r', message_index=1)],
                         resolution_status='accepted', superseded_by=None), **overrides)

    def test_exact_commit_is_not_current_eligibility(self):
        result = self.match([self.memory(superseded_by='later')])
        self.assertEqual(result['status'], 'exact_at_commit')
        self.assertEqual(result['exact_matches'][0]['superseded_by'], 'later')

    def test_same_source_different_claim_is_not_survival(self):
        self.assertEqual(self.match([self.memory(content='I sell pottery.')])['status'], 'same_source_only')

    def test_request_and_user_identity_required(self):
        for memory in [self.memory(sources=[dict(request_id='other', message_index=1)]),
                       self.memory(user_id='other')]:
            self.assertEqual(self.match([memory])['status'], 'unmatched_unknown')

    def test_episodes_and_missing_sources_are_not_atomic_survival(self):
        for memory in [self.memory(type='episode'), self.memory(sources=[])]:
            self.assertEqual(self.match([memory])['status'], 'unmatched_unknown')
        self.assertEqual(self.match([self.memory()], indices=())['status'], 'unmatched_unknown')

    def test_partial_source_overlap_is_not_exact(self):
        self.assertEqual(self.match([self.memory()], indices=(1, 2))['status'], 'same_source_only')


if __name__ == '__main__':
    unittest.main()
