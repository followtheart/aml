"""Identity joins must reject ambiguous or mismatched frozen evidence."""
import unittest
from inventory_main_residual_capacity import jobs_by_search_id


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.fixed = [dict(conversation_id='a', qa_id='1', packet_hash='pa'),
                      dict(conversation_id='b', qa_id='1', packet_hash='pb')]
        self.results = [dict(self.fixed[1], search_id='sb'), dict(self.fixed[0], search_id='sa')]

    def test_composite_identity_and_different_order(self):
        result = jobs_by_search_id(iter(self.fixed), iter(self.results))
        self.assertIs(result['sa'], self.fixed[0])
        self.assertIs(result['sb'], self.fixed[1])

    def test_missing_or_duplicate_identity_rejected(self):
        for fixed, results in ((self.fixed, self.results[:1]),
                               (self.fixed + self.fixed[:1], self.results),
                               (self.fixed, self.results + self.results[:1])):
            with self.subTest(fixed=fixed, results=results), self.assertRaises(ValueError):
                jobs_by_search_id(fixed, results)

    def test_packet_identity_rejected(self):
        changed = [dict(self.results[0], packet_hash='wrong'), self.results[1]]
        with self.assertRaises(ValueError):
            jobs_by_search_id(self.fixed, changed)


if __name__ == '__main__':
    unittest.main()
