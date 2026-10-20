"""Novel evidence and dependency proposals, without provider calls."""
import copy
import unittest
from selftest_fine_selection import candidate, source
from app import fine_evidence


class FineEvidenceTests(unittest.TestCase):
    def rows(self):
        rows = [candidate(f'base{i}', 1-i*.01,
                          [source(f's{i}', f'I read market analysis report {i}.')]) for i in range(16)]
        for row in rows:
            row['_ce_score'] = row['_fixture_score']
        return rows

    def test_low_ce_unique_relevant_source_can_expand_without_displacing_core(self):
        rows = self.rows()
        low = candidate('low', .01, [source('unique', 'I asked about stablecoins and digital currencies.')])
        low['_ce_score'] = .01
        rows.append(low)
        before = copy.deepcopy(rows)
        selected, trace = fine_evidence.propose(rows, 16,
            [dict(id='question', text='emerging cryptocurrency markets')], extra_limit=2)
        self.assertEqual([r['id'] for r in selected[:16]], [r['id'] for r in rows[:16]])
        self.assertIn('low', {r['id'] for r in selected})
        self.assertEqual(rows, before)
        self.assertTrue(trace['adaptive']['support_labels_unchanged'])

    def test_irrelevant_unique_and_redundant_sources_cannot_expand(self):
        rows = self.rows()
        rows += [candidate('unrelated', .01, [source('new', 'I bought a bicycle.')]),
                 candidate('duplicate', .02, [source('s0', 'I read market analysis report 0.')])]
        selected, _ = fine_evidence.propose(rows, 16, [dict(id='question', text='market analysis')])
        self.assertEqual(len(selected), 16)

    def test_explicit_path_added_as_whole_group(self):
        rows = self.rows()
        rows[0]['_relation_path_ids'] = ['bridge1', 'base0']
        first = candidate('bridge1', .01, [source('b1', 'The activity uses tool Q.')])
        first['_relation_path_ids'] = ['bridge2', 'bridge1']
        rows += [first, candidate('bridge2', .001, [source('b2', 'Tool Q is the required connection.')])]
        selected, trace = fine_evidence.propose(rows, 16, [], extra_limit=2)
        self.assertEqual(len(selected), 18)
        self.assertEqual(set(trace['adaptive']['additions'][0]['group_ids']), {'base0', 'bridge1', 'bridge2'})
        selected, trace = fine_evidence.propose(rows, 16, [], extra_limit=1)
        self.assertEqual(len(selected), 16)
        self.assertTrue(any(r['reason'] == 'group_capacity' for r in trace['adaptive']['rejected']))

    def test_unordered_retrieval_seeds_do_not_manufacture_an_atomic_path(self):
        rows = self.rows()
        rows[0]['_bridge_ids'] = ['neighbor']
        rows.append(candidate('neighbor', .01, [source('other', 'An unrelated graph neighbor.')]))
        selected, _ = fine_evidence.propose(rows, 16, [], extra_limit=2)
        self.assertEqual(len(selected), 16)

    def test_conflicting_state_not_collapsed_and_unscored_is_reviewable(self):
        rows = self.rows()
        rows[0]['state'] = dict(slot='interest', value='past')
        changed = candidate('changed', .01, [source('s0', 'I read market analysis report 0.')])
        changed['state'] = dict(slot='interest', value='current')
        changed['_ce_score'] = None
        rows.append(changed)
        selected, trace = fine_evidence.propose(rows, 16, [dict(id='question', text='market analysis')])
        self.assertIn('changed', {r['id'] for r in selected})
        self.assertTrue(trace['adaptive']['additions'][0]['unscored'])

    def test_admission_keeps_core_and_rejects_entire_group_on_token_budget(self):
        rows = self.rows()[:1] + [candidate('a', .2), candidate('b', .1)]
        addition = dict(candidate_id='a', group_ids=['a', 'b'])
        admitted, trace = fine_evidence.admit_additions(rows[:1], rows, [addition],
            count_limit=3, byte_limit=100, token_headroom=20,
            request_for=lambda rs: dict(prompt='x' * len(rs), reservation=10 * len(rs)))
        self.assertEqual([r['id'] for r in admitted], ['base0'])
        self.assertEqual(trace[0]['reason'], 'token_budget')

    def test_review_error_does_not_override_a_valid_negative_or_duplicate(self):
        rows = self.rows()
        recovery = dict(unreviewed_ids=['base1', 'base2'], reviewed_ids=['base1'])
        pending = fine_evidence.provisional_candidates(rows, rows[:1], recovery,
            [dict(id='question', text='market analysis')], 2)
        self.assertEqual([c['id'] for c in pending], ['base2'])


if __name__ == '__main__':
    unittest.main()
