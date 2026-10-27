"""Budget and preservation contracts for coarse-to-main residual admission."""
import copy
from types import SimpleNamespace
import unittest

import main_residual_admission as policy


def row(mid, score, size=5, **kwargs):
    return dict(id=mid, _ce_score=score, source=mid, size=size, type='episode', **kwargs)


class AdmissionTests(unittest.TestCase):
    def run_policy(self, candidates, fine=None, main=None, **limits):
        fine = candidates[:1] if fine is None else fine
        main = fine if main is None else main
        helpers = SimpleNamespace(key=lambda r: (r['_ce_score'], r.get('_fused', 0)),
            dependencies=lambda r: r.get('deps', []), selection=SimpleNamespace(
                fragments=lambda r: [r['source']] if r['source'] else [],
                represented_by=lambda r, selected, visible: any(r['source'] == x['source'] for x in selected)))
        def request_for(rows):
            return dict(prompt='文' * sum(r['size'] for r in rows),
                        reservation=sum(r.get('tokens', r['size']) for r in rows))
        return policy.propose(candidates, fine, main, helpers=helpers, request_for=request_for,
                              **dict(count_limit=3, byte_limit=60, token_headroom=40, **limits))

    def test_first_fitting_whole_candidate_preserves_prefix(self):
        rows = [row('old', 1), row('too_large', .9, 30), row('fits', .8, 10), row('later', .7)]
        original = copy.deepcopy(rows)
        selected, details = self.run_policy(rows)
        self.assertEqual([x['id'] for x in selected], ['old', 'fits'])
        self.assertIs(selected[0], rows[0]); self.assertIs(selected[1], rows[2])
        self.assertEqual(details['decisions'][0]['reason'], 'prompt_budget')
        self.assertEqual(rows, original)

    def test_invalid_sources_scores_and_dependencies_excluded(self):
        rows = [row('old', 1), row('dep', .99, deps=['other']), row('same', .98),
                row('empty', .97), row('nan', float('nan')), row('bool', True), row('valid', .5)]
        rows[2]['source'] = 'old'; rows[3]['source'] = ''
        selected, details = self.run_policy(rows)
        self.assertEqual([r['id'] for r in selected], ['old', 'valid'])
        self.assertEqual([r['reason'] for r in details['decisions']],
                         ['not_singleton', 'represented_source', 'no_visible_source', 'admitted'])

    def test_token_budget_and_no_change(self):
        rows = [row('old', 1), row('tokens', .9, tokens=50)]
        selected, details = self.run_policy(rows)
        self.assertEqual(selected, rows[:1]); self.assertIsNone(details['addition'])
        self.assertEqual(details['decisions'][0]['reason'], 'token_budget')

    def test_fine_members_and_full_main_are_not_replaced(self):
        rows = [row('old', 1), row('fine_only', .9), row('main2', .8), row('main3', .7), row('outside', .6)]
        main = [rows[0], rows[2], rows[3]]
        selected, details = self.run_policy(rows, fine=rows[:2], main=main)
        self.assertEqual(selected, main); self.assertIsNone(details['addition'])
        self.assertEqual([d['id'] for d in details['decisions']], ['outside'])
        self.assertEqual(details['decisions'][0]['reason'], 'count_budget')


if __name__ == '__main__':
    unittest.main()
