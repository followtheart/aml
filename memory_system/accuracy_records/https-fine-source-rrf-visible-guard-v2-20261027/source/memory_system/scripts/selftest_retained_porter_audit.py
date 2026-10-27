"""Small adversarial checks for offline audit invariants; no app/provider work."""
import copy
import unittest

from analyze_overflow_https import audit_coarse_pool, fine_core_eligible
from audit_retained_porter_https import canonical
from summarize_retained_porter_https import turn_changes


def trace():
    return dict(versions={'settings': dict(COARSE_EVIDENCE_EXTEND=True,
                    CASCADE_COARSE_LIMIT=50, RECALL_PORTER_SUPPLEMENT=True)},
                porter_recall=dict(status='ok', nominated_ids=['new'], admitted_ids=['new']),
                routes=[dict(channel='full_text', candidates=[{'id': 'old'}, {'id': 'extra'}])],
                cascade=dict(coarse=dict(baseline_ids=['old'], extra_ids=['extra'],
                    supplemental_ids=['new'], selected_ids=['old', 'extra', 'new'],
                    extra_limit=8, supplemental_limit=20, limit=78),
                    cross_encoder=dict(batches=[dict(candidate_ids=['old', 'extra', 'new'])])),
                ranked=[dict(id='old', _ce_score=.2, _fused=.8),
                        dict(id='extra', _ce_score=.4, _fused=.6),
                        dict(id='new', _ce_score=.9, _fused=.1)])


class AuditTests(unittest.TestCase):
    def test_new_candidate_competes_for_single_core_seat(self):
        t = trace()
        audit_coarse_pool(t)
        self.assertEqual(fine_core_eligible(t), {'old', 'new'})

    def test_failed_supplement_cannot_take_core_seat(self):
        t = trace()
        t['ranked'][2]['_ce_score'] = None
        self.assertEqual(fine_core_eligible(t), {'old', 'extra'})

    def test_nomination_must_be_absent_from_all_old_routes(self):
        t = trace()
        t['routes'].append(dict(channel='followup', candidates=[{'id': 'new'}]))
        with self.assertRaisesRegex(ValueError, 'eligibility'):
            audit_coarse_pool(t)

    def test_old_prefix_cannot_be_displaced(self):
        t = trace()
        t['cascade']['coarse']['selected_ids'] = ['old', 'new', 'extra']
        with self.assertRaisesRegex(ValueError, 'prefix'):
            audit_coarse_pool(t)

    def test_duplicate_or_oversized_nomination_fails(self):
        for nominations in (['new', 'new'], ['new'] + [f'x{i}' for i in range(20)]):
            with self.subTest(nominations=nominations):
                t = trace()
                t['porter_recall']['nominated_ids'] = nominations
                with self.assertRaisesRegex(ValueError, 'eligibility'):
                    audit_coarse_pool(t)

    def test_ce_cannot_score_outside_pool(self):
        t = trace()
        t['cascade']['cross_encoder']['batches'][0]['candidate_ids'].append('outside')
        with self.assertRaisesRegex(ValueError, 'outside'):
            audit_coarse_pool(t)

    def test_disabled_control_and_empty_treatment_keep_old_capacity(self):
        for enabled in (False, True):
            t = trace()
            t['versions']['settings']['RECALL_PORTER_SUPPLEMENT'] = enabled
            t['porter_recall'] = dict(status='ok' if enabled else 'disabled')
            t['cascade']['coarse'].update(selected_ids=['old', 'extra'], limit=58)
            t['cascade']['coarse'].pop('supplemental_ids')
            t['cascade']['coarse'].pop('supplemental_limit')
            t['cascade']['cross_encoder']['batches'] = []
            audit_coarse_pool(t)

    def test_normalization_does_not_hide_candidate_order_or_duplicates(self):
        a = dict(reservations=[{'id': 'b'}, {'id': 'a'}], selected_ids=['a', 'b'])
        b = copy.deepcopy(a)
        b['reservations'].reverse()
        self.assertEqual(canonical(a), canonical(b))
        b['selected_ids'].reverse()
        self.assertNotEqual(canonical(a), canonical(b))
        b = copy.deepcopy(a)
        b['reservations'].append({'id': 'a'})
        self.assertNotEqual(canonical(a), canonical(b))

    def test_equal_turn_totals_do_not_hide_a_lost_reference(self):
        def row(flags):
            return {'visible': {'catalog': {'turn_coverage': [
                dict(full=flag, normalized_chars=length) for flag, length in zip(flags, [10, 20])]}}}
        self.assertEqual(turn_changes(row([True, False]), row([False, True]), 'catalog'),
                         dict(lost=[0], gained=[1]))
        changed = row([True, False])
        changed['visible']['catalog']['turn_coverage'][0]['normalized_chars'] = 11
        with self.assertRaisesRegex(ValueError, 'inventory'):
            turn_changes(row([True, False]), changed, 'catalog')


if __name__ == '__main__':
    unittest.main()
