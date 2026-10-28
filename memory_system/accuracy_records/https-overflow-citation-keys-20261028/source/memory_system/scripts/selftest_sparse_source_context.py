"""Sparse source representatives preserve provenance, prefix and all budgets."""
import copy
import unittest
from unittest.mock import patch

from selftest_fine_context_lexical import candidate
from app import answer_context, config, evidence_packet, schemas, search_pipeline
from app import sparse_source_context as context, fine_sparse_reserve, evidence_selection


def visible(row):
    row['_rank_text'] = row['_packet_item']['content']
    return row


class SparseSourceTests(unittest.TestCase):
    def setUp(self):
        self.base = schemas.SearchItem(id='base', content='Accepted evidence.', packet_hash_version=4).model_dump()
        self.fine = [visible(candidate('one', 'Walking. ' + 'a'*1200)),
                     visible(candidate('two', 'Gardening. ' + 'b'*1200)),
                     visible(candidate('tea', 'Tea? ' + 'c'*1200))]
        self.small = visible(candidate('small', 'I attended the fusion festival.'))
        self.large = copy.deepcopy(self.small)
        self.large.update(id='large', _ce_score=.9)
        self.large['_packet_item']['sources'].append(candidate('padding', 'Other text. '+'x'*3500)['_packet_item']['sources'][0])
        self.large['_packet_item']['content'] = answer_context.with_evidence('Generated summary.', self.large['_packet_item']['sources'])
        visible(self.large)
        self.needs = [dict(id='option:0', text='modern fusion cuisine dinner party')]
        self.req = schemas.SearchRequest(user_id='u', query='Tea?', options=[], top_k=8, evidence_token_budget=12000)
        self.plan = dict(_coverage_requirements=self.needs, _selection=dict(selected_ids=['base']),
            _cascade=dict(coarse=dict(selected_ids=['base','one','two','tea','large','small']),
                fine=dict(selected_ids=['base','one','two','tea'])))
        self.flags = dict(PACKET_SPARSE_SOURCE_CONTEXT=True, PACKET_FINE_CONTEXT_RESERVE=True,
            PACKET_FINE_CONTEXT_5K=True, PACKET_FINE_CONTEXT_LEXICAL=True,
            PACKET_FINE_CONTEXT_FILL=False, PACKET_SOURCE_QUERY_CONTEXT=False,
            PACKET_QUESTION_CONTEXT_CE=False, PACKET_SOURCE_FIRST_RESIDUAL=False,
            PACKET_MARGINAL_QUERY_CONTEXT=False, PACKET_QUERY_CONTEXT_COMBINED=False,
            PACKET_CORE_GAP_CONTEXT=False)

    def pack(self, extras=None, flags=None, plan=None, req=None):
        with patch.multiple(config, **dict(self.flags, **(flags or {}))):
            return search_pipeline._pack_evidence(None, req or self.req, copy.deepcopy(plan or self.plan),
                [dict(id='base', _packet_item=self.base, _final=.9)], '2026-01-01T00:00:00Z',
                self.fine + ([self.large, self.small] if extras is None else extras))

    def test_smaller_existing_carrier_preserves_whole_source_and_original_score(self):
        before = copy.deepcopy((self.base, self.fine, self.small, self.large, self.plan))
        off = self.pack(flags=dict(PACKET_SPARSE_SOURCE_CONTEXT=False))
        on = self.pack()
        self.assertEqual(on[0][-1]['id'], 'small')
        self.assertEqual(on[0][-1]['sources'], self.small['_packet_item']['sources'])
        self.assertEqual(on[0][-1]['score'], .2)
        self.assertEqual([dict(p,packet_hash=None) for p in on[0][:-1]], [dict(p,packet_hash=None) for p in off[0]])
        ctx = on[2]['conversation_context']
        self.assertEqual(ctx['prefix_context'], off[2]['conversation_context'])
        self.assertEqual(ctx['reserve_bytes'], 5120)
        self.assertLessEqual(ctx['added_bytes'], 5120)
        self.assertEqual(ctx['sparse_source']['appended_ids'], ['small'])
        self.assertFalse(on[0][-1]['personal_evidence'])
        self.assertEqual(on[0][-1]['coverage_ids'], [])
        self.assertEqual(evidence_packet.digest(on[0]), on[1])
        self.assertEqual((self.base, self.fine, self.small, self.large, self.plan), before)

    def test_matching_term_on_different_source_does_not_borrow_carrier_priority(self):
        other = visible(candidate('other', 'Fusion details.'))
        other['_ce_score'] = .8
        plan = copy.deepcopy(self.plan)
        plan['_cascade']['coarse']['selected_ids'].append('other')
        # Duplicate many independent nonmatching sources to keep both hits rare.
        fillers = [visible(candidate('filler'+str(i), 'Ordinary passage '+str(i))) for i in range(30)]
        plan['_cascade']['coarse']['selected_ids'].extend(c['id'] for c in fillers)
        result = self.pack(extras=[self.large, self.small, other, *fillers], plan=plan)
        self.assertEqual(result[0][-1]['id'], 'small')
        ordered = result[2]['conversation_context']['sparse_source']['nomination']['ordered_ids']
        self.assertLess(ordered.index('small'), ordered.index('other'))

    def test_hidden_foreign_stale_partial_assistant_unscored_are_rejected(self):
        for kind in ('hidden', 'foreign', 'stale', 'partial', 'assistant', 'unscored', 'nan', 'constraint'):
            c = copy.deepcopy(self.small)
            s = c['_packet_item']['sources'][0]
            if kind == 'hidden': c['_packet_item']['content'] = 'Unrelated view.'
            if kind == 'foreign': s['user_id'] = 'another-user'
            if kind == 'stale': c['view_status'] = 'stale'
            if kind == 'partial': s['content_span'] = dict(start=1,end=10,original_length=100)
            if kind == 'assistant': s['role'] = 'assistant'
            if kind == 'unscored': c['_ce_score'] = None
            if kind == 'nan': c['_ce_score'] = float('nan')
            if kind == 'constraint': c['_packet_item']['is_constraint'] = True
            with self.subTest(kind=kind):
                self.assertFalse(self.pack(extras=[c])[2]['conversation_context']['sparse_source']['appended_ids'])

    def test_current_pool_fine_and_dependency_boundaries(self):
        for kind in ('outside_coarse','already_fine','missing_dependency','unavailable_dependency','scope'):
            plan = copy.deepcopy(self.plan)
            c = copy.deepcopy(self.small)
            if kind == 'outside_coarse': plan['_cascade']['coarse']['selected_ids'].remove('small')
            if kind == 'already_fine': plan['_cascade']['fine']['selected_ids'].append('small')
            if kind == 'missing_dependency': c['_relation_path_ids'] = ['small','absent']
            if kind == 'unavailable_dependency': c['_relation_path_ids'] = ['small','large']
            if kind == 'scope':
                plan['time_scope'] = dict(**{'from':'2025-01-01T00:00:00Z'})
                c.update(type='plan', temporal=dict(start='2020-01-01T00:00:00Z',end='2020-01-02T00:00:00Z'))
            with self.subTest(kind=kind):
                self.assertFalse(self.pack(extras=[c],plan=plan)[2]['conversation_context']['sparse_source']['appended_ids'])

    def test_common_or_repeated_requirement_term_is_not_a_rare_cue(self):
        for mode in ('core_user', 'two_requirements'):
            plan = copy.deepcopy(self.plan)
            fine = copy.deepcopy(self.fine)
            if mode == 'core_user':
                self.fine[0]['_packet_item']['sources'][0]['content'] = 'The fusion festival.'
                self.fine[0]['_packet_item']['content'] = 'The fusion festival.'
                visible(self.fine[0])
            else: plan['_coverage_requirements'].append(dict(id='question',text='fusion'))
            self.assertFalse(self.pack(plan=plan)[2]['conversation_context']['sparse_source']['appended_ids'])
            self.fine = fine

    def test_global_bytes_top_k_and_spare_budget_bind(self):
        off = self.pack(flags=dict(PACKET_SPARSE_SOURCE_CONTEXT=False))
        for req in (self.req.model_copy(update=dict(top_k=len(off[0]))),
                    self.req.model_copy(update=dict(evidence_token_budget=off[2]['token_upper_bound']))):
            self.assertEqual(self.pack(req=req)[:2], self.pack(req=req,flags=dict(PACKET_SPARSE_SOURCE_CONTEXT=False))[:2])
        c = visible(candidate('small','Fusion. '+'界'*1000))
        self.assertFalse(self.pack(extras=[c])[2]['conversation_context']['sparse_source']['appended_ids'])

    def test_all_feature_gates_and_bad_digest(self):
        for key, value in self.flags.items():
            with patch.multiple(config, **dict(self.flags, **{key:not value})):
                self.assertFalse(context.enabled(), key)
        packet,digest,manifest = self.pack(flags=dict(PACKET_SPARSE_SOURCE_CONTEXT=False))
        with self.assertRaises(ValueError):
            context.attach(packet,'wrong',manifest,[self.small],{},user_id='u',top_k=8)

    def test_profile_quote_projections_share_original_source_frequency(self):
        rows = [visible(candidate('filler'+str(i), 'Ordinary passage '+str(i))) for i in range(24)]
        rows.append(self.small)
        core = rows[:16]
        def nominations():
            return fine_sparse_reserve.nominate(rows, core, self.needs,
                {r['id']:evidence_selection.fragments(r) for r in rows})
        expected = nominations()['small']
        for i in range(30):
            clone = copy.deepcopy(self.small)
            clone.update(id='profile'+str(i), type='profile', evidence=[dict(
                quote=self.small['_packet_item']['sources'][0]['content'], request_id='r',
                message_index=len('small'))])
            rows.append(clone)
        self.assertEqual(nominations()['small'], expected)


if __name__ == '__main__':
    unittest.main()
