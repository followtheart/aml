"""Shared-topic retrieval keeps source ownership and all original packet limits."""
import copy
import unittest
from unittest.mock import patch

from selftest_fine_context_lexical import candidate
from app import answer_context, config, evidence_packet, exclusive_shared_context as context, schemas, search_pipeline


class ExclusiveSharedTests(unittest.TestCase):
    def setUp(self):
        self.base = schemas.SearchItem(id='base', content='Accepted evidence.', packet_hash_version=4).model_dump()
        self.fine = [candidate('one', 'Walking. '+'a'*900), candidate('two', 'Gardening. '+'b'*900),
                     candidate('tea', 'Tea? '+'c'*900)]
        self.extra = candidate('extra', 'Please polish this biography.')
        self.extra['_packet_item']['sources'] += candidate('reply', 'The office has orchids.', role='assistant')['_packet_item']['sources']
        self.extra['_packet_item']['content'] = answer_context.with_evidence('Generated summary.', self.extra['_packet_item']['sources'])
        self.req = schemas.SearchRequest(user_id='u', query='Tea?', options=['A. Add orchids.', 'B. Move orchids.'],
            top_k=8, evidence_token_budget=12000)
        self.plan = dict(_coverage_requirements=[], _selection=dict(selected_ids=['base']),
            _cascade=dict(coarse=dict(selected_ids=['base', 'one', 'two', 'tea', 'extra']),
                fine=dict(selected_ids=['base', 'one', 'two', 'tea'])))
        self.flags = dict(PACKET_EXCLUSIVE_SHARED_CONTEXT=True, PACKET_FINE_CONTEXT_RESERVE=True,
            PACKET_FINE_CONTEXT_5K=True, PACKET_FINE_CONTEXT_LEXICAL=True,
            **{key:False for key in context.INCOMPATIBLE})

    def pack(self, extras=None, plan=None, req=None, flags=None):
        with patch.multiple(config, **dict(self.flags, **(flags or {}))):
            return search_pipeline._pack_evidence(None, req or self.req, copy.deepcopy(plan or self.plan),
                [dict(id='base', _packet_item=self.base, _final=.9)], '2026-01-01T00:00:00Z',
                self.fine + ([self.extra] if extras is None else extras))

    def test_retains_assistant_anchor_with_user_context_without_fact_promotion(self):
        original = copy.deepcopy((self.base, self.fine, self.extra, self.plan))
        off = self.pack(flags=dict(PACKET_EXCLUSIVE_SHARED_CONTEXT=False))
        on = self.pack()
        self.assertEqual(on[0][-1]['id'], 'extra')
        self.assertEqual(on[0][-1]['sources'], self.extra['_packet_item']['sources'])
        self.assertFalse(on[0][-1]['personal_evidence'])
        self.assertEqual(on[0][-1]['coverage_ids'], [])
        self.assertEqual([dict(p,packet_hash=None) for p in on[0][:-1]], [dict(p,packet_hash=None) for p in off[0]])
        self.assertEqual(on[2]['conversation_context']['prefix_context'], off[2]['conversation_context'])
        self.assertEqual(evidence_packet.digest(on[0]), on[1])
        self.assertEqual((self.base, self.fine, self.extra, self.plan), original)

    def test_repeated_projection_is_not_an_independent_observation(self):
        clone = copy.deepcopy(self.extra)
        clone['id'] = 'clone'
        clone['_ce_score'] = .1
        plan = copy.deepcopy(self.plan)
        plan['_cascade']['coarse']['selected_ids'].append('clone')
        a = self.pack()
        b = self.pack(extras=[self.extra, clone], plan=plan)
        self.assertEqual(a[0], b[0])
        self.assertEqual(b[2]['conversation_context']['exclusive_shared']['nomination']['source_frequency']['orchids'], 1)
        distinct = copy.deepcopy(clone)
        distinct['_packet_item']['sources'][1]['source_event_id'] = 'different'
        distinct['_packet_item']['sources'][1]['message_index'] = 99
        self.assertFalse(self.pack(extras=[self.extra, distinct],plan=plan)[2]['conversation_context']['exclusive_shared']['appended_ids'])

    def test_repeated_word_in_one_option_does_not_count_as_two_options(self):
        req = self.req.model_copy(update=dict(options=['Orchids orchids orchids.', 'A bright lamp.']))
        self.assertFalse(self.pack(req=req)[2]['conversation_context']['exclusive_shared']['appended_ids'])

    def test_already_delivered_source_term_needs_no_second_carrier(self):
        self.fine[0] = candidate('one', 'The room has orchids.')
        result = self.pack()
        self.assertFalse(result[2]['conversation_context']['exclusive_shared']['appended_ids'])

    def test_ineligible_sources_and_dependencies_never_expand(self):
        for kind in ('foreign', 'hidden', 'stale', 'partial_user', 'no_user', 'nan', 'constraint', 'dependency', 'scope'):
            c = copy.deepcopy(self.extra)
            plan = copy.deepcopy(self.plan)
            if kind=='foreign': c['_packet_item']['sources'][1]['user_id']='other'
            if kind=='hidden': c['_packet_item']['content']='Summary only.'
            if kind=='stale': c['view_status']='stale'
            if kind=='partial_user': c['_packet_item']['sources'][0]['content_span']=dict(start=1,end=28,original_length=90)
            if kind=='no_user': c['_packet_item']['sources'][0]['role']='assistant'
            if kind=='nan': c['_ce_score']=float('nan')
            if kind=='constraint': c['_packet_item']['is_constraint']=True
            if kind=='dependency': c['_relation_path_ids']=['extra','missing']
            if kind=='scope':
                plan['time_scope']={'from':'2025-01-01T00:00:00Z'}
                c.update(type='plan',temporal=dict(start='2020-01-01T00:00:00Z',end='2020-01-02T00:00:00Z'))
            with self.subTest(kind=kind):
                self.assertFalse(self.pack(extras=[c],plan=plan)[2]['conversation_context']['exclusive_shared']['appended_ids'])

    def test_item_and_byte_limits_preserve_prefix(self):
        off = self.pack(flags=dict(PACKET_EXCLUSIVE_SHARED_CONTEXT=False))
        for req in (self.req.model_copy(update=dict(top_k=len(off[0]))),
                    self.req.model_copy(update=dict(evidence_token_budget=off[2]['token_upper_bound']))):
            self.assertEqual(self.pack(req=req)[:2], self.pack(req=req,flags=dict(PACKET_EXCLUSIVE_SHARED_CONTEXT=False))[:2])
        c = copy.deepcopy(self.extra)
        c['_packet_item']['sources'][1]['content'] += '界'*2000
        c['_packet_item']['content'] = answer_context.with_evidence('', c['_packet_item']['sources'])
        self.assertFalse(self.pack(extras=[c])[2]['conversation_context']['exclusive_shared']['appended_ids'])

    def test_feature_gates_and_digest_validation(self):
        for key, value in self.flags.items():
            with patch.multiple(config, **dict(self.flags, **{key:not value})):
                self.assertFalse(context.enabled(), key)
        packet, _, manifest = self.pack(flags=dict(PACKET_EXCLUSIVE_SHARED_CONTEXT=False))
        with self.assertRaises(ValueError):
            context.attach(packet,'wrong',manifest,[self.extra],{},user_id='u',top_k=8)


if __name__ == '__main__':
    unittest.main()
