"""Source nomination boundaries and shared packet accounting, without providers."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, evidence_packet, retrieval_queries, schemas, search_pipeline, source_query_context as context
from selftest_fine_context_lexical import candidate


class SourceQueryContextTests(unittest.TestCase):
    def setUp(self):
        self.base = [schemas.SearchItem(id='base', content='Accepted.', packet_hash_version=4).model_dump()]
        self.fine = [candidate('fine1', 'A request about gardens.'), candidate('fine2', 'Another request.')]
        self.outside = candidate('outside', 'Please translate my note about family photos.')
        self.options = ['Decorate with family photos.']
        self.nomination = dict(routes=[dict(channel='source_text', round=1,
            candidates=[dict(id='outside', _query_ranks={'1': 1})])],
            query_specs=[dict(origin='question', text='Decorations?'),
                dict(origin='option_topic', option_index=0, text=retrieval_queries.topic_anchor(self.options[0]))],
            query='Decorations?', options=self.options,
            coarse_ids=['base', 'fine1', 'fine2', 'outside'], fine_ids=['base', 'fine1', 'fine2'],
            selected_ids=['base'], fine_selection=dict(soft_reservation_policy=dict(max_ce_rank=20),
                decisions=[dict(id='outside', rank=15)]))
        self.kw = dict(user_id='u', top_k=8, byte_budget=9000, reserve_bytes=5120,
                       query='Decorations?', options=self.options)

    def nominate(self, spec):
        return context.nominate({'outside': self.outside}, **spec)[0]

    def test_exact_query_identity_and_rank_window(self):
        self.assertEqual(self.nominate(self.nomination), [self.outside])
        for kind in ('far', 'missing', 'boolean', 'zero', 'no_window', 'no_route', 'topic_changed',
                     'question_changed', 'second_hit', 'followup', 'fine', 'selected', 'not_coarse'):
            spec = copy.deepcopy(self.nomination)
            if kind in ('far', 'missing', 'boolean', 'zero'):
                spec['fine_selection']['decisions'][0]['rank'] = {'far':21, 'missing':None, 'boolean':True, 'zero':0}[kind]
            elif kind == 'no_window': spec['fine_selection'].pop('soft_reservation_policy')
            elif kind == 'no_route': spec['routes'] = []
            elif kind == 'topic_changed': spec['query_specs'][1]['text'] += ' invented'
            elif kind == 'question_changed':
                spec['routes'][0]['candidates'][0]['_query_ranks'] = {'0':1}
                spec['query_specs'][0]['text'] += ' invented'
            elif kind == 'second_hit': spec['routes'][0]['candidates'][0]['_query_ranks']['1'] = 2
            elif kind == 'followup': spec['query_specs'][1]['origin'] = 'grounded_followup'
            elif kind == 'fine': spec['fine_ids'].append('outside')
            elif kind == 'selected': spec['selected_ids'].append('outside')
            elif kind == 'not_coarse': spec['coarse_ids'].remove('outside')
            self.assertEqual(self.nominate(spec), [], kind)

    def test_whole_sources_without_fact_promotion_and_shared_limits(self):
        before = copy.deepcopy((self.base, self.fine, self.outside))
        result, trace = context.propose(self.base, self.fine, champions=[self.outside], **self.kw)
        self.assertEqual(result[:len(self.base)], self.base)
        item = result[len(self.base)]
        self.assertEqual(item['sources'], self.outside['_packet_item']['sources'])
        self.assertEqual(item['score_kind'], 'source_query_context')
        self.assertEqual(item['score'], self.outside['_ce_score'])
        self.assertFalse(item['personal_evidence'])
        self.assertEqual(item['coverage_ids'], [])
        self.assertNotIn('Generated summary', item['content'])
        self.assertEqual(trace['remainder']['reserve_bytes'], 5120-trace['source_seat']['added_bytes'])
        for change in (dict(top_k=1), dict(byte_budget=len(self.base[0]['content'])), dict(reserve_bytes=0)):
            packet, _ = context.propose(self.base, self.fine, champions=[self.outside], **dict(self.kw, **change))
            self.assertEqual(packet, self.base)
        self.assertEqual((self.base, self.fine, self.outside), before)

    def test_invalid_or_duplicate_sources_cannot_take_priority_seat(self):
        for kind in ('foreign', 'stale', 'invisible', 'partial', 'unscored', 'nan', 'assistant', 'constraint', 'duplicate'):
            bad = copy.deepcopy(self.outside)
            source = bad['_packet_item']['sources'][0]
            if kind == 'foreign': source['user_id'] = 'other'
            elif kind == 'stale': bad['view_status'] = 'stale'
            elif kind == 'invisible': bad['_packet_item']['content'] = 'Summary only'
            elif kind == 'partial': source['content_span'] = dict(start=1, end=10, original_length=20)
            elif kind == 'unscored': bad['_ce_score'] = None
            elif kind == 'nan': bad['_ce_score'] = float('nan')
            elif kind == 'assistant': source['role'] = 'assistant'
            elif kind == 'constraint': bad['_packet_item']['is_constraint'] = True
            elif kind == 'duplicate': bad['id'] = 'base'
            _, trace = context.propose(self.base, self.fine, champions=[bad], **self.kw)
            self.assertEqual(trace['source_seat']['appended_ids'], [], kind)

    def test_pipeline_dependencies_exclusions_and_hashes(self):
        spec = self.nomination
        plan = dict(_routes=spec['routes'], _query_specs=spec['query_specs'],
            _cascade=dict(coarse=dict(selected_ids=spec['coarse_ids']),
                fine=dict(selected_ids=spec['fine_ids'], selection=spec['fine_selection'])),
            _selection=dict(selected_ids=['base']), _coverage_requirements=[])
        anchor = dict(id='base', _packet_item=self.base[0])
        req = schemas.SearchRequest(user_id='u', query=self.kw['query'], options=self.options,
                                    top_k=8, evidence_token_budget=9000)
        flags = dict(PACKET_SOURCE_QUERY_CONTEXT=True, PACKET_FINE_CONTEXT_RESERVE=True,
                     PACKET_FINE_CONTEXT_5K=True, PACKET_FINE_CONTEXT_LEXICAL=True,
                     PACKET_FINE_CONTEXT_FILL=False, PACKET_ADJACENT_CONTEXT=False)
        variants = [{}, *[{key:False} for key in ('PACKET_SOURCE_QUERY_CONTEXT',
            'PACKET_FINE_CONTEXT_RESERVE', 'PACKET_FINE_CONTEXT_5K', 'PACKET_FINE_CONTEXT_LEXICAL')],
            {'PACKET_FINE_CONTEXT_FILL':True}]
        for override in variants:
            with patch.multiple(config, **dict(flags, **override)):
                packet, digest, manifest = search_pipeline._pack_evidence(None, req, copy.deepcopy(plan),
                    [anchor], None, [anchor, *self.fine, self.outside])
            self.assertEqual('outside' in [p['id'] for p in packet], not bool(override))
            self.assertEqual(evidence_packet.digest(packet), digest)
            self.assertTrue(all(p['packet_hash'] == digest for p in packet))
            self.assertLessEqual(manifest['token_upper_bound'], 9000)
        plan['_selection']['selected_ids'].append('outside')
        with patch.multiple(config, **flags):
            packet, _, _ = search_pipeline._pack_evidence(None, req, plan, [anchor], None, [anchor, *self.fine, self.outside])
        self.assertNotIn('outside', [p['id'] for p in packet])


if __name__ == '__main__':
    unittest.main()
