"""Graph identity and literal preservation contracts for display-only aliases."""
import copy
import json
from types import SimpleNamespace
import unittest

import local_candidate_ids as encoding


class IdentityTests(unittest.TestCase):
    def request(self):
        return SimpleNamespace(query='Which context?', options=['A. first', 'B. second'])

    def test_translation_filters_original_namespace_and_preserves_objects(self):
        rows = [dict(id='amu_first',content='first',_packet_item={'id':'amu_first'}),
                dict(id='amu_second',content='second')]
        plan = dict(_fusion_edges=[dict(source='amu_first',target='amu_second',kind='related'),
                                   dict(source='c0',target='amu_second',kind='outside')],
                    _fusion_triples=[dict(amu_id='amu_first',object='words',valid_to=None)],
                    _fine_path_groups=[['amu_first','amu_second'],['c0','amu_second']])
        old = copy.deepcopy((rows,plan))
        display, converted, aliases = encoding.translate(self.request(),plan,rows,{})
        self.assertEqual([r['id'] for r in display],['c0','c1'])
        self.assertEqual(converted['_fusion_edges'],[dict(source='c0',target='c1',kind='related')])
        self.assertEqual(converted['_fine_path_groups'],[['c0','c1']])
        self.assertIs(display[0]['_packet_item'],rows[0]['_packet_item'])
        self.assertEqual((rows,plan),old)

    def test_literal_ids_and_alias_collisions_are_protected(self):
        rows=[dict(id='amu_a',content='References amu_a and c1'),dict(id='amu_b',content='b'),
              dict(id='amu_c',content='c')]
        metadata=dict(triples=[dict(amu_id='amu_b',object='Literal amu_b')])
        _,_,aliases=encoding.translate(self.request(),{},rows,metadata)
        self.assertEqual(aliases['amu_a'],'amu_a');self.assertEqual(aliases['amu_b'],'amu_b')
        self.assertEqual(aliases['amu_c'],'_c2')

    def test_all_graph_identity_fields_decode_with_literal_values_unchanged(self):
        graph=dict(triples=[dict(amu_id='c0',object='literal c1',valid_from=None)],
            associations=[dict(source='c0',target='c1',relation='asks',weight=.2)],
            incomplete_candidate_context={'c0':['missing original source']},
            recorded_paths_for_joint_review=[['c0','c1']],path_scope='not proof')
        restored=encoding.decode_graph(graph,{'amu_a':'c0','amu_b':'c1'})
        self.assertEqual(restored['triples'][0],dict(amu_id='amu_a',object='literal c1',valid_from=None))
        self.assertEqual(restored['associations'][0]['target'],'amu_b')
        self.assertEqual(list(restored['incomplete_candidate_context']),['amu_a'])
        self.assertEqual(restored['recorded_paths_for_joint_review'],[['amu_a','amu_b']])
        self.assertEqual(graph['triples'][0]['amu_id'],'c0')

    def test_graph_loss_counts_duplicates_and_missing_null(self):
        graph=dict(triples=[{'id':1},{'id':1}],associations=[],path_scope=None)
        missing=encoding.missing_graph(graph,dict(triples=[{'id':1}],associations=[]))
        self.assertEqual(missing,[dict(field='triples',value={'id':1},count=1),
                                  dict(field='path_scope',value=None)])
        self.assertEqual(encoding.missing_graph(graph,dict(graph,extra=True)),[])

    def test_duplicate_ids_fail(self):
        with self.assertRaises(ValueError):
            encoding.translate(self.request(),{},[dict(id='x',content='a'),dict(id='x',content='b')],{})

    def test_old_graph_order_is_preserved_with_insertions(self):
        before=dict(triples=[{'id':1},{'id':2}])
        self.assertEqual(encoding.missing_graph(before,dict(triples=[{'id':1},{'id':3},{'id':2}])),[])
        self.assertEqual(encoding.missing_graph(before,dict(triples=[{'id':2},{'id':1}])),
                         [dict(field='triples',reason='original_order_changed')])

    def test_unselected_recorded_paths_do_not_disappear_or_alias_into_view(self):
        rows=[dict(id='amu_a',content='a')]
        _,plan,aliases=encoding.translate(self.request(),{'_fine_path_groups':[['c0']]},rows,{})
        self.assertTrue(plan['_fine_path_groups'])
        self.assertEqual([g for g in plan['_fine_path_groups'] if set(g) <= set(aliases.values())],[])


if __name__ == '__main__':
    unittest.main()
