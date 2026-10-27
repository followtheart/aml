"""Check the exact renderer boundary and preservation of graph/source data."""
import json
import unittest
from compact_graph_metadata import renderer


class GraphWhitespaceTests(unittest.TestCase):
    def setUp(self):
        self.render = renderer(lambda name, **kwargs: dict(template=name, **kwargs))

    def test_graph_values_order_and_non_graph_text_preserved(self):
        graph = dict(triples=[dict(amu_id='amu_x', subject='user', object='两  个词\n"quote"',
                                  valid_from=None, valid_to='2026-10-26')],
                     associations=[dict(source='a', target='b', confidence=.25, enabled=False)],
                     recorded_paths_for_joint_review=[['a', 'b']])
        original = dict(relations=json.dumps(graph, ensure_ascii=False),
                        query='What  happened?', candidates='0: [User] a\n  b',
                        options='A. x\nB. y', requirements='[  ]')
        result = self.render('05c_listwise_rerank.txt', **original)
        self.assertEqual(json.loads(result['relations']), graph)
        self.assertLess(len(result['relations'].encode()), len(original['relations'].encode()))
        for key in original.keys() - {'relations'}:
            self.assertEqual(result[key], original[key])
        self.assertEqual(original['relations'], json.dumps(graph, ensure_ascii=False))

    def test_noncanonical_json_and_duplicate_keys_unchanged(self):
        for value in ('{ "x": 1 }', '{"x": 1, "x": 2}', '{"x": "\\u4e2d"}'):
            self.assertEqual(self.render('05c_listwise_rerank.txt', relations=value)['relations'], value)

    def test_other_templates_and_already_compact_unchanged(self):
        self.assertEqual(self.render('other.txt', relations='not JSON')['relations'], 'not JSON')
        self.assertEqual(self.render('05c_listwise_rerank.txt', relations='{}')['relations'], '{}')


if __name__ == '__main__':
    unittest.main()
