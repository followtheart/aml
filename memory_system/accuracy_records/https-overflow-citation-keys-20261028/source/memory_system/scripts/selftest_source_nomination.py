import copy
import math
import unittest

from source_nomination import nominate, wire_body


class NominationTests(unittest.TestCase):
    def setUp(self):
        self.job = dict(scheduling=dict(eligible=True), views=dict(a={}, b={}, c={}),
            nominees=[dict(id='early', source_view_ids=['a'], ce_rank=3, candidate=dict(_ce_score=.9)),
                      dict(id='late', source_view_ids=['b','c'], ce_rank=30, candidate=dict(_ce_score=.1))])

    def test_novel_source_can_change_priority_without_changing_candidates(self):
        before = copy.deepcopy(self.job)
        result = nominate(self.job, dict(a=.3,b=.2,c=.8))
        self.assertEqual(result['addition'], 'late')
        self.assertFalse(result['factual_support_inferred'])
        self.assertEqual(self.job, before)

    def test_exact_ties_keep_original_ce_order(self):
        self.assertEqual(nominate(self.job, dict(a=.8,b=.2,c=.8))['addition'], 'early')

    def test_partial_unknown_nonfinite_boolean_scores_fall_back(self):
        for scores in (dict(a=.3,b=.2), dict(a=.3,b=.2,c=.8,d=.1), dict(a=.3,b=.2,c=math.nan),
                       dict(a=.3,b=.2,c=True),dict(a=.3,b=.2,c=math.inf)):
            with self.subTest(scores=scores):
                self.assertIsNone(nominate(self.job, scores)['addition'])
        self.job['nominees'][1]['source_view_ids'] = ['missing']
        self.assertIsNone(nominate(self.job, dict(a=.3,b=.2,c=.8))['addition'])

    def test_budget_skip_cannot_use_scores(self):
        self.job['scheduling'] = dict(eligible=False, reason='token_headroom')
        result = nominate(self.job, dict(a=.3,b=.2,c=.8))
        self.assertEqual(result['status'], 'skipped')
        self.assertIsNone(result['addition'])

    def test_provider_body_preserves_order_and_complete_text(self):
        docs = ['中文\nfirst', 'second']
        self.assertEqual(wire_body('qwen3-rerank','cohere','q',docs),
                         dict(model='qwen3-rerank', query='q',documents=docs,top_n=2))
        body = wire_body('qwen3-rerank','dashscope','q',docs)
        self.assertEqual(body['input'],dict(query='q', documents=docs))
        self.assertEqual(body['parameters'],dict(top_n=2, return_documents=False))
        with self.assertRaises(ValueError):wire_body('model','unknown','q',docs)


if __name__ == '__main__':
    unittest.main()
