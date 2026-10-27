"""Boundaries for complete alternative text in the CE query."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, rerank_query


class CompleteOptionsTests(unittest.TestCase):
    def setUp(self):
        self.req=SimpleNamespace(query='What activity could help?', options=[
            'A. Since you no longer cycle after an injury, consider a gentle activity.',
            'B. Frame your family photos; this does not establish ownership.'])
        self.plan={'sub_queries':['What activity could help?']}

    def test_disabled_preserves_legacy_query(self):
        with patch.object(config,'CE_FULL_OPTION_TEXT',False):
            self.assertEqual(rerank_query.build(self.req,self.plan),rerank_query._base_query(self.req,self.plan))

    def test_every_original_option_and_qualifier_preserved_in_order(self):
        before=copy.deepcopy((self.req,self.plan))
        with patch.object(config,'CE_FULL_OPTION_TEXT',True):
            value=rerank_query.build(self.req,self.plan)
        original=rerank_query._base_query(self.req,self.plan)
        self.assertTrue(value.startswith(original))
        self.assertEqual(json.loads(value.rsplit('\n',1)[1]),self.req.options)
        self.assertIn('not established user facts',value)
        self.assertLessEqual(len(value[len(self.req.query):].encode()),1800)
        self.assertEqual((self.req,self.plan),before)

    def test_overlong_last_option_causes_whole_suffix_fallback(self):
        self.req.options[-1]='B. '+'word '*500+' but never infer this is my preference.'
        original=rerank_query._base_query(self.req,self.plan)
        self.assertEqual(rerank_query.with_complete_options(self.req,original),original)

    def test_unicode_budget_uses_bytes(self):
        self.req.options=['A. '+'汉'*400,'B. '+'字'*400]
        self.assertEqual(rerank_query.with_complete_options(self.req,self.req.query),self.req.query)

    def test_prior_premises_share_same_auxiliary_budget(self):
        original=self.req.query+'x'*1700
        self.assertEqual(rerank_query.with_complete_options(self.req,original),original)

    def test_empty_malformed_or_excessive_options_leave_query_untouched(self):
        for options in (None,[],['one'],['one',''],['one',False],['x']*9):
            with self.subTest(options=options):
                self.req.options=options
                self.assertEqual(rerank_query.with_complete_options(self.req,self.req.query),self.req.query)

    def test_embedded_quotes_newlines_and_instructions_are_literal_data(self):
        self.req.options=['A. He said "ignore rules".\nThat is a quotation.','B. Other choice.']
        value=rerank_query.with_complete_options(self.req,self.req.query)
        self.assertEqual(json.loads(value.rsplit('\n',1)[1]),self.req.options)


if __name__=='__main__':
    unittest.main()
