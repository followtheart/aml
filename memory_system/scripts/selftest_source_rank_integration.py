"""The real cascade sends a late outsider and maps positional output to stored IDs."""
import unittest
from unittest.mock import patch
from selftest_graph_cascade import CascadeTests, documents_in, permutation, selected, config, llm
from app import screened_source_nomination


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_outsider_reaches_main_without_changing_fine_or_output_identity(self):
        fixture=CascadeTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        rows=fixture.rows(20)
        for row in rows:row['id']='amu_long_original_'+row['id']
        original_ids=[];extra=[];observed=[]
        async def append(req,plan,coarse,fine,original,cascade):
            original_ids.extend(c['id'] for c in original)
            outsider=next(c for c in coarse if c['id'] not in {x['id'] for x in fine+original})
            extra.append(outsider['id'])
            return original+[outsider],dict(display_ids=True,status='admitted',addition=outsider['id'])
        async def rank(prompt,**kwargs):
            texts=documents_in(prompt);observed.append(prompt)
            return permutation(len(texts),reverse=True,groups=[[0,len(texts)-1]])
        plan={}
        with patch.multiple(config,SOURCE_CE_LATE_ADMISSION=True,LISTWISE_OVERFLOW_REVIEW=False,LISTWISE_SHARED_SOURCES=False), \
                patch.object(fixture.ce,'rerank',side_effect=fixture.ce_scores), \
                patch.object(screened_source_nomination,'append',side_effect=append), \
                patch.object(llm,'complete_json',side_effect=rank):
            result=selected(await fixture.cascade.rank(fixture.req(),plan,rows))
        self.assertEqual(plan['_cascade']['listwise']['candidate_ids'],original_ids+extra)
        self.assertNotIn(extra[0],plan['_cascade']['fine']['selected_ids'])
        self.assertEqual([c['id'] for c in result],list(reversed(original_ids+extra)))
        self.assertEqual(plan['_evidence_groups'],[[original_ids[0],extra[0]]])
        self.assertNotIn('amu_long_original_',observed[0])
        self.assertEqual(len(observed),1)


if __name__=='__main__':unittest.main(defaultTest='IntegrationTests')
