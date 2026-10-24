"""Four-arm diagnostics share CE scores and preserve per-arm budget costs."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock,patch

os.environ['AML_FAKE']='1'
os.environ['AML_MEMORY_DEBUG_LOG']=''
os.environ['AML_SEARCH_DEBUG_LOG']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget,config,schemas,store
import diagnose_dual_association as diagnostic


class SharedScores(unittest.IsolatedAsyncioTestCase):
    def prepared(self,texts):
        req=schemas.SearchRequest(user_id='u',query='dinner choices')
        plan=dict(_coverage_requirements=[])
        rows=[dict(id=f'a{i}',type='fact',content=text,_rank_text=text,_fused=.9) for i,text in enumerate(texts)]
        return req,plan,rows,{}

    async def test_identical_documents_are_scored_once_across_arms(self):
        arms=dict(C00=self.prepared(['rice','nuts']),C10=self.prepared(['nuts','fish']),
                  C01=self.prepared(['rice']),C11=self.prepared(['rice','fish']))
        async def score(query,docs,**kwargs): return [len(d)/10 for d in docs]
        with patch.object(diagnostic.cross_encoder,'rerank',side_effect=score) as ce:
            query,scores,errors,usage=await diagnostic.shared_scores(arms)
        documents=[d for call in ce.call_args_list for d in call.args[1]]
        self.assertEqual(set(documents),{'rice','nuts','fish'})
        self.assertEqual(len(documents),3);self.assertEqual(errors,[])
        self.assertEqual(scores['rice'],.4)

    async def test_changed_ce_question_is_rejected(self):
        first=self.prepared(['rice']);second=self.prepared(['rice'])
        second[0].query='another question'
        with self.assertRaises(ValueError):
            await diagnostic.shared_scores(dict(C00=first,C11=second))

    async def test_failed_provider_batch_is_not_replaced_by_invented_scores(self):
        with patch.object(diagnostic.cross_encoder,'rerank',AsyncMock(side_effect=TimeoutError)):
            _,scores,errors,_=await diagnostic.shared_scores(dict(C00=self.prepared(['rice'])))
        self.assertEqual(scores,{})
        self.assertEqual(len(errors),1)

    async def test_query_preparation_is_charged_to_the_ranking_arm(self):
        st=store.Store(':memory:')
        try:
            mid=st.insert_amu(user_id='u',session_id='s',content='rice for dinner')
            req=schemas.SearchRequest(user_id='u',query='dinner choices')
            def make():
                plan=dict(_coverage_requirements=[])
                rows=[dict(st.get_amus_by_ids([mid])[0],_rank_text='rice for dinner',_fused=.8)]
                return req,plan,rows,{}
            prepared=make();query=diagnostic.cascade_rerank.query_text(req,prepared[1])
            baseline=await diagnostic.rank(st,prepared,query,{'rice for dinner':.9},{})
            treatment=await diagnostic.rank(st,make(),query,{'rice for dinner':.9},dict(calls=2,tokens=300,seconds=.1))
            self.assertEqual(treatment['simulated_calls']-baseline['simulated_calls'],2)
            self.assertEqual(treatment['simulated_tokens']-baseline['simulated_tokens'],300)
            self.assertEqual(baseline['stages'],treatment['stages'])
            self.assertEqual(baseline['stages']['main_input'],[mid])
            self.assertIsNone(budget.current.get())
        finally: st.conn.close()


if __name__=='__main__': unittest.main()
