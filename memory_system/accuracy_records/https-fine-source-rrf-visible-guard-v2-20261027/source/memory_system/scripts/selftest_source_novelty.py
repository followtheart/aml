"""Source novelty must preserve eligibility, provenance and bounded allocation."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ['AML_FAKE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import config, schemas, search_pipeline as search, store
from analyze_overflow_https import audit_source_merge


class SourceNovelty(unittest.TestCase):
    def test_overlapping_subsets_yield_to_new_observations(self):
        routes=[[dict(id='a',content='Full text A',_score=9),dict(id='b',content='Full text B',_score=8),
                 dict(id='c',content='Full text C',_score=7)]]
        original=copy.deepcopy(routes)
        keys={'a':{('r',0),('r',1)},'b':{('r',0)},'c':{('r',2)}}
        trace={}
        chosen=search._merge_source_query_results(routes,2,keys,novel_first=True,trace=trace)
        self.assertEqual([v['id'] for v in chosen],['a','c'])
        self.assertEqual(trace['baseline_ids'],['a','b'])
        self.assertEqual(routes,original)
        self.assertEqual([v['content'] for v in chosen],['Full text A','Full text C'])
        t=dict(versions=dict(settings=dict(SOURCE_RECALL_DIVERSITY=True,SOURCE_RECALL_NOVELTY=True,
            RECALL_SOURCE_LIMIT=2)),routes=[dict(channel='source_text',candidates=chosen,source_merge=trace)])
        audit_source_merge(t)
        broken=copy.deepcopy(t);broken['routes'][0]['candidates'][1]['id']='b'
        with self.assertRaises(ValueError):audit_source_merge(broken)

    def test_additional_sources_and_unlinked_candidates_stay_eligible(self):
        rows=[[dict(id=k) for k in ('a','b','c','d')]]
        keys={'a':{('r',0)},'b':{('r',0),('r',1)},'c':{('r',0)}}
        result=search._merge_source_query_results(rows,3,keys,novel_first=True)
        self.assertEqual([v['id'] for v in result],['a','b','d'])
        full=search._merge_source_query_results(rows,4,keys,novel_first=True)
        self.assertEqual([v['id'] for v in full],['a','b','d','c'])
        self.assertEqual(search._merge_source_query_results(rows,0,keys,novel_first=True),[])

    def test_entrypoint_flags_and_source_trace(self):
        st=store.Store(':memory:');self.addCleanup(st.conn.close)
        st.save_messages(schemas.AddRequest(request_id='r',user_id='u',session_id='s',messages=[
            schemas.Message(role='user',content=f'OrchidLedger observation {i}.') for i in range(3)]))
        ids=[]
        for indices in ([0,1],[0],[2]):
            mid=st.insert_amu(user_id='u',session_id='s',content='OrchidLedger',type='fact')
            st.link_sources(mid,'r',indices);ids.append(mid)
        rows=[dict(v,_score=1) for mid in ids for v in st.get_amus_by_ids([mid])]
        req=schemas.SearchRequest(user_id='u',query='OrchidLedger')
        for diversity,novelty,expected in [(True,False,ids[:2]),(True,True,[ids[0],ids[2]]),(False,True,ids[:2])]:
            plan={'_include_history':False,'_include_sensitive':False}
            with patch.multiple(config,SOURCE_RECALL_DIVERSITY=diversity,SOURCE_RECALL_NOVELTY=novelty,
                    RECALL_SOURCE_LIMIT=2),patch.object(st,'source_search',return_value=copy.deepcopy(rows)), \
                    patch.object(st,'fts_search',return_value=[]):
                _,found=search._lexical_recall(st,req,plan,[dict(text='OrchidLedger')])
            self.assertEqual([v['id'] for v in found],expected)
            search._route(plan,[],'source_text',found)
            self.assertEqual('source_merge' in plan['_routes'][0],diversity)
            if diversity:
                audit_source_merge(dict(versions=dict(settings=dict(SOURCE_RECALL_DIVERSITY=True,
                    SOURCE_RECALL_NOVELTY=novelty,RECALL_SOURCE_LIMIT=2)),routes=plan['_routes']))

    def test_real_store_filters_before_novel_allocation(self):
        st=store.Store(':memory:');self.addCleanup(st.conn.close)
        st.save_messages(schemas.AddRequest(request_id='r',user_id='u',session_id='s',messages=[
            schemas.Message(role='user',content='OrchidLedger public.'),
            schemas.Message(role='user',content='OrchidLedger private.',sensitivity='sensitive'),
            schemas.Message(role='user',content='OrchidLedger suppressed.',sensitivity='suppressed')]))
        ids=[]
        for index in (0,1,2,0,0):
            mid=st.insert_amu(user_id='u',session_id='s',content='OrchidLedger',type='fact')
            st.link_sources(mid,'r',[index]);ids.append(mid)
        st.conn.execute("UPDATE amu SET valid_to='2020-01-01' WHERE id=?",(ids[3],))
        st.conn.execute("UPDATE amu SET resolution_status='retracted' WHERE id=?",(ids[4],))
        st.conn.commit()
        with patch.multiple(config,SOURCE_RECALL_DIVERSITY=True,SOURCE_RECALL_NOVELTY=True):
            for user,history,expected in [('u',False,{ids[0]}),('u',True,{ids[0],ids[3]}),('other',False,set())]:
                plan={'_include_history':history,'_include_sensitive':False}
                _,found=search._lexical_recall(st,schemas.SearchRequest(user_id=user,query='OrchidLedger'),
                    plan,[dict(text='OrchidLedger')])
                self.assertEqual({v['id'] for v in found},expected)


if __name__=='__main__':unittest.main()
