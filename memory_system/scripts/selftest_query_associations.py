"""Query hypothesis isolation, optional budgets, and a true two-sided bridge."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock,patch

import numpy as np

os.environ['AML_FAKE']='1'
os.environ['AML_MEMORY_DEBUG_LOG']=''
os.environ['AML_SEARCH_DEBUG_LOG']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import association_views as av,budget,config,query_associations as qa,schemas,store,search_pipeline


def vector(i=0):
    value=np.zeros(config.EMBED_DIM,dtype=np.float32);value[i]=1
    return value


def request():
    return schemas.SearchRequest(user_id='u',query='What should I eat for dinner?',options=['Peanut noodles','Rice'])


def response():
    return {'views':[{'source':'question','anchor':'dinner','text':'Food allergies and dietary restrictions'}]}


class Preparation(unittest.IsolatedAsyncioTestCase):
    async def test_valid_views_and_vectors_do_not_modify_the_question(self):
        req=request();before=req.model_dump()
        with patch.object(config,'FAKE',False),patch.object(qa.llm,'complete_json',AsyncMock(return_value=response())) as call, \
             patch.object(qa.embeddings,'embed',AsyncMock(return_value=[vector(2)])):
            views,vectors,diag=await qa.prepare(req)
        self.assertEqual(len(views),1);self.assertEqual(diag['status'],'ok')
        np.testing.assert_array_equal(vectors,[vector(2)])
        self.assertEqual(req.model_dump(),before)
        self.assertIn('Options are unverified hypotheses',call.call_args.args[0])

    async def test_unknown_or_nonverbatim_anchors_are_discarded_individually(self):
        raw=response();raw['views'] += [dict(source='memory:0',anchor='dinner',text='Invented fact'),
                                       dict(source='question',anchor='breakfast',text='Bad anchor')]
        rows,rejected=qa.clean(raw,qa.sources_for(request()))
        self.assertEqual(len(rows),1);self.assertEqual(rejected,2)

    async def test_options_anchor_is_hypothesis_not_evidence(self):
        raw={'views':[dict(source='option:0',anchor='Peanut',text='Possible peanut allergy restriction')]}
        self.assertEqual(qa.clean(raw,qa.sources_for(request()))[0],raw['views'])

    async def test_duplicate_and_literal_input_views_do_not_multiply(self):
        row=response()['views'][0]
        raw={'views':[row,row,dict(source='question',anchor='dinner',text=request().query)]}
        self.assertEqual(len(qa.clean(raw,qa.sources_for(request()))[0]),1)

    async def test_excess_count_is_rejected(self):
        with self.assertRaises(ValueError):
            qa.clean({'views':response()['views']*4},qa.sources_for(request()))

    async def test_no_budget_consumption_when_insufficient_headroom(self):
        with patch.object(config,'FAKE',False),patch.object(qa.llm,'complete_json',AsyncMock()) as call:
            with budget.scope(calls=2) as limits:
                _,_,diag=await qa.prepare(request())
                self.assertEqual(diag['status'],'request_budget');self.assertEqual(limits.calls,0)
            call.assert_not_called()

    async def test_failure_returns_empty_and_releases_reserved_call(self):
        with patch.object(config,'FAKE',False),patch.object(qa.llm,'complete_json',AsyncMock(side_effect=TimeoutError)):
            with budget.scope() as limits:
                views,_,diag=await qa.prepare(request())
                self.assertEqual(views,[]);self.assertEqual(diag['status'],'error')
                self.assertEqual(limits.reserved_calls,0)

    async def test_invalid_embedding_cannot_enter_recall(self):
        with patch.object(config,'FAKE',False),patch.object(qa.llm,'complete_json',AsyncMock(return_value=response())), \
             patch.object(qa.embeddings,'embed',AsyncMock(return_value=[vector()*float('nan')])):
            views,values,diag=await qa.prepare(request())
        self.assertEqual(views,[]);self.assertEqual(len(values),0);self.assertEqual(diag['status'],'error')

    async def test_large_input_is_not_truncated_or_sent(self):
        req=request().model_copy(update={'query':'x'*33000})
        with patch.object(qa.llm,'complete_json',AsyncMock()) as call:
            self.assertEqual((await qa.prepare(req))[2]['status'],'input_budget')
            call.assert_not_called()


class Recall(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.st=store.Store(':memory:')
        req=schemas.AddRequest(request_id='r',user_id='u',session_id='s',messages=[
            schemas.Message(role='user',content='I cannot eat peanuts.')])
        self.st.save_messages(req)
        self.mid=self.st.insert_amu(user_id='u',session_id='s',content='Peanut allergy',embedding=vector(0))
        self.st.link_sources(self.mid,'r',[0])
        async def generate(sources):
            return [(s,'Meal selection constraints','cannot eat peanuts') for s in sources]
        with patch.object(av,'generate',side_effect=generate),patch.object(av.embeddings,'embed',AsyncMock(return_value=[vector(2)])):
            with self.st.staged('u') as work: await av.index_sources(work,'u')

    async def asyncTearDown(self):
        self.st.conn.close()

    def recall(self,write,query,user='u'):
        with self.st.snapshot(user) as snap:
            return qa.recall(snap,user,[vector(1)],[vector(2)] if query else [],write_enabled=write)[0]

    async def test_bridge_requires_both_sides(self):
        self.assertEqual(self.recall(False,False),[])
        self.assertEqual(self.recall(True,False),[])
        self.assertEqual(self.recall(False,True),[])
        rows=self.recall(True,True)
        self.assertEqual([r['id'] for r in rows],[self.mid])
        self.assertEqual(rows[0]['content'],'Peanut allergy')
        self.assertNotIn('Meal selection constraints',str(rows))

    async def test_write_only_matches_existing_implementation(self):
        with self.st.snapshot('u') as snap:
            original=av.recall(snap,'u',[vector(2)])[0]
            integrated=qa.recall(snap,'u',[vector(2)],[],write_enabled=True)[0]
        self.assertEqual([r['id'] for r in original],[r['id'] for r in integrated])
        self.assertEqual([r['_score'] for r in original],[r['_score'] for r in integrated])

    async def test_query_only_can_retrieve_original_amus(self):
        with self.st.snapshot('u') as snap:
            rows,_=qa.recall(snap,'u',[vector(1)],[vector(0)],write_enabled=False)
        self.assertEqual([r['id'] for r in rows],[self.mid])

    async def test_query_vectors_work_when_base_embedding_fails(self):
        with self.st.snapshot('u') as snap:
            rows,_=qa.recall(snap,'u',[],[vector(2)],write_enabled=True)
        self.assertEqual([r['id'] for r in rows],[self.mid])

    async def test_cross_tenant_and_suppression(self):
        self.assertEqual(self.recall(True,True,'other'),[])
        self.st.suppress(self.mid,'2026-10-24T00:00:00Z')
        self.assertEqual(self.recall(True,True),[])

    async def test_source_dedup_and_joint_cap(self):
        row=self.st.get_amus_by_ids([self.mid])[0]
        with patch.object(config,'ASSOCIATION_RECALL_LIMIT',1):
            rows=qa.merge(self.st,[[dict(row,_score=.7)],[dict(row,_score=.9),dict(row,_score=.8)]])
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['_score'],.9)

    async def test_search_integration_keeps_original_query_and_all_base_routes(self):
        req=request();plans=[];dump=list(self.st.conn.iterdump())
        with patch.object(search_pipeline,'embed',AsyncMock(return_value=np.stack([vector(1)]*3))), \
             patch.object(search_pipeline,'_expand_recall',side_effect=lambda st,req,plan,routes,specs,vecs:routes), \
             patch.object(qa,'prepare',AsyncMock(return_value=(response()['views'],np.stack([vector(2)]),dict(status='ok',views=1)))) as prepare:
            for enabled in (False,True):
                plan=dict(intent='preference',entities=[],time_scope={})
                with patch.object(config,'ASSOCIATION_QUERY',enabled),patch.object(config,'ASSOCIATION_RECALL',True),self.st.snapshot('u') as snap:
                    routes=await search_pipeline._recall(snap,req,plan)
                plans.append((plan,routes))
            self.assertEqual(prepare.await_count,1)
        self.assertEqual(plans[0][0]['_query_specs'],plans[1][0]['_query_specs'])
        self.assertEqual([[c['id'] for c in r] for r in plans[0][1][:3]],[[c['id'] for c in r] for r in plans[1][1][:3]])
        self.assertEqual([r['id'] for r in plans[1][1][3]],[self.mid])
        self.assertEqual(list(self.st.conn.iterdump()),dump)
        self.assertNotIn('Food allergies',json.dumps(plans[1][0].get('_coverage_requirements',[])))


if __name__=='__main__': unittest.main()
