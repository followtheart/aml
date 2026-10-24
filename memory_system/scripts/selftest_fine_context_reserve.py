"""A retrieval reserve cannot invent facts, revive sources or displace evidence."""
import copy
import unittest
from unittest.mock import patch
from probe_fine_context_reserve import propose,answer_context
from app import config,conversation_context,evidence_packet,schemas,search_pipeline


class FineContextReserveTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.multiple(config, PACKET_FINE_CONTEXT_RESERVE=False,
            PACKET_FINE_CONTEXT_5K=False, PACKET_FINE_CONTEXT_LEXICAL=False, PACKET_FINE_CONTEXT_FILL=False))
        self.source=dict(user_id='u',request_id='r',message_index=0,source_event_id='s0',
                         role='user',content='Why do shaded paths feel cooler?')
        self.candidate=dict(id='context',_ce_score=0.8,_packet_item=dict(
            content=answer_context.with_evidence('Generated personal summary.',[self.source]),
            sources=[self.source],personal_evidence=True,coverage_ids=['option:0'],equivalent_ids=['fact']))
        self.base=[dict(id='accepted',content='Previously accepted evidence.',sources=[])]

    def run_proposal(self,**kwargs):
        return propose(self.base,[self.candidate],user_id='u',top_k=10,byte_budget=8000,**kwargs)

    def test_original_prefix_and_source_only_context_without_fact_upgrade(self):
        original=copy.deepcopy((self.base,self.candidate))
        packet,trace=self.run_proposal()
        self.assertEqual(trace['appended_ids'],['context'])
        self.assertEqual(packet[0],self.base[0])
        self.assertEqual(packet[1]['sources'],[self.source])
        self.assertNotIn('Generated personal summary',packet[1]['content'])
        self.assertFalse(packet[1]['personal_evidence'])
        self.assertEqual(packet[1]['coverage_ids'],[])
        self.assertEqual(packet[1]['equivalent_ids'],[])
        self.assertEqual((self.base,self.candidate),original)

    def test_stale_unscored_foreign_invisible_and_partial_user_rejected(self):
        original=copy.deepcopy(self.candidate)
        for mode in ('stale','unscored','foreign','invisible','partial','assistant_only'):
            with self.subTest(mode=mode):
                self.candidate=copy.deepcopy(original)
                src=self.candidate['_packet_item']['sources'][0]
                if mode=='stale':self.candidate['view_status']='stale'
                if mode=='unscored':self.candidate['_ce_score']=None
                if mode=='foreign':src['user_id']='other'
                if mode=='invisible':self.candidate['_packet_item']['content']='Only summary'
                if mode=='partial':src['content_span']=dict(start=2,end=2+len(src['content']),original_length=100)
                if mode=='assistant_only':src['role']='assistant'
                self.assertEqual(self.run_proposal()[0],self.base)

    def test_existing_user_and_capacity_never_displace_original(self):
        self.assertEqual(self.run_proposal(limit=0)[0],self.base)
        self.assertEqual(self.run_proposal(reserve_bytes=0)[0],self.base)
        for top_k,byte_budget in ((1,8000),(10,len(self.base[0]['content'].encode()))):
            self.assertEqual(propose(self.base,[self.candidate],user_id='u',top_k=top_k,byte_budget=byte_budget)[0],self.base)
        self.base[0]['sources']=[self.source]
        self.assertEqual(self.run_proposal()[0],self.base)

    def test_shared_user_is_not_counted_twice_and_total_reserve_is_bounded(self):
        other=dict(self.candidate,id='duplicate')
        packet,trace=propose(self.base,[self.candidate,other],user_id='u',top_k=10,byte_budget=8000)
        self.assertEqual(trace['appended_ids'],['context'])
        self.assertEqual(trace['added_bytes'],len(packet[1]['content'].encode())+1)
        self.assertEqual(trace['total_bytes'],sum(len(p['content'].encode()) for p in packet)+len(packet)-1)

    def test_search_entry_uses_eligible_fine_and_updates_hash_without_support_upgrade(self):
        anchor=dict(id='accepted',_packet_item=dict(self.base[0],memory_type='episode'))
        req=schemas.SearchRequest(user_id='u',query='Where could I walk?',top_k=10,evidence_token_budget=8000)
        plan=dict(_cascade=dict(fine=dict(selected_ids=['accepted','context'])),
                  _selection=dict(selected_ids=['accepted']),_coverage_requirements=[])
        before=copy.deepcopy((anchor,self.candidate))
        with patch.multiple(config,PACKET_ADJACENT_CONTEXT=False,PACKET_FINE_CONTEXT_RESERVE=False):
            baseline,base_hash,base_manifest=search_pipeline._pack_evidence(None,req,copy.deepcopy(plan),[anchor],None,[anchor,self.candidate])
            with patch.object(config,'PACKET_FINE_CONTEXT_RESERVE',True):
                packet,digest,manifest=search_pipeline._pack_evidence(None,req,copy.deepcopy(plan),[anchor],None,[anchor,self.candidate])
                self.assertEqual([p['id'] for p in packet],['accepted','context'])
                self.assertEqual(dict(packet[0],packet_hash=base_hash),baseline[0])
                self.assertEqual(evidence_packet.digest(packet),digest)
                self.assertTrue(all(p['packet_hash']==digest for p in packet))
                self.assertEqual(manifest['included_ids'],['accepted','context'])
                self.assertEqual(manifest['evidence_count'],base_manifest['evidence_count']+1)
                for key in ('covered_requirements','missing_requirements','core_bytes','constraint_bytes','constraint_count'):
                    self.assertEqual(manifest[key],base_manifest[key])
                self.assertEqual(manifest['conversation_context']['policy'],'fine_original_context_v1')
                plan['_cascade']['fine']['selected_ids']=['accepted']
                rejected,_,_=search_pipeline._pack_evidence(None,req,copy.deepcopy(plan),[anchor],None,[anchor,self.candidate])
                self.assertEqual([p['id'] for p in rejected],['accepted'])
                plan['_cascade']['fine']['selected_ids'].append('context')
                plan['_selection']['selected_ids'].append('context')
                rejected,_,_=search_pipeline._pack_evidence(None,req,copy.deepcopy(plan),[anchor],None,[anchor,self.candidate])
                self.assertEqual([p['id'] for p in rejected],['accepted'])
                plan['_selection']['selected_ids'].remove('context')
                with patch.object(search_pipeline,'_foresight_filter',side_effect=lambda rows,_:[r for r in rows if r['id']!='context']):
                    rejected,_,_=search_pipeline._pack_evidence(None,req,plan,[anchor],None,[anchor,self.candidate])
                    self.assertEqual([p['id'] for p in rejected],['accepted'])
        self.assertEqual((anchor,self.candidate),before)

    def test_capacity_extension_preserves_full_quotes_and_original_total_caps(self):
        self.base=[schemas.SearchItem.model_validate(dict(self.base[0],packet_hash_version=4)).model_dump()]
        fine=[]
        for i in range(3):
            source=dict(self.source,content='原文' * 360,source_event_id=f's{i}',message_index=i)
            fine.append(dict(self.candidate,id=f'c{i}',_packet_item=dict(
                content=answer_context.with_evidence('Summary',[source]),sources=[source])))
        original=copy.deepcopy(fine)
        base_hash=evidence_packet.digest(self.base)
        manifest=dict(token_budget=8000,token_upper_bound=len(self.base[0]['content'].encode()),
                      included_ids=['accepted'],evidence_count=1)
        def attach(extended,**changes):
            return conversation_context.attach(self.base,base_hash,dict(manifest,**changes),fine,
                user_id='u',top_k=10,enabled=False,reserve_enabled=True,reserve_extended=extended)
        before,_,_,old=attach(False)
        after,digest,updated,new=attach(True)
        self.assertEqual(old['appended_ids'],['c0'])
        self.assertEqual(new['appended_ids'],['c0','c1'])
        self.assertEqual(new['policy'],'fine_original_context_5k_v1')
        self.assertEqual(new['reserve_bytes'],5120)
        self.assertLessEqual(new['added_bytes'],5120)
        self.assertEqual(evidence_packet.digest(after),digest)
        self.assertEqual(after[0]['content'],self.base[0]['content'])
        self.assertEqual([p['sources'] for p in after[1:]],[c['_packet_item']['sources'] for c in fine[:2]])
        self.assertEqual(len(attach(True,token_budget=4000)[0]),2)
        self.assertEqual(fine,original)
        inactive=conversation_context.attach(self.base,base_hash,manifest,fine,user_id='u',top_k=10,
            enabled=False,reserve_enabled=False,reserve_extended=True)
        self.assertEqual(inactive[0],self.base)
        self.assertFalse(inactive[3]['enabled'])


if __name__=='__main__':
    unittest.main()
