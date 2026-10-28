"""Lexical context may add one intact source unit, never replace accepted context."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_context,config,evidence_packet,fine_context_lexical as lexical,schemas,search_pipeline
from app.fine_context_reserve import propose as original_propose


def candidate(mid,text,role='user'):
    source=dict(user_id='u',request_id='r',message_index=len(mid),source_event_id=mid,role=role,content=text)
    return dict(id=mid,_ce_score=.2,_packet_item=dict(sources=[source],
        content=answer_context.with_evidence('Generated summary.',[source]),memory_type='episode'))


class LexicalContextTests(unittest.TestCase):
    def setUp(self):
        self.base=[schemas.SearchItem(id='base',content='Accepted evidence.',packet_hash_version=4).model_dump()]
        self.fine=[candidate('first','A question about walking.'),candidate('second','A question about gardens.'),
                   candidate('unrelated','Please review this short email.'),
                   candidate('relevant','Could you revise this note: I will bring chamomile tea.'),
                   candidate('assistant','Chamomile tea for the evening.',role='assistant')]
        self.kwargs=dict(user_id='u',top_k=10,byte_budget=8000,reserve_bytes=5120)

    def test_original_reserve_preserved_and_extra_is_complete_source_not_fact(self):
        original=copy.deepcopy((self.base,self.fine))
        baseline,_=original_propose(self.base,self.fine,**self.kwargs)
        result,trace=lexical.propose(self.base,self.fine,query='Chamomile tea?',options=[],**self.kwargs)
        self.assertEqual(result[:len(baseline)],baseline)
        self.assertEqual(trace['appended_ids'],['first','second','relevant'])
        self.assertEqual(result[-1]['sources'],self.fine[3]['_packet_item']['sources'])
        self.assertNotIn('Generated summary',result[-1]['content'])
        self.assertFalse(result[-1]['personal_evidence'])
        self.assertEqual(result[-1]['coverage_ids'],[])
        self.assertEqual(result[-1]['score'],.2)
        self.assertEqual((self.base,self.fine),original)

    def test_no_matching_user_no_extra_and_caps_do_not_displace(self):
        baseline,old=original_propose(self.base,self.fine,**self.kwargs)
        for changes in (dict(query='Astronomy?',options=[]),dict(query='Chamomile',options=[],top_k=len(baseline)),
                        dict(query='Chamomile',options=[],reserve_bytes=old['added_bytes']),
                        dict(query='Chamomile',options=[],byte_budget=old['total_bytes'])):
            kw=dict(self.kwargs,**changes)
            self.assertEqual(lexical.propose(self.base,self.fine,**kw)[0],baseline)

    def test_assistant_foreign_stale_and_invisible_sources_cannot_supply_lexical_score(self):
        for kind in ('foreign','stale','invisible','unscored'):
            rows=copy.deepcopy(self.fine)
            target=rows[3]
            if kind=='foreign':target['_packet_item']['sources'][0]['user_id']='other'
            if kind=='stale':target['view_status']='stale'
            if kind=='invisible':target['_packet_item']['content']='Summary only'
            if kind=='unscored':target['_ce_score']=None
            scores,_=lexical.source_scores(rows,user_id='u',query='Chamomile',options=[])
            self.assertEqual(scores.get('relevant',0),0)
            self.assertEqual(scores.get('assistant',0),0)

    def test_duplicate_projection_does_not_change_corpus_scores(self):
        before,_=lexical.source_scores(self.fine,user_id='u',query='Chamomile tea',options=[])
        duplicate=dict(self.fine[3],id='duplicate')
        after,_=lexical.source_scores(self.fine+[duplicate],user_id='u',query='Chamomile tea',options=[])
        self.assertEqual(before,{k:after[k] for k in before})
        self.assertEqual(after['duplicate'],after['relevant'])

    def test_search_entry_uses_original_selection_and_rehashes_entire_packet(self):
        anchor=dict(id='base',_packet_item=self.base[0])
        req=schemas.SearchRequest(user_id='u',query='Chamomile tea?',top_k=10,evidence_token_budget=8000)
        plan=dict(_cascade=dict(fine=dict(selected_ids=['base']+[c['id'] for c in self.fine])),
                  _selection=dict(selected_ids=['base']),_coverage_requirements=[])
        with patch.multiple(config,PACKET_ADJACENT_CONTEXT=False,PACKET_FINE_CONTEXT_RESERVE=True,
                            PACKET_FINE_CONTEXT_5K=True,PACKET_FINE_CONTEXT_LEXICAL=True):
            packet,digest,manifest=search_pipeline._pack_evidence(None,req,copy.deepcopy(plan),[anchor],None,[anchor,*self.fine])
            self.assertEqual([p['id'] for p in packet],['base','first','second','relevant'])
            self.assertEqual(evidence_packet.digest(packet),digest)
            self.assertTrue(all(p['packet_hash']==digest for p in packet))
            self.assertEqual(manifest['conversation_context']['policy'],lexical.POLICY)
            plan['_selection']['selected_ids'].append('relevant')
            packet,_,_=search_pipeline._pack_evidence(None,req,plan,[anchor],None,[anchor,*self.fine])
            self.assertNotIn('relevant',[p['id'] for p in packet])


if __name__=='__main__':unittest.main()
