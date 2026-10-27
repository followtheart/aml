"""Source route provenance and packet bounds, independent of gold labels."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice, answer_context, evidence_packet
from replay_source_vector_route import source_route


class SourceRoute(unittest.TestCase):
    def setUp(self):
        self.source = dict(request_id='r',message_index=0,role='user',
                           content='我不喜欢拼图。 This is my own wording.',timestamp='2025-01-02',
                           source_event_id='event',amu_ids=['eligible-parent'])
        self.probe = dict(sources=[self.source],rankings={'q':{'user':[dict(source_index=0)]}})
        self.question = dict(query_specs=[dict(text='q')])
        rule_source = dict(request_id='r',message_index=1,role='user',content='Please forget that I like tea.',timestamp=None)
        self.packet,_,_ = evidence_packet.pack([dict(id='forget',is_constraint=True,memory_type='rule',
            sources=[rule_source],content=answer_context.with_evidence('Forget rule',[rule_source]))],50,32000)

    def test_original_role_time_text_and_constraint_preserved(self):
        original = copy.deepcopy((self.probe,self.packet))
        packet,trace = source_route(self.probe,self.question,self.packet,32000)
        self.assertEqual((self.probe,self.packet),original)
        self.assertEqual(packet[0]['id'],'forget')
        added=packet[1];self.assertIsNone(added['personal_evidence'])
        self.assertEqual(added['linked_memory_ids'],['eligible-parent'])
        self.assertEqual(added['sources'][0],{k:v for k,v in self.source.items() if k!='amu_ids'})
        self.assertIn(self.source['content'],answer_context.build(packet))
        sources, constraints = answer_choice.build_catalog(packet)
        self.assertTrue(any(s['text']==self.source['content'] for s in sources.values()))
        self.assertTrue(constraints)
        self.assertEqual(evidence_packet.digest(packet),trace['packet_hash'])

    def test_whole_sources_omitted_when_they_do_not_fit(self):
        limit=len(self.packet[0]['content'].encode())
        packet,trace=source_route(self.probe,self.question,self.packet,limit)
        self.assertEqual([m['id'] for m in packet],['forget'])
        self.assertFalse(trace['admitted_source_ids'])
        self.assertLessEqual(trace['manifest']['token_upper_bound'],limit)

    def test_no_assistant_or_unlinked_source_promoted(self):
        for changes in (dict(role='assistant'),dict(amu_ids=[])):
            probe=copy.deepcopy(self.probe);probe['sources'][0].update(changes)
            with self.assertRaises(ValueError):source_route(probe,self.question,self.packet,32000)


if __name__=='__main__':unittest.main()
