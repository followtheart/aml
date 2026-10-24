"""Conversation display links must preserve provenance and citation boundaries."""
import copy
import json
from types import SimpleNamespace
import unittest
from probe_conversation_links import conversation_links, proposal, spans, recovery


class ConversationLinkTests(unittest.TestCase):
    def setUp(self):
        self.req = SimpleNamespace(query='Where could I walk?', options=[])
        self.user = dict(source_id='s0',role='user',text='Why are wooded trails cooler?',
                         request_id='r',message_index=0,source_event_id='u0')
        self.reply = dict(source_id='s0',role='assistant',text='The trees provide shade.',
                          request_id='r',message_index=1,source_event_id='a1')
        self.cards = [dict(candidate_id='question',sources=[self.user]),
                      dict(candidate_id='reply',sources=[self.reply])]

    def test_exact_display_ownership_and_input_objects_unchanged(self):
        built = spans.build(self.req,self.cards)
        before = copy.deepcopy(built)
        request = recovery.request_for(self.req,self.cards)
        result, links = proposal(request,built)
        self.assertEqual(links,[dict(user_source='s0',assistant_source='s1')])
        prefix, raw = result['prompt'].rsplit('\n',1)
        payload = json.loads(raw)
        self.assertEqual(payload.pop('reply_to'),[['s1','s0']])
        self.assertEqual(prefix+'\n'+json.dumps(payload,ensure_ascii=False),request['prompt'])
        self.assertEqual(result['schema'],request['schema'])
        self.assertEqual(result['max_tokens'],request['max_tokens'])
        self.assertEqual(built,before)
        self.assertGreater(result['reservation'],request['reservation'])

    def test_cross_request_nonadjacent_unknown_and_ambiguous_metadata_cannot_link(self):
        original = copy.deepcopy(self.reply)
        for field,value in [('request_id','other'),('message_index',3),('message_index',True),
                            ('message_index',-1),('source_event_id',None),('source_event_id','u0')]:
            with self.subTest(field=field,value=value):
                self.reply.clear();self.reply.update(original);self.reply[field]=value
                built=spans.build(self.req,self.cards)
                self.assertEqual(conversation_links(built),[])
                request=recovery.request_for(self.req,self.cards)
                self.assertEqual(proposal(request,built)[0],request)
        self.reply.clear();self.reply.update(original)
        self.cards.append(dict(candidate_id='conflicting',sources=[dict(self.reply,source_event_id='different')]))
        self.assertEqual(conversation_links(spans.build(self.req,self.cards)),[])

    def test_shared_excerpts_do_not_duplicate_links_or_extend_citation_rights(self):
        self.cards.append(dict(candidate_id='duplicate',sources=[copy.deepcopy(self.reply)]))
        built=spans.build(self.req,self.cards)
        self.assertEqual(conversation_links(built),[dict(user_source='s0',assistant_source='s1')])
        output={key:dict(useful=False,target_id='',evidence_kind='none',citation_parts=[])
                for key in built['candidate_map']}
        output['c0']=dict(useful=True,target_id='question',evidence_kind='topic_interest',citation_parts=['s0.p0'])
        self.assertEqual(spans.decode_partial(output,built)[0],{'question'})
        output['c1']=dict(useful=True,target_id='question',evidence_kind='topic_interest',citation_parts=['s0.p0'])
        selected, judgments=spans.decode_partial(output,built)
        self.assertEqual(selected,{'question'})
        self.assertIn('span_reference_error',next(j for j in judgments if j['candidate_id']=='reply')['validation_errors'])



if __name__ == '__main__':
    unittest.main()
