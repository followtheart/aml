"""Supplied-text relevance guidance preserves sources and factual validators."""
import copy
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ['AML_FAKE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_context,budget,config,llm,listwise_recovery as recovery,recovery_spans as spans


class TransformContext(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.req=SimpleNamespace(query='What could make a room welcoming?',options=['A. Display photos.'])
        self.text='Please translate the following passage.\n\nEn la sala, las fotos de la familia ocupan un lugar especial.'
        self.quote=self.text.split('\n\n')[1]
        self.cards=[dict(candidate_id='one',sources=[dict(source_id='s0',role='user',text=self.text,
                        request_id='r',message_index=0)])]

    def result(self,kind='third_party_context',part='s0.p1'):
        return dict(c0=dict(useful=True,target_id='option:0',evidence_kind=kind,citation_parts=[part]))

    def test_guidance_changes_only_system_and_accounts_for_all_bytes(self):
        before=copy.deepcopy(self.cards)
        for span_refs in (False,True):
            for role_bound in (False,True):
                with self.subTest(span_refs=span_refs,role_bound=role_bound),patch.multiple(config,
                        LISTWISE_SPAN_REFS=span_refs,LISTWISE_ROLE_BOUND_REFS=role_bound,LISTWISE_TRANSFORM_CONTEXT=False):
                    control=recovery.request_for(self.req,self.cards)
                    with patch.object(config,'LISTWISE_TRANSFORM_CONTEXT',True):
                        treatment=recovery.request_for(self.req,self.cards)
                    for field in ('prompt','schema','max_tokens'):
                        self.assertEqual(control[field],treatment[field])
                    self.assertEqual(treatment['system'],control['system']+spans.TRANSFORM_CONTEXT_GUIDANCE)
                    self.assertEqual(treatment['reservation']-control['reservation'],len(spans.TRANSFORM_CONTEXT_GUIDANCE.encode()))
        self.assertEqual(before,self.cards)

    def test_multilingual_material_can_be_context_without_personal_assertion(self):
        built=spans.build(self.req,self.cards,transform_context=True)
        kept,judgments=spans.decode_partial(self.result(),built)
        self.assertEqual(kept,{'one'})
        self.assertEqual(judgments[0]['citations'][0]['quote'],self.quote)
        kept,judgments=spans.decode_partial(self.result('user_context'),built)
        self.assertEqual(kept,set())
        self.assertIn('no_user_assertion',judgments[0]['validation_errors'])

    def test_explicit_other_speaker_not_promoted_to_current_user(self):
        cards=copy.deepcopy(self.cards)
        cards[0]['sources'][0]['text']='Please polish this quotation.\nAlex wrote: I collect family photos.'
        built=spans.build(self.req,cards,transform_context=True)
        kept,judgments=spans.decode_partial(self.result('user_context'),built)
        self.assertFalse(kept)
        self.assertIn('not_actual_user_context',judgments[0]['validation_errors'])
        kept,_=spans.decode_partial(self.result(),built)
        self.assertEqual(kept,{'one'})

    def test_wrong_owner_and_assistant_user_label_still_rejected(self):
        built=spans.build(self.req,self.cards,transform_context=True)
        kept,judgments=spans.decode_partial(self.result(part='not-owned'),built)
        self.assertFalse(kept)
        self.assertIn('span_reference_error',judgments[0]['validation_errors'])
        cards=copy.deepcopy(self.cards)
        cards[0]['sources'][0].update(role='assistant',text='I collect family photos.')
        kept,judgments=spans.decode_partial(self.result('user_context','s0.p0'),
                                          spans.build(self.req,cards,transform_context=True))
        self.assertFalse(kept)
        self.assertIn('not_user_source',judgments[0]['validation_errors'])

    async def test_real_entry_keeps_original_context_and_no_factual_marks(self):
        source=dict(request_id='r',message_index=0,source_event_id='s',role='user',content=self.text)
        row=dict(id='one',content='A passage supplied for translation.',_rank_text=answer_context.with_evidence('',[source]),
                 _packet_item=dict(sources=[source],personal_evidence=False),_supported_coverage_ids=[])
        before=copy.deepcopy(row)
        systems=[]
        async def model(prompt,**kwargs):
            budget.current.get().before_call()
            systems.append(kwargs['system'])
            return self.result()
        with patch.multiple(config,LISTWISE_SPAN_REFS=True,LISTWISE_TRANSFORM_CONTEXT=True),patch.object(llm,'complete_json',model):
            with budget.scope(seconds=30,calls=1,tokens=40000) as limits:
                selected,trace=await recovery.recover(self.req,[row],[],{0})
            self.assertEqual(selected,[before])
            self.assertEqual(row,before)
            self.assertEqual(trace['context_policy'],'supplied_text_relevance_v1')
            self.assertEqual(trace['restored_ids'],['one'])
            self.assertTrue(systems[0].endswith(spans.TRANSFORM_CONTEXT_GUIDANCE))
            self.assertEqual((limits.calls,limits.reserved_calls,limits.reserved_tokens),(1,0,0))


if __name__=='__main__':
    unittest.main()
