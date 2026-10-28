from pathlib import Path
import sys
import copy
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_context,answer_choice,config
from app.rank_visible_constraints import catalog


def row(text):
    source=dict(user_id='u',request_id='r',message_index=1,source_event_id='e',role='user',content=text)
    rendered=answer_context.with_evidence('Conversation excerpt:',[source])
    return dict(id='a',content=rendered,_rank_text=rendered,
        _packet_item=dict(id='a',memory_type='episode',content=rendered,sources=[source]))


class CatalogTests(unittest.TestCase):
    def test_complete_visible_constraint_survives_legacy_clip(self):
        r=row('Background. '*30+'Please forget that I enjoy chess.')
        with patch.multiple(config,ANSWER_CONTEXT_ITEM_MAX_CHARS=64,ANSWER_CONTEXT_MAX_CHARS=128):
            self.assertFalse(answer_choice.build_catalog([r['_packet_item']])[1])
            # The existing span parser retains the sentence's leading space.
            self.assertEqual(catalog([r])[0]['text'],' Please forget that I enjoy chess.')

    def test_hidden_source_is_not_exposed(self):
        r=row('Please forget that I enjoy chess.');r['_rank_text']='Other visible material.'
        self.assertEqual(catalog([r]),[])

    def test_role_label_must_be_visible(self):
        r=row('Please forget that I enjoy chess.');r['_rank_text']=r['_rank_text'].replace('[User]','[Assistant]')
        self.assertEqual(catalog([r]),[])

    def test_identity_text_and_input_are_preserved(self):
        r=row('Please forget that I enjoy chess.');original=copy.deepcopy(r)
        c=catalog([r])[0]
        self.assertEqual((c['request_id'],c['message_index'],c['source_event_id'],c['role']),('r',1,'e','user'))
        self.assertEqual(r,original)

    def test_display_whitespace_normalization(self):
        r=row('Please forget that I enjoy chess.')
        r['_rank_text']=' '.join(r['_rank_text'].split())
        self.assertEqual(len(catalog([r])),1)

    def test_hidden_reference_does_not_create_legacy_rule(self):
        r=row('Original material not displayed.')
        r['_packet_item']['content']='Please forget that I enjoy chess.'
        r['_rank_text']='Please forget that I enjoy chess.'
        self.assertEqual(catalog([r]),[])


if __name__=='__main__':unittest.main()
