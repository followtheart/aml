"""Source ordering must not remove context or weaken quotation validation."""
import copy
import json
import unittest
from unittest.mock import patch

from probe_witness_source_order import ordered_request,batches_for,decode,config


class SourceOrder(unittest.TestCase):
    def setUp(self):
        self.cards=[dict(id='s0',role='user',text='I enjoy painting.'),
                    dict(id='s1',role='assistant',text='Cycling can be pleasant.'),
                    dict(id='s2',role='user',text='I enjoy walking.')]
        self.request=dict(prompt='Question <sources>\n'+json.dumps(self.cards,ensure_ascii=False)+'\n</sources> end',
                          schema={'type':'object'},system='Do not answer.',max_tokens=3072)

    def test_all_text_ids_roles_and_contract_preserved(self):
        before=copy.deepcopy(self.request)
        result,order=ordered_request(self.request,self.cards,[.2,.9,.4])
        self.assertEqual(order,['s1','s2','s0'])
        block=result['prompt'].split('<sources>\n')[1].split('\n</sources>')[0]
        self.assertEqual({c['id']:c for c in json.loads(block)},{c['id']:c for c in self.cards})
        for key in ('schema','system','max_tokens'):
            self.assertEqual(result[key],before[key])
        self.assertEqual(self.request,before)
        self.assertEqual(len(result['prompt'].encode()),len(before['prompt'].encode()))

    def test_ties_keep_original_order(self):
        self.assertEqual(ordered_request(self.request,self.cards,[.5,.5,.5])[0],self.request)

    def test_bad_scores_rejected(self):
        for scores in ([.5],[.5,.2,float('nan')],[.5,True,.2]):
            with self.assertRaises(ValueError):
                ordered_request(self.request,self.cards,scores)

    def test_batch_boundaries_do_not_drop_sources(self):
        with patch.object(config,'CE_BATCH_SIZE',2):
            self.assertEqual(batches_for('art',self.cards),[[0,1],[2]])
        with patch.object(config,'CE_MAX_DOCUMENT_BYTES',3),self.assertRaises(ValueError):
            batches_for('art',self.cards)

    def test_reordered_source_does_not_authorize_cross_source_quote(self):
        job=dict(options=['A. Try painting.'],sources={c['id']:c for c in self.cards},
                 coverage_witnesses=False,packet=[])
        response=dict(options=[dict(letter='A',witnesses=[dict(source_id='s1',quote='I enjoy painting.')])])
        out=decode(response,job)
        self.assertEqual(out['status'],'fallback')
        self.assertFalse(out['semantic'])
        self.assertEqual(out['validation']['rejected_witnesses'][0]['reason'],'quote_not_in_source')


if __name__=='__main__':
    unittest.main()
