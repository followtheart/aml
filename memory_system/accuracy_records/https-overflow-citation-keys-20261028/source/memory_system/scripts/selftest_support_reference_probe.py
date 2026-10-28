"""Reference decoding must preserve source identity and literal quotation."""
import copy
import json
import unittest
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice
from support_reference_probe import request, decode


class ReferenceTransport(unittest.TestCase):
    def setUp(self):
        self.sources=[dict(id='s0',role='user',text='I do not own a camera. My sister owns one.',timestamp=42),
                      dict(id='s1',role='assistant',text='You could borrow a camera.',timestamp=43)]
        self.prompt='<sources>\n'+json.dumps(self.sources)+'\n</sources>'
        self.actual,self.schema,self.registry=request(self.prompt,answer_choice.Assessments.model_json_schema())
        self.payload={'options':[{'letter':'A','claims':[{'citations':[dict(
            source_id='s0',span_id='s0.p0',basis='self_report',subject='current_user')]}]}]}

    def test_exact_negation_and_no_mutation(self):
        original=copy.deepcopy(self.payload)
        result=decode(self.payload,self.registry)
        self.assertEqual(self.payload,original)
        self.assertEqual(result['options'][0]['claims'][0]['citations'][0]['quote'],'I do not own a camera.')

    def test_unknown_cross_source_and_mixed_format_rejected(self):
        for patch in [dict(span_id='missing'),dict(source_id='s1'),dict(quote='I own a camera.')]:
            payload=copy.deepcopy(self.payload)
            payload['options'][0]['claims'][0]['citations'][0].update(patch)
            with self.subTest(patch=patch),self.assertRaises(ValueError):decode(payload,self.registry)

    def test_complete_sources_roles_time_and_schema(self):
        cards=json.loads(self.actual.split('<sources>\n')[1].split('\n</sources>')[0])
        for original,card in zip(self.sources,cards):
            self.assertEqual(''.join(p['text'] for p in card['parts']),original['text'])
            self.assertEqual((card['role'],card['timestamp']),(original['role'],original['timestamp']))
        citation=self.schema['$defs']['Citation']
        self.assertNotIn('quote',citation['properties'])
        self.assertIn('span_id',citation['required'])


if __name__=='__main__':unittest.main()
