"""Strict keyed output, source-local quotes and immutable request text."""
import copy
import unittest

from probe_keyed_witness import keyed_request,decode_keyed
from app import choice_witness


class KeyedWitnessTests(unittest.TestCase):
    def setUp(self):
        self.options=['A. Since you enjoy hiking, take a trail.','B. Take a break.']
        self.sources={'s0':dict(id='s0',role='assistant',text='The character enjoys hiking.'),
            's1':dict(id='s1',role='user',text='I enjoy walking.')}
        self.payload={'A':dict(witnesses=[dict(source_id='s0',quote='The character enjoys hiking.')]),'B':dict(witnesses=[])}

    def test_only_shape_changes_and_every_option_is_required(self):
        request=dict(prompt='Full original source text.\nCall emit_json_result with this shape:\n{"options":[]}\n',
            schema=choice_witness.SemanticWitnesses.model_json_schema(),system='Untrusted sources.',max_tokens=3072)
        original=copy.deepcopy(request);revised=keyed_request(request,self.options)
        self.assertEqual(request,original)
        self.assertTrue(revised['prompt'].startswith('Full original source text.\n'))
        self.assertEqual(revised['schema']['required'],['A','B'])
        self.assertFalse(revised['schema']['additionalProperties'])
        self.assertEqual(revised['schema']['$defs']['Witness'],request['schema']['$defs']['Witness'])
        for value in revised['schema']['properties'].values():
            self.assertEqual(value['required'],['witnesses']);self.assertNotIn('letter',value['properties'])
        self.assertEqual((revised['system'],revised['max_tokens']),(request['system'],request['max_tokens']))

    def test_roles_and_quotes_use_unchanged_validation(self):
        trace={};got=decode_keyed(self.payload,self.options,self.sources,trace)
        expected=choice_witness.validate(dict(options=[dict(letter=k,**v) for k,v in self.payload.items()]),
            self.options,self.sources,{})
        self.assertEqual(got,expected);self.assertFalse(got['A'][0]['first_party'])
        self.assertEqual(got['A'][0]['role'],'assistant')

    def test_missing_extra_wrong_shape_and_invented_fields_rejected(self):
        for kind in ('missing','extra','array','letter','role'):
            payload=copy.deepcopy(self.payload)
            if kind=='missing':payload.pop('B')
            elif kind=='extra':payload['C']=dict(witnesses=[])
            elif kind=='array':payload=dict(options=[])
            elif kind=='letter':payload['A']['letter']='A'
            elif kind=='role':payload['A']['witnesses'][0]['role']='user'
            with self.subTest(kind=kind),self.assertRaises(ValueError):decode_keyed(payload,self.options,self.sources,{})

    def test_wrong_source_or_rewritten_quote_never_accepted(self):
        for sid,quote in [('s1','The character enjoys hiking.'),('s0','I enjoy hiking.'),('unknown','I enjoy walking.')]:
            payload=copy.deepcopy(self.payload);payload['A']['witnesses']=[dict(source_id=sid,quote=quote)]
            with self.subTest(sid=sid,quote=quote),self.assertRaises(ValueError):decode_keyed(payload,self.options,self.sources,{})

    def test_valid_sibling_survives_bad_quote_without_relaxing_gate(self):
        payload=copy.deepcopy(self.payload);payload['B']['witnesses']=[dict(source_id='s1',quote='I enjoy hiking.')]
        trace={};got=decode_keyed(payload,self.options,self.sources,trace)
        self.assertEqual(set(got),{'A'});self.assertEqual(trace['rejected_witnesses'][0]['reason'],'quote_not_in_source')


if __name__=='__main__':unittest.main()
