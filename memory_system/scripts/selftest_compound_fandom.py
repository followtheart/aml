import copy,importlib.util,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from selftest_answer_choice import memory,seal
from app import answer_choice as gate
class FandomTests(unittest.TestCase):
 def test_negative_and_unrelated_claims_unchanged(self):
  for text in ['Since you are not a football fan and enjoy cooking,','Since you are no longer a football fan and enjoy cooking,','Since you are a teacher and enjoy cooking,','Since you are into rock and roll,']:
   with self.subTest(text=text):self.assertIsNone(gate._COMPOUND_INTEREST.fullmatch(text))
 def test_citations_and_spans_do_not_transfer_to_enjoyment(self):
  for text in ['Since you’re a football fan and enjoy sports stories,','Since you are an opera fan and prefer historical fiction,','Since you are a fan and like comedy,','Since you are into anime and enjoy political stories,']:
   with self.subTest(text=text):
    sources,_=gate.build_catalog(seal(memory('I am a football fan.')));sid=next(iter(sources));option='A. '+text+' Try a series.'
    payload={'options':[{'letter':'A','kind':'personal','claims':[{'text':text,'status':'supported','premise_type':'interest','citations':[{'source_id':sid,'quote':'I am a football fan.','basis':'self_report','subject':'current_user'}]}]}]}
    entries=gate.validate_assessments(payload,[option],sources);before=copy.deepcopy(entries);d={};result=gate.split_compound_interest_claims(entries,sources,d)
    self.assertEqual(entries,before);a,b=result[0]['claims'];self.assertEqual(len(a['citations']),1);self.assertEqual(b['citations'],[]);self.assertEqual(b['status'],'unsupported')
    for c in [a,b]:self.assertEqual(option[c['option_span']['start']:c['option_span']['end']],c['text'])
    self.assertEqual(result[0]['primary_claim'],0)
 def test_invalid_quote_not_resurrected(self):
  text='Since you are a football fan and enjoy sports stories,';sources,_=gate.build_catalog(seal(memory('I enjoy cooking.')));sid=next(iter(sources))
  payload={'options':[{'letter':'A','kind':'personal','claims':[{'text':text,'status':'supported','premise_type':'interest','citations':[{'source_id':sid,'quote':'I am a football fan.','basis':'self_report','subject':'current_user'}]}]}]}
  entries=gate.validate_assessments(payload,['A. '+text+' Try TV.'],sources);result=gate.split_compound_interest_claims(entries,sources,{})
  self.assertTrue(all(c['status']=='unsupported' and not any(x.get('anchor') for x in c['citations']) for c in result[0]['claims']))
if __name__=='__main__':unittest.main()
