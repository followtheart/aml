import copy,os,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import selftest_citation_recovery as recovery
gate=recovery.gate
class SupplementTests(unittest.IsolatedAsyncioTestCase):
 invoke=recovery.RecoveryTests.invoke
 def fixture(self):
  sources={sid:dict(id=sid,role='user',text=text) for sid,text in [('s0','I enjoy astronomy.'),('s1','I like reading about distant galaxies.')]}
  options=['A. Since you enjoy astronomy, visit an observatory.']
  ref=dict(source_id='s0',quote=sources['s0']['text'],basis='self_report',subject='current_user')
  entries=gate.validate_assessments({'options':[dict(letter='A',kind='personal',claims=[dict(text='you enjoy astronomy',status='unsupported',premise_type='interest',reason='none',citations=[ref])])]},options,sources)
  reply={'matches':[dict(claim_id='A:0',premise_type='interest',supported=True,citations=[ref,dict(source_id='s1',quote=sources['s1']['text'],basis='self_report',subject='current_user')])]}
  return entries,sources,options,reply
 async def test_preserve_deduplicate_and_verify(self):
  entries,sources,options,reply=self.fixture();before=copy.deepcopy(entries);out,mock=await self.invoke(entries,sources,reply)
  self.assertEqual(mock.await_count,1);self.assertEqual(entries,before);claim=out[0]['claims'][0];self.assertEqual([x['source_id'] for x in claim['citations']],['s0','s1']);self.assertEqual(claim['citations'][0],entries[0]['claims'][0]['citations'][0]);self.assertEqual(claim['status'],'unsupported')
  checks=gate.entailment_checks(out,sources,options);v={'checks':[dict(claim_id=c['claim_id'],entailed=False) for c in checks]};self.assertEqual(gate.validate_entailments(v,out,checks)[0]['status'],'unsupported')
 async def test_existing_citation_does_not_bypass_negative_gates(self):
  for mode in ['experience','contradicted','error','dropped','invalid']:
   with self.subTest(mode=mode):
    entries,sources,_,reply=self.fixture();c=entries[0]['claims'][0]
    if mode=='experience':c['premise_type']='experience'
    if mode=='contradicted':c['reason']='contradicted'
    if mode=='error':c['validation_errors']=['not_entailed']
    if mode=='dropped':c['dropped_citations']=[{}]
    if mode=='invalid':c['citations'][0]['valid']=False
    out,mock=await self.invoke(entries,sources,reply);self.assertEqual(mock.await_count,0);self.assertEqual(out,entries)
 async def test_bad_supplement_keeps_original(self):
  for mode in ['assistant','invented','third_party']:
   with self.subTest(mode=mode):
    entries,sources,_,reply=self.fixture();reply['matches'][0]['citations']=reply['matches'][0]['citations'][1:]
    if mode=='assistant':sources['s1']['role']='assistant'
    if mode=='invented':reply['matches'][0]['citations'][0]['quote']='invented quote'
    if mode=='third_party':reply['matches'][0]['citations'][0]['subject']='third_party'
    out,_=await self.invoke(entries,sources,reply);self.assertEqual(out,entries)
if __name__=='__main__':unittest.main()
