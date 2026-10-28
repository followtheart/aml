"""Offline integration checks for required option objects and scoped repair."""
import copy,os,sys,unittest
from pathlib import Path
from unittest.mock import patch,AsyncMock
os.environ['AML_FAKE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import config
from app import answer_choice as ac

class ObjectSupportTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self,replies,options=None):
        options=options or ['A. Take a walk.','B. Read a book.']
        diag={}
        with patch.object(ac.llm,'complete_json',AsyncMock(side_effect=replies)) as calls:
            result=await ac.assess_support('Options:\n'+'\n'.join(options),ac.Assessments.model_json_schema(),options,{},diag)
        return result,diag,calls
    async def test_complete_map_uses_one_call_and_preserves_raw_log(self):
        reply={'options':{'A':{'kind':'generic','claims':[]},'B':{'kind':'generic','claims':[]}}}
        result,diag,calls=await self.run_case([reply])
        self.assertEqual([x['letter'] for x in result],['A','B'])
        self.assertEqual(calls.call_count,1)
        schema=calls.call_args.kwargs['schema']['properties']['options']
        self.assertEqual(schema['required'],['A','B'])
        self.assertFalse(schema['additionalProperties'])
        self.assertIsInstance(diag['answer_calls'][0]['response']['options'],dict)
    async def test_missing_sibling_repairs_only_that_option(self):
        result,diag,calls=await self.run_case([
            {'options':{'A':{'kind':'generic','claims':[]}}},
            {'options':[{'letter':'B','kind':'generic','claims':[]}]}])
        self.assertEqual(calls.call_count,2)
        self.assertEqual(diag['answer_support_validation']['unresolved_options'],[])
        self.assertEqual(calls.call_args.kwargs['schema']['$defs']['Assessment']['properties']['letter']['enum'],['B'])
        self.assertEqual(result[0]['status'],'generic')
    async def test_embedded_letter_cannot_move_evidence_to_another_option(self):
        result,diag,calls=await self.run_case([
            {'options':{'A':{'letter':'B','kind':'generic','claims':[]},'B':{'kind':'generic','claims':[]}}},
            {'options':[{'letter':'A','kind':'generic','claims':[]}]}])
        self.assertEqual(calls.call_count,2)
        self.assertEqual(calls.call_args.kwargs['schema']['$defs']['Assessment']['properties']['letter']['enum'],['A'])
        self.assertEqual([x['letter'] for x in result],['A','B'])
    async def test_malformed_value_does_not_discard_valid_sibling(self):
        result,diag,calls=await self.run_case([
            {'options':{'A':42,'B':{'kind':'generic','claims':[]}}},
            {'options':[{'letter':'A','kind':'generic','claims':[]}]}])
        self.assertEqual(calls.call_count,2)
        self.assertEqual(diag['answer_support_validation']['unresolved_options'],[])
        self.assertEqual(result[1]['status'],'generic')
    async def test_legacy_array_remains_source_checked(self):
        result,diag,calls=await self.run_case([{'options':[
            {'letter':'A','kind':'generic','claims':[]},
            {'letter':'B','kind':'generic','claims':[]}]}])
        self.assertEqual(calls.call_count,1)
        self.assertEqual(diag['answer_support_validation']['unresolved_options'],[])
    def test_option_requests_do_not_leak_scope_or_mutate_claim_schema(self):
        schema=ac.Assessments.model_json_schema();original=copy.deepcopy(schema)
        _, first=ac.support_object_request('prompt',schema,['A','B'])
        _, second=ac.support_object_request('prompt',schema,['C'])
        self.assertEqual(first['properties']['options']['required'],['A','B'])
        self.assertEqual(second['properties']['options']['required'],['C'])
        self.assertNotIn('letter', first['$defs']['Assessment']['properties'])
        self.assertEqual(first['$defs']['Claim'], original['$defs']['Claim'])
        self.assertEqual(schema,original)
if __name__=='__main__':unittest.main()
