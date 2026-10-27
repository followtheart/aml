"""Rejected proposals stay diagnostic; accepted storage decisions stay independent."""
import asyncio,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import AsyncMock,patch
os.environ['AML_FAKE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import add_pipeline as add,config,memory_debug as debug,schemas,store

class ExtractionTraceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'debug.jsonl'
        self.setting=patch.object(config,'MEMORY_DEBUG_LOG',str(self.path));self.setting.start()
        self.st=store.Store(':memory:')
        self.req=schemas.AddRequest(request_id='trace-1',user_id='u',session_id='s',messages=[schemas.Message(role='user',content='I tried surfing with my family.')])
    def tearDown(self):
        self.st.conn.close();self.setting.stop();self.tmp.cleanup()
    def capture(self):return debug.capture(self.st,self.req)['extraction_trace']

    async def test_rejected_proposal_and_fallback_remain_distinct(self):
        raw={'facts':[{'content':'I own a yacht.','type':'fact','evidence':[{'message_index':0,'quote':'I own a yacht.'}]}]}
        with debug.extraction_scope(),patch.object(add.llm,'complete_json',AsyncMock(return_value=raw)):
            result=await add._extract_segment(self.req,None,0,[0]);trace=self.capture()
        events={e['stage']:e for e in trace[0]['events']}
        self.assertEqual(events['extract_response']['response'],raw)
        self.assertIn('grounding_rejected',events)
        self.assertEqual(events['segment_output']['facts'],result)
        self.assertEqual(result[0]['type'],'episode');self.assertNotIn('yacht',result[0]['content'])
        self.assertEqual(self.capture(),[])

    async def test_semantic_rejection_records_exact_request_and_verdict(self):
        raw={'facts':[{'content':'I tried surfing with my family.','type':'fact','evidence':[{'message_index':0,'quote':'I tried surfing with my family.'}]}]}
        responses=[raw,{'valid':False,'reason':'Synthetic rejection for trace boundary test'}]
        with debug.extraction_scope(),patch.object(config,'FAKE',False),patch.object(add.llm,'complete_json',AsyncMock(side_effect=responses)) as call:
            result=await add._extract_segment(self.req,None,0,[0]);trace=self.capture()
        events={e['stage']:e for e in trace[0]['events']}
        self.assertEqual(events['semantic_request']['prompt'],call.await_args_list[1].args[0])
        self.assertEqual(events['semantic_response']['response'],responses[1])
        self.assertEqual([f['type'] for f in result],['episode'])

    async def test_parallel_segments_and_requests_are_isolated(self):
        async def request(label):
            with debug.extraction_scope():
                async def segment(index):
                    with debug.segment_scope(index,[index]):
                        await asyncio.sleep(0);debug.extraction_event('test',label=label)
                await asyncio.gather(segment(2),segment(0));return self.capture()
        first,second=await asyncio.gather(request('first'),request('second'))
        for rows,label in [(first,'first'),(second,'second')]:
            self.assertEqual([r['segment_index'] for r in rows],[0,2])
            self.assertTrue(all(r['events']==[dict(stage='test',label=label)] for r in rows))
        self.assertEqual(self.capture(),[])

    def test_trace_snapshots_mutable_proposals(self):
        payload={'facts':[{'content':'original'}]}
        with debug.extraction_scope(),debug.segment_scope(0,[0]):
            debug.extraction_event('extract_response',response=payload);payload['facts'][0]['content']='changed'
            self.assertEqual(self.capture()[0]['events'][0]['response']['facts'][0]['content'],'original')

    def test_disabled_does_not_collect(self):
        with patch.object(config,'MEMORY_DEBUG_LOG',''),debug.extraction_scope(),debug.segment_scope(0,[0]):
            debug.extraction_event('extract_response',response={'secret':'source'});self.assertEqual(self.capture(),[])

    async def test_publish_contains_trace_once_and_rollback_contains_none(self):
        await add.run_add(self.st,self.req);await add.run_add(self.st,self.req)
        records=[json.loads(x) for x in self.path.read_text().splitlines()]
        self.assertEqual(len(records),1);self.assertTrue(records[0]['extraction_trace']);self.assertEqual(records[0]['schema_version'],4)
        self.req.request_id='trace-2'
        with patch.object(add,'_update_summary',AsyncMock(side_effect=RuntimeError('rollback'))):
            with self.assertRaises(RuntimeError):await add.run_add(self.st,self.req)
        self.assertEqual(len(self.path.read_text().splitlines()),1);self.assertEqual(self.capture(),[])

    def test_diagnostic_copy_failure_does_not_escape(self):
        class Broken:
            def __deepcopy__(self,memo):raise ValueError('copy error')
        with debug.extraction_scope(),debug.segment_scope(0,[0]):
            with self.assertLogs('aml.memory_debug',level='WARNING'):debug.extraction_event('extract_response',response=Broken())
            self.assertEqual(self.capture()[0]['events'],[])

if __name__=='__main__':unittest.main()
