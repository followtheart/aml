"""Single-pass packets stay immutable and only returned memories strengthen."""
import asyncio
import sys
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context, config, schemas, search_pipeline, store


class PacketTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_sees_final_packet_and_only_used_memory_strengthens(self):
        st = store.Store(':memory:')
        try:
            for i in range(3):
                st.insert_amu(user_id='u', session_id='s', content=f'fact {i}')
            rows = st.get_amus('u')
            with patch.object(search_pipeline, '_understand', AsyncMock(return_value={'intent': 'fact'})), patch.object(
                    search_pipeline, '_recall', AsyncMock(return_value=[rows])), patch.object(
                    search_pipeline, '_filter_rerank', AsyncMock(return_value=rows)), patch.object(
                    search_pipeline.llm, 'complete_json', AsyncMock(side_effect=AssertionError('extra model call'))):
                result = await search_pipeline.run_search(st, schemas.SearchRequest(user_id='u', query='q', top_k=1))
            self.assertEqual(result.evidence_status, 'retrieved')
            self.assertEqual(result.verification_status, 'not_run')
            self.assertEqual(answer_context.build([x.model_dump() for x in result.data]), result.data[0].content)
            # §4.5: only the memory in the final packet records one actual use;
            # candidates that were merely recalled never gain strength.
            after = {r['id']: r for r in st.get_amus('u')}
            returned = result.data[0].id
            self.assertGreater(after[returned]['strength'], 1.0)
            self.assertEqual(after[returned]['recall_count'], 1)
            self.assertTrue(all(r['strength'] == 1.0 and r['recall_count'] == 0
                                for rid, r in after.items() if rid != returned))
        finally:
            st.conn.close()



if __name__ == '__main__':
    unittest.main()
