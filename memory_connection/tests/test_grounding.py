import copy
import unittest
from memory_connection.app.temporal import source_time
from memory_connection.app.ranking import cover_query_anchors, query_anchors
from memory_connection.app.answering import reading_packet, answer, date_arithmetic
from memory_connection.tests.test_reasoning import source, candidates


class CalendarTests(unittest.TestCase):
    def test_approximate_year_is_anchored_to_source_without_false_day(self):
        s=source('I watched it around three years ago.','2022-01-21T12:00:00+00:00')
        n=source_time(s)['relative_dates'][0]
        self.assertEqual((n['resolved'],n['from'],n['to']),('2019','2019-01-01','2019-12-31'))
        self.assertTrue(n['approximate']);self.assertEqual(n['precision'],'year')
        self.assertEqual(s['content'][n['start']:n['end']],n['phrase'])

    def test_month_rollover_and_leap_year(self):
        n=source_time(source('One month ago.','2024-03-31T00:00:00+00:00'))['relative_dates'][0]
        self.assertEqual((n['from'],n['to']),('2024-02-01','2024-02-29'))
        n=source_time(source('Next month.','2023-12-31T00:00:00+00:00'))['relative_dates'][0]
        self.assertEqual(n['resolved'],'2024-01')

    def test_missing_anchor_and_duration_do_not_become_event_dates(self):
        self.assertEqual(source_time({'content':'three years ago','timestamp':None})['relative_dates'],[])
        self.assertEqual(source_time(source('It lasted three years.'))['relative_dates'],[])
        self.assertEqual(source_time(source('twenty one years ago'))['relative_dates'],[])
        self.assertEqual(source_time(source('two or three years ago'))['relative_dates'],[])
        self.assertEqual(source_time(source('3.5 years ago'))['relative_dates'],[])
        self.assertEqual(source_time(source('1,000 years ago'))['relative_dates'],[])
        self.assertEqual(source_time(source('-3 years ago'))['relative_dates'],[])

    def test_lossless_long_message_adapter_and_explicit_clock(self):
        from memory_connection.scripts.longmemeval_data import split_original,instant
        text=('A paragraph with unicode 中文.\n'*300)+'  '
        self.assertEqual(''.join(split_original(text)),text)
        self.assertTrue(all(s.strip() for s in split_original(text)))
        self.assertEqual(instant('2023/05/20 (Sat) 02:21').isoformat(),'2023-05-20T02:21:00+00:00')


class CoverageTests(unittest.TestCase):
    def test_source_novelty_alone_is_not_enough(self):
        cs=candidates();ordered,ids=cover_query_anchors(cs,'What happened?',16)
        self.assertEqual(ids,[]);self.assertEqual([c['event'] for c in ordered],[c['event'] for c in cs])

    def test_missing_named_premise_gets_slot_with_semantic_support(self):
        cs=candidates();cs[0]['event']['sources'][0]['content']='Toby was adopted in July.'
        cs[20]['event']['sources'][0]['content']='Buddy was adopted in October.'
        ordered,ids=cover_query_anchors(cs,'When were Toby and Buddy adopted?',16)
        self.assertEqual(ids,[cs[20]['event']['event_id']]);self.assertEqual(len(ordered[:16]),16)
        self.assertEqual(len({c['event']['event_id'] for c in ordered}),len(cs))
        self.assertEqual(ordered[0]['event'],cs[0]['event'])

    def test_covered_or_semantically_weak_entity_is_not_forced(self):
        cs=candidates();cs[0]['event']['sources'][0]['content']='Toby and Buddy.'
        cs[20]['event']['sources'][0]['content']='Buddy.'
        self.assertEqual(cover_query_anchors(cs,'Toby and Buddy?',16)[1],[])
        cs[0]['event']['sources'][0]['content']='Toby.';cs[20]['semantic_score']=-10
        self.assertEqual(cover_query_anchors(cs,'Toby and Buddy?',16)[1],[])


class ReaderTests(unittest.IsolatedAsyncioTestCase):
    def items(self):
        e=candidates(1)[0]['event'];e['sources'][0].update(source('Around three years ago.','2022-01-21T00:00:00+00:00'))
        return [{'content':e['text'],'sources':e['sources'],'evidence_spans':[]}]

    async def test_reader_preserves_originals_and_source_local_dates(self):
        items=self.items();original=copy.deepcopy(items);packet=reading_packet(items*2)
        self.assertEqual(len(packet),1);self.assertEqual(packet[0]['relative_dates'][0]['resolved'],'2019')
        self.assertEqual(items,original)

    async def test_source_date_arithmetic_is_labeled_and_budgeted(self):
        ss=[{'ref':'s0','stated_at':'2023-07-11T00:00:00+00:00','verified_quotes':['adopted']},
            {'ref':'s1','stated_at':'2023-10-19T00:00:00+00:00','verified_quotes':['adopted']}]
        result=date_arithmetic('How many months between?',ss,24000)
        self.assertEqual(result[0]['calendar_month_difference'],3)
        self.assertEqual(result[0]['source_refs'],['s0','s1'])
        self.assertIn('statement dates only',result[0]['scope'])
        self.assertEqual(date_arithmetic('How many months?',ss,10),[])

    async def test_unknown_answer_reference_is_rejected(self):
        class Fake:
            async def complete(self,*args):return {'answer':'2019','source_refs':['invented'],'missing_premises':[]}
        with self.assertRaises(ValueError):await answer(Fake(),'When?',self.items())

    async def test_no_evidence_abstains_without_model_call(self):
        result=await answer(object(),'When?',[])
        self.assertEqual(result['answer'],'Unknown')

    async def test_reference_validation_does_not_claim_semantic_verification(self):
        class Fake:
            async def complete(self,*args):return {'answer':'2019','source_refs':['s0'],'missing_premises':[]}
        result=await answer(Fake(),'When?',self.items())
        self.assertEqual(result['reference_status'],'references_valid')

    async def test_purge_during_answer_invalidates_all_returned_evidence(self):
        from memory_connection.app.service import MemoryService
        from memory_connection.app.schemas import SearchRequest
        class Store:
            revision=(1,0)
            def state(self,user):return self.revision
        store=Store()
        class Fake:
            async def complete(self,*args):
                store.revision=(2,1)
                return {'answer':'2019','source_refs':['s0'],'missing_premises':[]}
        service=MemoryService(store,Fake())
        async def search(req):return {'data':self.items(),'read_revision':1,'read_epoch':0,'verification_status':'complete'}
        service.search=search
        result=await service.answer(SearchRequest(user_id='u',query='When?'))
        self.assertEqual(result['data'],[]);self.assertEqual(result['answer']['reference_status'],'invalidated')
