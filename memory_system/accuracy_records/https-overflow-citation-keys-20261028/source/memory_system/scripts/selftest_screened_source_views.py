"""Scope-aware novelty and whole-request budget preservation."""
import copy
import unittest
from screened_source_views import novel_views, schedule_bounds


def source(text, **changes):
    return dict(user_id='u',request_id='r',message_index=0,source_event_id='e',role='user',content=text,**changes)


class SourceViewTests(unittest.TestCase):
    def build(self, views, members, old, **limits):
        return novel_views(views,members,[{'_packet_item':{'sources':old}}],'q',
            request_size=lambda q,docs: sum(len(q.encode())+len(d.encode()) for d in docs),
            **dict(batch_limit=2,byte_limit=40,**limits))

    def test_only_fully_represented_same_scope_source_is_removed(self):
        views={'a':dict(text='one',source=source('one')), 'b':dict(text='one two',source=source('one two'))}
        members={'first':['a','b'],'second':['b']}; before=copy.deepcopy((views,members))
        result=self.build(views,members,[source('one')])
        self.assertEqual(list(result['views']),['b'])
        self.assertEqual(result['members'],{'first':['b'],'second':['b']})
        self.assertEqual(result['batches'],[['b']]);self.assertEqual((views,members),before)

    def test_different_role_user_or_event_cannot_suppress_view(self):
        view={'a':dict(text='one',source=source('one'))}
        for changes in ({'role':'assistant'},{'user_id':'other'},{'source_event_id':'other'}):
            old=dict(source('one'),**changes)
            self.assertEqual(list(self.build(view,{'m':['a']},[old])['views']),['a'])

    def test_batch_limits_use_complete_utf8_query_document_cost(self):
        views={str(i):dict(text='文'*7,source=dict(source(str(i)),source_event_id=str(i))) for i in range(3)}
        result=self.build(views,{'m':list(views)},[])
        self.assertEqual(result['batches'],[['0'],['1'],['2']])
        views['0']['text']='文'*20
        with self.assertRaises(ValueError):self.build(views,{'m':list(views)},[])

    def test_original_main_token_call_and_time_reservations(self):
        base=dict(main_reservation=60,token_headroom=100,call_headroom=3,
                  deadline_seconds=100,main_seconds=45,ce_seconds=45)
        self.assertTrue(schedule_bounds([20,20],**base)['eligible'])
        self.assertEqual(schedule_bounds([21,20],**base)['reason'],'token_headroom')
        self.assertEqual(schedule_bounds([10,10,10],**base)['reason'],'call_headroom')
        self.assertEqual(schedule_bounds([10],**dict(base,deadline_seconds=40))['reason'],'deadline_headroom')
        self.assertEqual(schedule_bounds([],**base)['reason'],'no_source_views')


if __name__=='__main__':unittest.main()
