import copy
import unittest
from user_source_order import views,order


def candidate(mid,score,text='I enjoy visiting science museums.',**extra):
    s=dict(user_id='u',request_id='r',message_index=1,source_event_id='event_'+mid,role='user',content=text)
    s.update(extra)
    return dict(id=mid,content=text,_rank_text=text,_ce_score=score,_fused=0,
                _packet_item=dict(content=text,sources=[s]))


class Ordering(unittest.TestCase):
    def test_only_movable_slots_change(self):
        rows=[candidate('a',.9),candidate('pin',.8),candidate('b',.7),candidate('assist',.6,role='assistant')]
        before=copy.deepcopy(rows);data=views(rows,'u',protected_ids=['pin'],document_limit=1000)
        scores={data['members']['a'][0]:.1,data['members']['b'][0]:.9}
        self.assertEqual(order(rows,data,scores,protected_ids=['pin'])['ordered_ids'],['b','pin','a','assist'])
        self.assertEqual(rows,before)

    def test_ties_keep_original_order(self):
        rows=[candidate('a',.9),candidate('b',.7)]
        data=views(rows,'u',document_limit=1000)
        self.assertEqual(order(rows,data,{k:.4 for k in data['views']})['ordered_ids'],['a','b'])

    def test_partial_or_invalid_scores_fall_back(self):
        rows=[candidate('a',.9),candidate('b',.7)];data=views(rows,'u',document_limit=1000)
        for scores in ({},{k:float('nan') for k in data['views']},{k:True for k in data['views']}):
            result=order(rows,data,scores);self.assertEqual(result['status'],'fallback');self.assertEqual(result['ordered_ids'],['a','b'])

    def test_partial_hidden_and_foreign_sources_not_scored(self):
        variants=[dict(role='assistant'),dict(user_id='other'),dict(content_excerpted=True),
                  dict(content_omitted='budget'),dict(content_span=dict(start=1,end=4,original_length=10))]
        for extra in variants:self.assertFalse(views([candidate('a',.9,**extra)],'u',document_limit=1000)['views'])
        c=candidate('a',.9);c['_rank_text']='Some other text.'
        self.assertFalse(views([c],'u',document_limit=1000)['views'])

    def test_no_truncation(self):
        c=candidate('a',.9,'完整原文'*100)
        result=views([c],'u',document_limit=100)
        self.assertFalse(result['views']);self.assertEqual(result['omissions'][0]['reason'],'whole_user_source_document_budget')

    def test_same_observation_deduplicated(self):
        a=candidate('a',.9);b=copy.deepcopy(a);b.update(id='b',_ce_score=.8)
        result=views([a,b],'u',document_limit=1000)
        self.assertEqual(len(result['views']),1);self.assertEqual(result['members']['a'],result['members']['b'])

    def test_failed_original_ce_stays_unscored(self):
        c=candidate('a',None)
        self.assertFalse(views([c],'u',document_limit=1000)['views'])


if __name__=='__main__':unittest.main()
