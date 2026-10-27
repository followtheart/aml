"""Ordering must stay local even when selection is re-entered."""
import copy,unittest
from selftest_fine_selection import candidate,source
from app import fine_evidence
class LocalOrderTests(unittest.TestCase):
    def test_nested_baseline_selection_during_custom_order_uses_original_key(self):
        rows=[candidate('a',.9,[source('a','Alpha background.')]),candidate('b',.1,[source('b','Beta background.')])]
        for c in rows:c['_ce_score']=c['_fixture_score']
        before=copy.deepcopy(rows);original_key=fine_evidence.key;observed=[]
        def custom(c):
            baseline,_=fine_evidence.propose(rows,1,[],extra_limit=0)
            observed.append(baseline[0]['id'])
            return (-c['_ce_score'],0)
        chosen,_=fine_evidence.propose(rows,1,[],extra_limit=0,ordering_key=custom)
        self.assertEqual([c['id'] for c in chosen],['b'])
        self.assertTrue(observed);self.assertEqual(set(observed),{'a'})
        self.assertIs(fine_evidence.key,original_key);self.assertEqual(rows,before)
if __name__=='__main__':unittest.main()
