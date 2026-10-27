import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.joint_prompt_selection import choose, preserves


def row(mid,cost,path=()):
    text='text '+mid
    return dict(id=mid,cost=cost,content=text,_rank_text=text,_relation_path_ids=list(path),
                _packet_item=dict(sources=[dict(source_event_id=mid,role='user',content=text)]))


class BudgetTests(unittest.TestCase):
    def run_policy(self,rows,**kwargs):
        options=dict(rank_ids=[r['id'] for r in rows],protected_fine_ids=[],mandatory_main_ids=[],
            render=lambda rs:dict(prompt='x'*(10+sum(r['cost'] for r in rs)),reservation=10+sum(r['cost'] for r in rs)),
            graph=lambda rs:{},token_headroom=200,byte_limit=80,count_limit=3,fine_limit=3)
        options.update(kwargs)
        return choose(rows,[r['id'] for r in rows],rows,**options)

    def test_shorter_low_rank_items_do_not_displace_feasible_relevant_context(self):
        rows=[row('a',65),row('b',20),row('c',20)];before=copy.deepcopy(rows)
        p=self.run_policy(rows)
        self.assertEqual([c['id'] for c in p['submitted']],['a'])
        self.assertEqual(rows,before)

    def test_oversized_earlier_group_is_reconsidered_after_source_sharing(self):
        def render(rows):
            sizes={frozenset():10,frozenset(['a']):90,
                   frozenset(['b']):50,frozenset(['a','b']):60}
            size=sizes[frozenset(r['id'] for r in rows)]
            return dict(prompt='x'*size,reservation=size)
        p=self.run_policy([row('a',80),row('b',40)],render=render)
        self.assertEqual([d['id'] for d in p['decisions']],['b','a'])
        self.assertEqual([c['id'] for c in p['submitted']],['a','b'])

    def test_an_oversized_top_group_does_not_block_feasible_lower_groups(self):
        p=self.run_policy([row('a',100),row('b',20),row('c',20)])
        self.assertEqual([c['id'] for c in p['submitted']],['b','c'])

    def test_mandatory_main_retained(self):
        p=self.run_policy([row('a',65),row('b',20)],mandatory_main_ids=['a'])
        self.assertEqual([c['id'] for c in p['submitted']],['a'])

    def test_protected_fine_consumes_real_slot(self):
        p=self.run_policy([row('a',65),row('b',20),row('c',20)],protected_fine_ids=['a'],fine_limit=2)
        self.assertEqual({c['id'] for c in p['fine']},{'a','b'})

    def test_dependency_group_is_atomic(self):
        p=self.run_policy([row('a',15,['b']),row('b',15)],count_limit=1,fine_limit=1)
        self.assertNotIn('a',[c['id'] for c in p['submitted']])

    def test_missing_mandatory_dependency_falls_back(self):
        p=self.run_policy([row('a',10,['missing'])],mandatory_main_ids=['a'])
        self.assertEqual(p['reason'],'mandatory_missing_dependency')

    def test_token_budget_separate_from_prompt(self):
        p=self.run_policy([row('a',10)],mandatory_main_ids=['a'],token_headroom=15)
        self.assertEqual(p['reason'],'mandatory_request_budget')

    def test_shared_cost_is_measured_on_actual_combination(self):
        def render(rows):
            ids={r['id'] for r in rows}
            size={frozenset():10,frozenset(['a']):50,frozenset(['b']):90,frozenset(['a','b']):60}[frozenset(ids)]
            return dict(prompt='x'*size,reservation=size)
        p=self.run_policy([row('a',40),row('b',80)],mandatory_main_ids=['a'],render=render)
        self.assertEqual([c['id'] for c in p['submitted']],['a','b'])
        self.assertEqual(p['decisions'][0]['marginal_prompt_bytes'],10)

    def test_expired_selection_deadline_falls_back(self):
        p=self.run_policy([row('a',10)],deadline=0)
        self.assertEqual(p['reason'],'selection_deadline')

    def test_graph_values_and_order_preserved(self):
        self.assertTrue(preserves({'edges':[1,2],'scope':'recorded'},{'edges':[0,1,2,3],'scope':'recorded'}))
        self.assertFalse(preserves({'edges':[1,2]},{'edges':[2,1]}))
        self.assertFalse(preserves({'scope':'recorded'},{}))
        p=self.run_policy([row('a',20),row('b',10)],mandatory_main_ids=['a'],
            graph=lambda rs:dict(edges=['a'] if len(rs)==1 and rs[0]['id']=='a' else ['b'] if rs else []))
        self.assertEqual([c['id'] for c in p['submitted']],['a'])


if __name__=='__main__':unittest.main()
