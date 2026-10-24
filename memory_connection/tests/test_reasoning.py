import copy
import json
import unittest
import time
from datetime import datetime
from unittest.mock import patch
from memory_connection.app.temporal import source_time, source_annotations
from memory_connection.app.references import reference_packet
from memory_connection.app.ranking import reserve_exclusive, source_keys, single_side_sources
from memory_connection.app.config import Settings
from memory_connection.app.event_builder import build_events, valid_spans
from memory_connection.app.schemas import AddRequest, Span, View
from memory_connection.scripts.benchmark_metrics import common_packet


def source(text, date="2023-10-22T01:55:00+00:00"):
    return {"content":text,"timestamp":int(datetime.fromisoformat(date).timestamp()*1000)}


def candidates(count=40):
    es=build_events(AddRequest(request_id="r",user_id="u",session_id="s",messages=[
        {"role":"user","content":f"A: original source {i}","message_id":str(i)} for i in range(count)]))
    for e in es:e["sources"]=[s for s in e["sources"] if s["source_id"]==e["primary_source_id"]]
    return [{"event":e,"score":1/(1+i),"semantic_score":1/(1+i),"paths":[{"a_view":0,"b_view":0,"rank":1}]} for i,e in enumerate(es)]


class TemporalTests(unittest.TestCase):
    def test_relative_weekday_uses_source_date_not_query_date(self):
        s=source("Passed last Friday.")
        result=source_time(s)
        self.assertEqual(result["relative_dates"][0]["from"],"2023-10-20")
        self.assertEqual(result["stated_at"],"2023-10-22T01:55:00+00:00")
        self.assertEqual(s["content"],"Passed last Friday.")

    def test_weekend_and_year_boundary(self):
        self.assertEqual(source_time(source("Joined last weekend.","2023-07-17T06:31:00+00:00"))["relative_dates"][0]["from"],"2023-07-15")
        self.assertEqual(source_time(source("Yesterday.","2024-01-01T00:00:00+00:00"))["relative_dates"][0]["from"],"2023-12-31")

    def test_missing_or_ambiguous_date_does_not_invent_precision(self):
        self.assertEqual(source_time({"content":"Yesterday.","timestamp":None})["relative_dates"],[])
        note=source_time(source("Last weekend.","2023-10-22T01:55:00+00:00"))["relative_dates"][0]
        self.assertTrue(note["ambiguous"])
        self.assertNotIn("from",note)

    def test_exact_source_offsets_are_unchanged_by_annotations(self):
        es=[c["event"] for c in candidates(1)]
        es[0]["sources"][0].update(source("A: I did NOT pass last Friday."))
        original=copy.deepcopy(es)
        packet,_,refs=reference_packet(es)
        self.assertTrue(all(valid_spans([Span(**r)],es[0]) for r in refs.values()))
        self.assertEqual(es,original)
        self.assertEqual(packet["events"][0]["sources"][0]["relative_dates"][0]["from"],"2023-10-20")

    def test_answer_adapter_keeps_annotations_without_rewriting_sources(self):
        e=candidates(1)[0]["event"]
        item={"content":e["text"],"sources":e["sources"],"source_annotations":source_annotations(e["sources"])}
        p=common_packet([item],24000)
        self.assertEqual(p,[item])


class ExclusiveTests(unittest.TestCase):
    def test_only_absent_single_side_sources_receive_a_slot(self):
        cs=candidates();known=set().union(*(source_keys(c) for i,c in enumerate(cs) if i!=18))
        ordered,ids=reserve_exclusive(cs,known,16)
        self.assertEqual(ids,[cs[18]['event']['event_id']])
        self.assertEqual([c['event'] for c in ordered[:15]],[c['event'] for c in cs[:15]])
        self.assertEqual(ordered[15]['event'],cs[18]['event'])
        self.assertEqual(len({c['event']['event_id'] for c in ordered}),40)

    def test_path_exclusivity_alone_cannot_displace_evidence(self):
        cs=candidates();known=set().union(*(source_keys(c) for c in cs))
        cs[18]['paths']=[{'a_view':2,'b_view':1,'rank':1}]
        self.assertEqual(reserve_exclusive(cs,known,16)[1],[])

    def test_source_exposed_by_an_adjacent_event_is_not_exclusive(self):
        cs=candidates();cs[18]['event']['sources']=copy.deepcopy(cs[0]['event']['sources'])
        known=set().union(*(source_keys(c) for i,c in enumerate(cs) if i!=18))
        self.assertEqual(reserve_exclusive(cs,known,16)[1],[])

    def test_existing_exclusive_last_slot_is_not_evicted(self):
        cs=candidates();known=set().union(*(source_keys(c) for i,c in enumerate(cs) if i not in {15,18}))
        self.assertEqual(reserve_exclusive(cs,known,16)[1],[])

    def test_duplicate_exclusive_context_does_not_block_a_new_source(self):
        cs=candidates();cs[15]['event']['sources']=copy.deepcopy(cs[0]['event']['sources'])
        known=set().union(*(source_keys(c) for i,c in enumerate(cs) if i not in {0,15,18}))
        self.assertEqual(reserve_exclusive(cs,known,16)[1],[cs[18]['event']['event_id']])


class CounterfactualTests(unittest.IsolatedAsyncioTestCase):
    async def test_reuses_vectors_and_disables_each_side_separately(self):
        cs=candidates(3);seen=[]
        def fake(events,views,vectors,cfg,deadline):
            seen.append((len(views),cfg.memory_expansion,vectors))
            return [cs[0] if len(views)==1 else cs[1]],{'view_comparisons':7}
        vectors=[[1.,0.],[0.,1.]]
        with patch('memory_connection.app.ranking.recall',side_effect=fake):
            keys,compared=await single_side_sources([c['event'] for c in cs],
                [View(kind='original',text='need'),View(kind='paraphrase',text='bridge')],
                vectors,Settings(),time.monotonic()+5)
        self.assertEqual(keys,source_keys(cs[0])|source_keys(cs[1]))
        self.assertEqual(compared,14)
        self.assertIn((1,True,vectors[:1]),seen)
        self.assertIn((2,False,vectors),seen)
