import asyncio
import copy
import json
import time
import unittest
import httpx
from memory_connection.app.config import Settings
from memory_connection.app.references import reference_packet, decode_references
from memory_connection.app.event_builder import build_events, valid_spans
from memory_connection.app.schemas import AddRequest, Span
from memory_connection.app.ranking import diverse_order, rank_candidates
from memory_connection.app.verification import verify_candidates
from memory_connection.app.providers import HTTPProvider
from memory_connection.tests.test_pipeline import FixtureProvider


def events():
    return build_events(AddRequest(request_id="r", user_id="u", session_id="s", messages=[
        {"role": "user", "content": "A: I might buy a car. I have NOT bought it."},
        {"role": "assistant", "content": "B: I plan to open a garage."}]))


class ReferenceTests(unittest.TestCase):
    def test_original_refs_preserve_negation_and_attribution(self):
        original = events()
        packet, aliases, refs = reference_packet(original)
        self.assertEqual(len(packet["events"][1]["sources"]), 2)
        self.assertEqual(packet["events"][1]["source_ids"], ["s0", "s1"])
        self.assertIn("I have NOT bought it.", [r["quote"] for r in refs.values()])
        self.assertTrue(all(valid_spans([Span(**ref)], original[1]) for ref in refs.values()))
        self.assertNotIn("views", json.dumps(packet))

    def test_unknown_ref_fails_one_item_only(self):
        original = events();_, aliases, refs = reference_packet(original)
        raw = {"decisions": [{"event_id": "e0", "verdict": "partial", "evidence_refs": ["r0"]},
                             {"event_id": "e1", "verdict": "relevant", "evidence_refs": ["invented"]}]}
        decoded = decode_references(raw, aliases, refs)["decisions"]
        self.assertEqual(decoded[0]["verdict"], "partial")
        self.assertEqual(decoded[1], {"event_id": original[1]["event_id"]})

    def test_repeated_quotes_have_unambiguous_server_offsets(self):
        original = events()[:1];original[0]["sources"][0]["content"] = "Repeat. Repeat."
        _,_,refs = reference_packet(original)
        self.assertEqual([s["start"] for s in refs.values()], [0,8])
        self.assertTrue(all(valid_spans([Span(**s)],original[0]) for s in refs.values()))

    def test_source_diversity_promotes_new_evidence_not_adjacent_duplicates(self):
        e = events()[0]
        candidates = [{"event": copy.deepcopy(e), "score": 1, "paths": []} for _ in range(5)]
        candidates[-1]["event"]["sources"][0]["message_id"] = "new-evidence"
        ordered = diverse_order(candidates, [.99,.98,.97,.96,.95])
        self.assertEqual(ordered[1]["coarse_rank"], 5)
        self.assertEqual(len(ordered), 5)

    def test_semantic_ranking_recovers_low_coarse_rank(self):
        base = events()[0]
        candidates = []
        for i in range(30):
            e = copy.deepcopy(base);e["sources"][0]["message_id"] = str(i)
            candidates.append({"event":e,"score":1,"paths":[]})
        scores = [.1]*30;scores[29]=.99
        self.assertEqual(diverse_order(candidates,scores)[0]["coarse_rank"],30)


class AsyncOptimizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_premise_is_not_starved_during_final_packing(self):
        from memory_connection.app.packing import pack
        original=events();provider=FixtureProvider();raw=await provider.verify({},original)
        raw["decisions"][0]["verdict"]="partial"
        decisions={d["event_id"]:d for d in raw["decisions"]}
        data,_,_=pack([{"event":e,"score":1-i*.2} for i,e in enumerate(original)],decisions,1,24000)
        self.assertEqual(data[0]["id"],original[0]["event_id"])

    async def test_paired_query_replay_is_identical_and_rejects_drift(self):
        from memory_connection.scripts.optimization_benchmark import PreparedQueryProvider
        prepared={"query":{"query":"unknown need"},"expansion":{"views":[{"text":"bridge"}]},"vectors":{"unknown need":[1.,0.],"bridge":[0.,1.]}}
        replay=PreparedQueryProvider(object(),prepared)
        first=await replay.expand_query(prepared["query"]);first["views"].clear()
        self.assertEqual(len((await replay.expand_query(prepared["query"]))["views"]),1)
        self.assertEqual(await replay.embed(["bridge","unknown need"]),[[0.,1.],[1.,0.]])
        with self.assertRaises(ValueError):await replay.expand_query({"query":"different need"})

    async def test_holdout_excludes_all_previous_conversations(self):
        from memory_connection.scripts.benchmark_metrics import prepare
        plan=prepare("memory_system/data/locomo_eval.json","memory_system/data/locomo10.json",3,12,20261024,exclude=["conv-43","conv-50"])
        self.assertEqual(len(plan),3)
        self.assertTrue(all(c["conversation_id"] not in {"conv-43","conv-50"} for c in plan))
        self.assertEqual(sum(len(c["qa"]) for c in plan),36)

    async def test_one_malformed_decision_does_not_discard_good_neighbor(self):
        original = events();provider = FixtureProvider()
        normal = provider.verify
        async def verify(query,batch):
            raw = await normal(query,batch);raw["decisions"][1]["verdict"]="invalid";return raw
        provider.verify=verify
        decisions,errors,_ = await verify_candidates(provider,{},[{"event":e} for e in original],Settings(),time.monotonic()+5)
        self.assertEqual(list(decisions),[original[0]["event_id"]])
        self.assertEqual(errors[0]["reason"],"invalid_decision")

    async def test_cross_event_reference_cannot_pass_source_ownership(self):
        original=events();_,aliases,refs=reference_packet(original)
        raw=decode_references({"decisions":[{"event_id":"e0","verdict":"relevant","evidence_refs":["r2"]},
                                           {"event_id":"e1","verdict":"irrelevant","evidence_refs":[]}]},aliases,refs)
        class Provider:
            async def verify(self,*args):return raw
        decisions,errors,_=await verify_candidates(Provider(),{},[{"event":e} for e in original],Settings(),time.monotonic()+5)
        self.assertNotIn(original[0]["event_id"],decisions)
        self.assertEqual(errors[0]["reason"],"invalid_evidence")

    async def test_partial_rerank_failure_never_misaligns_scores(self):
        class Provider:
            calls=0
            async def rerank(self,*args):
                self.calls+=1
                if self.calls==2:raise RuntimeError("failure")
                return [.9]
        candidates=[{"event":e,"score":1,"paths":[]} for e in events()]
        ordered,metrics,errors=await rank_candidates(Provider(),{},candidates,Settings(rerank_model="test",rerank_batch_size=1),time.monotonic()+5)
        self.assertEqual(len(ordered),2)
        self.assertTrue(all(c["semantic_score"] is None for c in ordered))
        self.assertEqual(errors[0]["stage"],"rerank")

    async def test_reranker_wire_and_actual_usage(self):
        records=[]
        def handle(request):
            body=json.loads(request.content)
            self.assertEqual(body["input"]["documents"],["original one","original two"])
            return httpx.Response(200,json={"output":{"results":[{"index":1,"relevance_score":.8},{"index":0,"relevance_score":.2}]},"usage":{"input_tokens":17}})
        provider=HTTPProvider(Settings(model="m",embed_model="e",provider_api_key="test",rerank_model="r",rerank_api_key="test",rerank_api_url="https://example.test/rerank"),records.append)
        provider.client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:self.assertEqual(await provider.rerank("question",["original one","original two"]),[.2,.8])
        finally:await provider.aclose()
        self.assertEqual(records[0]["prompt_tokens"],17)
        self.assertTrue(records[0]["usage_reported"])

    async def test_duplicate_decision_only_invalidates_that_identity(self):
        original=events();provider=FixtureProvider();normal=provider.verify
        async def verify(query,batch):
            raw=await normal(query,batch);raw["decisions"].append(raw["decisions"][0]);return raw
        provider.verify=verify
        decisions,errors,_=await verify_candidates(provider,{},[{"event":e} for e in original],Settings(),time.monotonic()+5)
        self.assertEqual(list(decisions),[original[1]["event_id"]])
        self.assertEqual(errors[0]["reason"],"duplicate_decision")
