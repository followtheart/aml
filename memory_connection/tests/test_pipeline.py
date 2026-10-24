"""Deterministic provider fixtures prove control flow, not model accuracy."""
import asyncio
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from memory_connection.app.config import Settings
from memory_connection.app.main import create_app
from memory_connection.app.schemas import AddRequest, SearchRequest, View
from memory_connection.app.service import MemoryService
from memory_connection.app.store import Store, Conflict
from memory_connection.app.recall import recall
from memory_connection.app.providers import normalize_vectors, HTTPProvider, OfflineProvider
import time


class FixtureProvider:
    space_id = "fixture-orthogonal-v1"
    dimension = 3

    def __init__(self):
        self.seen = []
        self.verdict = "relevant"
        self.bad_span = False
        self.bad_ids = False
        self.fail_memory = False
        self.fail_query = False
        self.fail_embed = False
        self.fail_verify = False
        self.duplicates = False
        self.before_verify = None
        self.before_expand = None
        self.bad_anchor = False

    async def expand_memory(self, event):
        if self.before_expand:
            self.before_expand()
        if self.fail_memory:
            raise RuntimeError("fixture memory failure")
        source = next(s for s in event["sources"] if s["source_id"] == event["primary_source_id"])
        text = source["content"]
        view = {"kind": "applicability", "text": "共同橋接", "derivation": "associative",
                "anchor_spans": [{"source_id": source["source_id"], "start": 0, "end": len(text),
                                  "quote": "invented" if self.bad_anchor else text}]}
        return {"views": [view] * (12 if self.duplicates else 1)}

    async def expand_query(self, query):
        if self.fail_query:
            raise RuntimeError("fixture query failure")
        return {"views": [{"kind": "applicability", "text": "共同橋接", "derivation": "associative",
                           "facet": "need"}]}

    async def embed(self, texts):
        if self.fail_embed:
            raise RuntimeError("fixture embedding failure")
        return [[0., 0., 1.] if t == "共同橋接" else [0., 1., 0.] if t == "乙需" else [1., 0., 0.] for t in texts]

    async def verify(self, query, originals):
        self.seen.append(copy.deepcopy({"query": query, "events": originals}))
        if self.before_verify:
            self.before_verify()
        if self.fail_verify:
            raise RuntimeError("fixture verifier failure")
        result = []
        for event in originals:
            source = next(s for s in event["sources"] if s["source_id"] == event["primary_source_id"])
            text = source["content"]
            verdict = "irrelevant" if "朋友" in text else self.verdict
            result.append({"event_id": "nonexistent" if self.bad_ids else event["event_id"],
                           "verdict": verdict, "relation_type": "constraint", "subject_match": "yes",
                           "time_match": "unknown", "polarity": "consistent",
                           "evidence_spans": [] if verdict in {"irrelevant", "uncertain"} else [{
                               "source_id": source["source_id"], "start": 0, "end": len(text),
                               "quote": "fabricated" if self.bad_span else text}], "reason": "fixture decision"})
        return {"decisions": result}


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.provider = FixtureProvider()
        self.settings = Settings()
        self.store = Store(":memory:", self.provider.space_id)
        self.service = MemoryService(self.store, self.provider, self.settings)

    def tearDown(self):
        self.store.close()

    def add_req(self, text="甲源", user="u", rid="r", **message):
        return AddRequest(request_id=rid, user_id=user, session_id="s",
                          messages=[{"role": "user", "content": text, **message}])

    async def search(self, **kwargs):
        return await self.service.search(SearchRequest(user_id=kwargs.pop("user_id", "u"), query=kwargs.pop("query", "乙需"), debug=True, **kwargs))

    async def test_dual_expansion_bridge_returns_original(self):
        await self.service.add(self.add_req())
        result = await self.search()
        self.assertEqual([d["content"] for d in result["data"]], ["甲源"])
        self.assertTrue(any(p["a_view"] == 1 and p["b_view"] == 1 for p in result["trace"]["candidates"][0]["paths"]))
        self.assertEqual(result["verification_status"], "complete")

    async def test_four_way_ablation(self):
        counts = {}
        for memory in (False, True):
            for query in (False, True):
                settings = Settings(memory_expansion=memory, query_expansion=query)
                store = Store(":memory:", self.provider.space_id)
                service = MemoryService(store, self.provider, settings)
                try:
                    await service.add(self.add_req())
                    result = await service.search(SearchRequest(user_id="u", query="乙需"))
                    counts[f"C{int(memory)}{int(query)}"] = len(result["data"])
                finally:
                    store.close()
        self.assertEqual(counts, {"C00": 0, "C01": 0, "C10": 0, "C11": 1})

    async def test_memory_only_expansion_can_connect_to_original_query(self):
        self.service.settings = Settings(query_expansion=False)
        await self.service.add(self.add_req())
        result = await self.search(query="共同橋接")
        self.assertEqual(len(result["data"]), 1)
        self.assertTrue(any(p["a_view"] == 1 and p["b_view"] == 0 for p in result["trace"]["candidates"][0]["paths"]))

    async def test_query_only_expansion_can_connect_to_original_memory(self):
        self.service.settings = Settings(memory_expansion=False)
        await self.service.add(self.add_req("共同橋接"))
        result = await self.search()
        self.assertEqual(len(result["data"]), 1)
        self.assertTrue(any(p["a_view"] == 0 and p["b_view"] == 1 for p in result["trace"]["candidates"][0]["paths"]))

    async def test_no_expansion_direct_original_route(self):
        self.service.settings = Settings(memory_expansion=False, query_expansion=False)
        await self.service.add(self.add_req())
        result = await self.search(query="甲源")
        self.assertEqual(len(result["data"]), 1)
        self.assertTrue(all(p["a_view"] == p["b_view"] == 0 for p in result["trace"]["candidates"][0]["paths"]))

    async def test_verifier_sees_only_originals(self):
        await self.service.add(self.add_req())
        await self.search()
        payload = self.provider.seen[0]
        self.assertEqual(payload["query"]["query"], "乙需")
        self.assertNotIn("共同橋接", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("views", payload["events"][0])
        self.assertNotIn("score", payload["events"][0])

    async def test_false_bridge_is_rejected(self):
        await self.service.add(self.add_req("朋友甲源"))
        result = await self.search()
        self.assertEqual(len(result["trace"]["candidates"]), 1)
        self.assertEqual(result["data"], [])
        self.assertEqual(result["trace"]["decisions"][0]["verdict"], "irrelevant")

    async def test_contradicting_is_useful(self):
        self.provider.verdict = "contradicting"
        await self.service.add(self.add_req())
        result = await self.search()
        self.assertEqual(result["data"][0]["verdict"], "contradicting")
        self.assertEqual(result["evidence_status"], "conflicting")

    async def test_partial_not_reported_sufficient(self):
        self.provider.verdict = "partial"
        await self.service.add(self.add_req())
        result = await self.search()
        self.assertEqual(len(result["data"]), 1)
        self.assertEqual(result["evidence_status"], "incomplete")

    async def test_duplicate_views_do_not_add_votes(self):
        await self.service.add(self.add_req())
        first = await self.search()
        self.store.purge("u")
        self.provider.duplicates = True
        await self.service.add(self.add_req())
        second = await self.search()
        self.assertEqual(first["data"][0]["score"], second["data"][0]["score"])
        events = self.store.snapshot(SearchRequest(user_id="u", query="x"))[2]
        self.assertEqual(len(events[0]["views"]), 2)

    async def test_isolation_before_recall(self):
        await self.service.add(self.add_req(user="someone_else"))
        result = await self.search()
        self.assertEqual(result["trace"]["candidates"], [])
        self.assertEqual(result["metrics"]["view_comparisons"], 0)

    async def test_sensitive_and_suppressed(self):
        await self.service.add(self.add_req(sensitivity="sensitive"))
        self.assertEqual((await self.search(include_sensitive=False))["data"], [])
        self.assertEqual(len((await self.search())["data"]), 1)
        await self.service.add(self.add_req(rid="r2", sensitivity="suppressed"))
        self.assertEqual(len((await self.search())["data"]), 1)

    async def test_idempotent_and_payload_conflict(self):
        request = self.add_req()
        first = await self.service.add(request)
        self.assertEqual(first, await self.service.add(request))
        self.assertEqual(len((await self.search())["data"]), 1)
        with self.assertRaises(Conflict):
            await self.service.add(self.add_req("different"))

    async def test_atomic_embedding_failure(self):
        self.provider.fail_embed = True
        with self.assertRaises(RuntimeError):
            await self.service.add(self.add_req())
        self.assertEqual(self.store.state("u"), (0, 0))
        self.assertIsNone(self.store.replay(self.add_req()))

    async def test_memory_expansion_failure_preserves_original(self):
        self.provider.fail_memory = True
        response = await self.service.add(self.add_req())
        self.assertEqual(response["degradation"][0]["stage"], "memory_expansion")
        result = await self.search(query="甲源")
        self.assertEqual(len(result["data"]), 1)

    async def test_bad_anchor_discarded(self):
        self.provider.bad_anchor = True
        await self.service.add(self.add_req())
        self.assertEqual((await self.search())["data"], [])

    async def test_query_expansion_failure_uses_original(self):
        await self.service.add(self.add_req())
        self.provider.fail_query = True
        result = await self.search(query="甲源")
        self.assertEqual(len(result["data"]), 1)
        self.assertEqual(result["degradation"][0]["stage"], "query_expansion")

    async def test_query_embedding_failure_uses_lexical(self):
        await self.service.add(self.add_req())
        self.provider.fail_embed = True
        result = await self.search(query="甲源")
        self.assertEqual(len(result["data"]), 1)
        self.assertEqual(result["degradation"][0]["stage"], "query_embedding")

    async def test_verifier_failure_is_not_verified(self):
        await self.service.add(self.add_req())
        self.provider.fail_verify = True
        result = await self.search()
        self.assertEqual(result["data"], [])
        self.assertEqual(result["verification_status"], "failed")
        self.assertEqual(result["evidence_status"], "incomplete")

    async def test_fabricated_quote_rejected(self):
        await self.service.add(self.add_req("甲源🙂"))
        self.provider.bad_span = True
        result = await self.search()
        self.assertEqual(result["data"], [])
        self.assertEqual(result["trace"]["errors"][0]["reason"], "invalid_evidence")

    async def test_unknown_event_id_rejected(self):
        await self.service.add(self.add_req())
        self.provider.bad_ids = True
        self.assertEqual((await self.search())["verification_status"], "failed")

    async def test_unicode_spans(self):
        await self.service.add(self.add_req("甲🙂否定"))
        item = (await self.search())["data"][0]
        self.assertEqual(item["evidence_spans"][0]["end"], 4)

    async def test_context_keeps_role_and_original(self):
        req = self.add_req()
        req.messages = [req.messages[0].model_copy(update={"content": "誰對蝦過敏？", "role": "assistant"}),
                        req.messages[0].model_copy(update={"content": "是我。"})]
        await self.service.add(req)
        await self.search()
        events = self.provider.seen[0]["events"]
        event = next(e for e in events if len(e["sources"]) == 2)
        self.assertEqual(event["sources"][0]["role"], "assistant")
        self.assertEqual(event["sources"][1]["content"], "是我。")

    async def test_purge_invalidates_inflight_search_including_trace(self):
        await self.service.add(self.add_req())
        self.provider.before_verify = lambda: self.store.purge("u")
        result = await self.search()
        self.assertEqual(result["data"], [])
        self.assertNotIn("trace", result)
        self.assertEqual(result["verification_status"], "invalidated")

    async def test_purge_prevents_inflight_add_resurrection(self):
        self.provider.before_expand = lambda: self.store.purge("u")
        with self.assertRaises(Conflict):
            await self.service.add(self.add_req())
        self.assertEqual(self.store.snapshot(SearchRequest(user_id="u", query="x"))[2], [])

    async def test_versions_and_as_of(self):
        await self.service.add(self.add_req(message_id="m", valid_from="2024-01-01T00:00:00Z"))
        await self.service.add(self.add_req("新甲源", rid="r2", message_id="m", valid_from="2025-01-01T00:00:00Z"))
        current = await self.search()
        self.assertEqual([x["content"] for x in current["data"]], ["新甲源"])
        history = await self.search(as_of="2024-06-01T00:00:00Z")
        self.assertEqual([x["content"] for x in history["data"]], ["甲源"])
        self.assertEqual(len((await self.search(include_history=True))["data"]), 2)

    async def test_revision_guard(self):
        with self.assertRaises(Conflict):
            await self.search(min_revision=1)

    async def test_future_replacement_does_not_hide_current(self):
        await self.service.add(self.add_req(message_id="m", valid_from="2024-01-01T00:00:00Z"))
        await self.service.add(self.add_req("未來甲源", rid="r2", message_id="m", valid_from="2099-01-01T00:00:00Z"))
        self.assertEqual([x["content"] for x in (await self.search())["data"]], ["甲源"])
        self.assertEqual([x["content"] for x in (await self.search(as_of="2099-02-01T00:00:00Z"))["data"]], ["未來甲源"])

    async def test_reference_time_from_source_timestamp(self):
        await self.service.add(self.add_req(timestamp=1704067200000))
        await self.search()
        self.assertEqual(self.provider.seen[0]["query"]["reference_time"], "2024-01-01T00:00:00+00:00")

    async def test_no_partial_commit_across_messages(self):
        original = self.provider.embed
        async def embed(texts):
            if "FAIL" in texts:
                raise RuntimeError("second event embedding failed")
            return await original(texts)
        self.provider.embed = embed
        req = self.add_req()
        req.messages.append(req.messages[0].model_copy(update={"content": "FAIL"}))
        with self.assertRaises(RuntimeError):
            await self.service.add(req)
        self.assertEqual(self.store.snapshot(SearchRequest(user_id="u", query="x"))[2], [])

    async def test_partial_batch_failure_retains_verified_only(self):
        for i in range(9):
            await self.service.add(self.add_req(rid=str(i)))
        original = self.provider.verify
        count = 0
        async def verify(query, events):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("second batch failed")
            return await original(query, events)
        self.provider.verify = verify
        result = await self.search()
        self.assertEqual(len(result["data"]), 8)
        self.assertEqual(result["verification_status"], "partial")

    async def test_counterexample_retained_when_top_k_is_one(self):
        await self.service.add(self.add_req())
        await self.service.add(self.add_req("否定甲源", rid="r2"))
        original = self.provider.verify
        async def verify(query, events):
            response = await original(query, events)
            response["decisions"][-1]["verdict"] = "contradicting"
            return response
        self.provider.verify = verify
        result = await self.search(top_k=1)
        self.assertEqual(result["data"][0]["verdict"], "contradicting")
        self.assertTrue(result["conflicting"])
        self.assertEqual(result["evidence_status"], "incomplete")

    async def test_top_k_is_upper_bound(self):
        for i in range(4):
            await self.service.add(self.add_req(rid=str(i)))
        self.assertEqual(len((await self.search(top_k=2))["data"]), 2)

    async def test_context_overflow_is_explicit(self):
        await self.service.add(self.add_req("甲源" * 1000))
        self.service.settings = Settings(batch_bytes=256)
        result = await self.search()
        self.assertEqual(result["data"], [])
        self.assertEqual(result["trace"]["errors"][0]["reason"], "context_overflow")

    async def test_packing_budget_is_explicit(self):
        self.service.settings = Settings(evidence_bytes=256)
        await self.service.add(self.add_req())
        result = await self.search()
        self.assertEqual(result["data"], [])
        self.assertEqual(result["omitted"][0]["reason"], "evidence_bytes")

    async def test_verification_timeout(self):
        await self.service.add(self.add_req())
        async def slow(*args):
            await asyncio.sleep(.05)
        self.provider.verify = slow
        self.service.settings = Settings(verify_seconds=.001)
        result = await self.search()
        self.assertEqual(result["verification_status"], "failed")

    async def test_candidate_limits_are_event_limits(self):
        for i in range(5):
            await self.service.add(self.add_req(rid=str(i)))
        self.service.settings = Settings(route_limit=3, candidate_limit=3, verify_limit=2)
        result = await self.search()
        self.assertLessEqual(len(result["trace"]["candidates"]), 3)
        self.assertEqual(sum(len(v["events"]) for v in self.provider.seen), 2)

    async def test_search_does_not_persist_query(self):
        await self.service.add(self.add_req())
        before = self.store.db.execute("SELECT payload FROM events").fetchall()[0][0]
        await self.search()
        after = self.store.db.execute("SELECT payload FROM events").fetchall()[0][0]
        self.assertEqual(before, after)

    async def test_empty_store(self):
        result = await self.search()
        self.assertEqual(result["evidence_status"], "not_found")
        self.assertEqual(result["verification_status"], "not_run")


class InfrastructureTests(unittest.TestCase):
    def test_unique_exact_quote_repairs_model_offsets(self):
        from memory_connection.app.event_builder import resolve_spans, valid_spans
        from memory_connection.app.schemas import Span
        event = {"sources": [{"source_id": "s", "content": "人物：我對蝦過敏。"}]}
        resolved = resolve_spans([Span(source_id="s", start=0, end=2, quote="我對蝦過敏")], event)
        self.assertTrue(valid_spans(resolved, event))
        self.assertEqual(resolved[0].start, 3)
        self.assertIsNone(resolve_spans([Span(source_id="s", start=0, end=2, quote="不存在")], event))

    def test_ambiguous_quote_is_not_relocated(self):
        from memory_connection.app.event_builder import resolve_spans
        from memory_connection.app.schemas import Span
        event = {"sources": [{"source_id": "s", "content": "前綴引用與引用"}]}
        self.assertIsNone(resolve_spans([Span(source_id="s", start=0, end=2, quote="引用")], event))

    def test_http_provider_wire_contract(self):
        import httpx
        from unittest.mock import patch
        received = []
        def handle(request):
            body = json.loads(request.content)
            received.append(body)
            if request.url.path.endswith("/embeddings"):
                return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 0]}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"views":[]}'}}]})
        real_client = httpx.AsyncClient
        def factory(**kwargs):
            return real_client(transport=httpx.MockTransport(handle), **kwargs)
        provider = HTTPProvider(Settings(model="test-chat", embed_model="test-embed", embed_dimension=2, provider_api_key="test"))
        async def run():
            self.assertEqual(await provider.embed(["a", "b"]), [[1., 0.], [0., 1.]])
            self.assertEqual(await provider.expand_query({"query": "x"}), {"views": []})
        with patch("memory_connection.app.providers.httpx.AsyncClient", side_effect=factory):
            asyncio.run(run())
        self.assertEqual(received[1]["response_format"], {"type": "json_object"})
        self.assertIn("JSON schema", received[1]["messages"][0]["content"])

    def test_invalid_vectors(self):
        for vectors in ([[0, 0]], [[1, float("nan")]], [[1]], []):
            with self.assertRaises(ValueError):
                normalize_vectors(vectors, 1, 2)

    def test_space_mismatch_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test.db")
            store = Store(path, "space-a")
            store.purge("u")
            store.close()
            reopened = Store(path, "space-a")
            self.assertEqual(reopened.state("u"), (1, 1))
            reopened.close()
            with self.assertRaises(Conflict):
                Store(path, "space-b")

    def test_real_provider_requires_configuration(self):
        with self.assertRaises(ValueError):
            HTTPProvider(Settings())

    def test_http_contract_auth_and_purge(self):
        provider = FixtureProvider()
        with TestClient(create_app(Settings(db_path=":memory:", api_key="test"), provider)) as client:
            body = {"request_id": "r", "user_id": "u", "session_id": "s", "messages": [{"role": "user", "content": "甲源"}]}
            self.assertEqual(client.post("/add", json=body).status_code, 401)
            headers = {"Authorization": "Bearer test"}
            self.assertEqual(client.post("/add", json=body, headers=headers).status_code, 200)
            response = client.post("/search", json={"user_id": "u", "query": "乙需", "top_k": 1}, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()["data"]), 1)
            self.assertNotIn("trace", response.json())
            body["messages"][0]["content"] = "changed"
            self.assertEqual(client.post("/add", json=body, headers=headers).status_code, 409)
            self.assertEqual(client.delete("/users/u", headers=headers).status_code, 200)
            self.assertEqual(client.post("/search", json={"user_id": "u", "query": "乙需"}, headers=headers).json()["data"], [])

    def test_explicit_offline_mode_never_claims_verified(self):
        provider = OfflineProvider()
        async def run():
            store = Store(":memory:", provider.space_id)
            try:
                service = MemoryService(store, provider)
                await service.add(AddRequest(request_id="r", user_id="u", session_id="s", messages=[{"role": "user", "content": "hello"}]))
                result = await service.search(SearchRequest(user_id="u", query="hello"))
                self.assertEqual(result["data"], [])
                self.assertEqual(result["evidence_status"], "incomplete")
            finally:
                store.close()
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
