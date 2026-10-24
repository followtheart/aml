import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from memory_connection.scripts.benchmark_metrics import prepare, quote_metrics, common_packet, call_cost, summarize


class MetricsTests(unittest.TestCase):
    def test_prefill_publishes_closed_sqlite_backups(self):
        import sqlite3
        from memory_connection.scripts.prefill_benchmark import publish
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root/"prefill"
            output.mkdir()
            (output/"status.json").write_text('{"status":"complete"}')
            conn = sqlite3.connect(output/"c-C0.db")
            conn.execute("create table marker(value text)")
            conn.execute("insert into marker values ('complete')")
            conn.commit()
            conn.close()
            publish(output, root, "c")
            conn = sqlite3.connect(root/"c-C0.db")
            self.assertEqual(conn.execute("select value from marker").fetchone()[0], "complete")
            conn.close()
            with self.assertRaises(RuntimeError):
                publish(output, root, "c")

    def test_sample_is_frozen_and_history_is_complete(self):
        args = ("memory_system/data/locomo_eval.json", "memory_system/data/locomo10.json")
        a, b = prepare(*args), prepare(*args)
        self.assertEqual(a, b)
        self.assertEqual(sum(len(c["qa"]) for c in a), 24)
        self.assertEqual(sum(len(s) for c in a for s in c["sessions"]), 1248)
        self.assertTrue(all("answer" not in m for c in a for s in c["sessions"] for m in s))

    def test_quote_coverage_not_answer_substring(self):
        qa = {"answer": "Paris", "evidence": ["D1", "D2"]}
        sources = {"D1": "I moved to Paris.", "D2": "It happened yesterday."}
        self.assertEqual(quote_metrics([{"content": "Paris"}], qa, sources)["gold_quote_recall"], 0)
        self.assertEqual(quote_metrics([{"content": sources["D1"]}], qa, sources)["gold_quote_recall"], .5)

    def test_missing_gold_is_not_zero_recall(self):
        self.assertIsNone(quote_metrics([], {"evidence": []}, {})["gold_quote_recall"])
        self.assertFalse(quote_metrics([], {"evidence": ["missing"]}, {})["eligible"])

    def test_whole_item_byte_cap(self):
        packet = common_packet([{"content": "🙂"*1000}, {"content": "small"}], 100)
        self.assertEqual(packet, [{"content": "small", "sources": []}])
        self.assertLessEqual(len(json.dumps(packet, ensure_ascii=False).encode()), 100)

    def test_unknown_cost_not_zero(self):
        record = {"model": "m", "status": "ok", "usage_reported": True, "prompt_tokens": 100, "completion_tokens": 10}
        self.assertIsNone(call_cost([record], {})["priced_reported_usage_cost"])
        prices = {"m": {"input_per_million": 1, "output_per_million": 2}}
        priced = call_cost([record], prices)
        self.assertAlmostEqual(priced["priced_reported_usage_cost"], .00012)
        self.assertTrue(priced["cost_complete"])
        self.assertFalse(call_cost([{**record, "usage_reported": False}], prices)["cost_complete"])

    def test_paired_summary(self):
        rows = []
        for arm, score in (("C11", 1), ("C00", 0)):
            rows.append({"arm": arm, "conversation_id": "c", "qa_id": "q", "score": score,
                         "search_seconds": 2, "retrieval": {"eligible": False}})
        summary = summarize(rows, [], {})
        self.assertEqual(summary["paired_differences"]["C00"]["accuracy_delta_C11_minus_baseline"], 1)
        self.assertIsNone(summary["paired_differences"]["C00"]["cluster_bootstrap_95"])


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_five_arm_adapter_contract_without_network(self):
        from memory_connection.scripts.real_benchmark import execute
        from memory_connection.app.config import Settings
        from memory_connection.tests.test_pipeline import FixtureProvider
        from memory_system.app import schemas
        args = argparse.Namespace(search_seconds=120, expansion_seconds=30, verify_seconds=80,
                                  evidence_bytes=24000, chunk_messages=6, top_k=10, seed=1, add_concurrency=1, query_concurrency=5)
        plan = [{"conversation_id": "c", "sessions": [[{"role": "user", "content": "甲源"}]],
                 "qa": [{"qa_id": "q", "question": "乙需", "answer": "甲源", "evidence": ["d"]}],
                 "source_texts": {"d": "甲源"}}]
        async def evaluate(qa, memories):
            self.assertTrue(all(isinstance(m, dict) for m in memories))
            return "甲源", 1.0, {}
        with tempfile.TemporaryDirectory() as tmp:
            with patch("memory_connection.scripts.real_benchmark.HTTPProvider", side_effect=lambda *args: FixtureProvider()), \
                 patch("memory_system.app.add_pipeline.run_add", new=AsyncMock()), \
                 patch("memory_system.app.search_pipeline.run_search", new=AsyncMock(return_value=schemas.SearchResponse(data=[]))), \
                 patch("memory_system.app.eval_scoring.evaluate", side_effect=evaluate):
                await execute(args, plan, Settings(), {}, Path(tmp), {})
            rows = [json.loads(s) for s in (Path(tmp)/"results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({r["arm"] for r in rows}, {"C00", "C10", "C01", "C11", "legacy"})
            self.assertEqual(json.loads((Path(tmp)/"status.json").read_text())["status"], "complete")
