"""Offline regressions for explicit task-feedback experience memory."""
import os
import sys
import unittest
from pathlib import Path

os.environ["AML_FAKE"] = "1"
os.environ["AML_MEMORY_DEBUG_LOG"] = ""
os.environ["AML_SEARCH_DEBUG_LOG"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import experience, schemas, store


class ExperienceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(":memory:")

    def tearDown(self):
        self.st.conn.close()

    def request(self, outcome="success", verified=False):
        return schemas.FeedbackRequest(
            user_id="u", session_id="s1", task="Fix parser test",
            outcome=outcome, trace="Changed parser and ran the failing test.",
            task_signature="python:test:parser", environment_verified=verified)

    async def test_feedback_creates_procedural_memory(self):
        ids = await experience.run_feedback(self.st, self.request("failure"))
        item = self.st.get_amus_by_ids(ids)[0]
        self.assertEqual(item["type"], "strategy")
        self.assertEqual(item["polarity"], "failure")
        self.assertEqual(item["harmful"], 0)
        self.assertEqual(item["epistemic_status"], "inferred")
        self.assertFalse(item["verified"])

    async def test_repeated_feedback_updates_counter(self):
        first = await experience.run_feedback(self.st, self.request())
        req = self.request()
        req.session_id = "s2"
        req.feedback_event_id = "event-2"
        req.used_memory_ids = first
        req.attribution_reason = "Applied this strategy in the traced attempt"
        second = await experience.run_feedback(self.st, req)
        self.assertEqual(first, second)
        item = self.st.get_amus_by_ids(first)[0]
        self.assertEqual(item["helpful"], 1)
        await experience.run_feedback(self.st, req)
        self.assertEqual(self.st.get_amus_by_ids(first)[0]["helpful"], 1)
        self.assertIn("event-2", item["support_sessions"])


if __name__ == "__main__":
    unittest.main(verbosity=2)