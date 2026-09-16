"""Offline regression coverage for oversized answer contexts."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context as context, config, eval_scoring


class ContextTests(unittest.TestCase):
    def test_provenance_removed_but_evidence_time_preserved(self):
        sources = [{"content": "Caroline: I transitioned last year.",
                    "timestamp": 1685020440000, "role": "user",
                    "request_id": "LONG_PRIVATE_ID", "session_id": "SESSION"},
                   {"request_id": "OMITTED_ID", "content_omitted": "duplicate"}]
        memory = {"content": "[event: 2023] identity" + context.SOURCE_MARKER +
                  "OLD_VERBOSE_JSON", "sources": sources}
        original = copy.deepcopy(memory)
        result = context.build([memory])
        for text in ("Caroline:", "last year", "1685020440000", "2023", "user"):
            self.assertIn(text, result)
        for text in ("LONG_PRIVATE_ID", "SESSION", "OMITTED_ID", "OLD_VERBOSE_JSON"):
            self.assertNotIn(text, result)
        self.assertEqual(memory, original)

    def test_total_and_item_limits_apply_to_all_types(self):
        memories = [{"content": kind + "中文🙂" * 10000, "memory_type": kind}
                    for kind in ("fact", "episode", "session_summary")]
        with patch.object(config, "ANSWER_CONTEXT_MAX_CHARS", 1000), \
                patch.object(config, "ANSWER_CONTEXT_ITEM_MAX_CHARS", 400):
            result = context.build(memories)
        self.assertEqual(len(result), 1000)
        self.assertIn("fact", result)
        self.assertIn("episode", result)
        self.assertIn("session_summary", result)
        self.assertIn("[truncated]", result)

    def test_duplicates_and_plain_memories(self):
        self.assertEqual(context.build([{"content": "one"}, {"content": "one"},
                                        {"content": "two"}]), "one\ntwo")
        self.assertEqual(context.build([]), "(no memories)")

    def test_episode_covered_by_ranked_facts_is_dropped(self):
        src = lambda i: {"request_id": "r", "message_index": i, "content": f"m{i}",
                         "timestamp": 1, "role": "user"}
        memories = [
            {"content": "fact A", "memory_type": "fact", "sources": [src(0)]},
            {"content": "fact B", "memory_type": "fact", "sources": [src(1)]},
            {"content": "episode AB", "memory_type": "episode", "sources": [src(0), src(1)]},
            {"content": "episode BC", "memory_type": "episode", "sources": [src(1), src(2)]},
            {"content": "fact A again", "memory_type": "fact", "sources": [src(0)]},
        ]
        result = context.build(memories)
        self.assertNotIn("episode AB", result)
        self.assertIn("episode BC", result)
        self.assertIn("fact A again", result)

    def test_question_options_and_instructions_outside_budget(self):
        qa = {"question": "QUESTION_END", "question_date": "2023-05-25",
              "scoring": "choice", "qa_type": "single_choice",
              "options": ["A. OPTION_END"], "system_prompt": "TASK_INSTRUCTION"}
        with patch.object(config, "ANSWER_CONTEXT_MAX_CHARS", 300):
            prompt = eval_scoring.answer_prompt(qa, [{"content": "x" * 100000}])
        for text in ("QUESTION_END", "2023-05-25", "A. OPTION_END", "TASK_INSTRUCTION"):
            self.assertIn(text, prompt)
        self.assertLess(len(prompt), 1500)

    def test_binary_template_also_bounded(self):
        with patch.object(config, "ANSWER_CONTEXT_MAX_CHARS", 300):
            prompt = eval_scoring.answer_prompt({"question": "QUESTION_END"},
                                                [{"content": "x" * 100000}])
        self.assertIn("QUESTION_END", prompt)
        self.assertNotIn("x" * 301, prompt)


if __name__ == "__main__":
    unittest.main()
