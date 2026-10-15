"""Offline regression coverage for oversized answer contexts."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context as context, answer_choice, config, eval_scoring, evidence_packet


class ContextTests(unittest.TestCase):
    def legacy_packet(self):
        sources = [dict(role='user', timestamp=1684972800000,
                        content='I practice yoga.\nSource messages:\nA quoted marker.'),
                   dict(role='assistant', timestamp=None, content='Try a quiet morning.'),
                   dict(role='user', timestamp=None, content='Please forget that I enjoy skiing.'),
                   dict(role='user', content_omitted='duplicate')]
        suffix = context.LEGACY_SOURCE_MARKER + '\n'.join(
            f"[timestamp: {s.get('timestamp')}; role: {s.get('role')}] {s['content']}"
            for s in sources if s.get('content'))
        packet = [dict(id='legacy', content='Summary' + suffix,
                       sources=sources, packet_hash_version=4)]
        packet[0]['packet_hash'] = evidence_packet.digest(packet)
        return packet

    def test_verified_legacy_packet_uses_new_format_and_restores_catalog(self):
        packet = self.legacy_packet()
        original = copy.deepcopy(packet)
        expected = context.with_evidence('Summary', packet[0]['sources'])
        self.assertEqual(context.build(packet), expected)
        self.assertIn('[25 May 2023, User]', expected)
        sources, constraints = answer_choice.build_catalog(packet)
        self.assertEqual([s['role'] for s in sources.values()], ['user', 'assistant'])
        self.assertEqual(len(constraints), 1)
        self.assertEqual(packet, original)
        self.assertEqual(evidence_packet.digest(packet), packet[0]['packet_hash'])

    def test_verified_legacy_packet_rejects_tampering_before_conversion(self):
        for field in ('content', 'role'):
            packet = self.legacy_packet()
            packet[0]['sources'][0][field] = 'tampered'
            with self.assertRaises(ValueError):
                context.build(packet)

    def test_verified_legacy_packet_does_not_expose_hidden_sources(self):
        packet = self.legacy_packet()
        packet[0]['sources'].append(dict(role='user', content='Hidden evidence.'))
        packet[0]['packet_hash'] = evidence_packet.digest(packet)
        self.assertEqual(context.build(packet), packet[0]['content'])
        sources, constraints = answer_choice.build_catalog(packet)
        self.assertEqual((sources, constraints), ({}, {}))

    def test_verified_current_packet_stays_unchanged(self):
        packet = self.legacy_packet()
        packet[0]['content'] = context.with_evidence('Summary', packet[0]['sources'])
        packet[0]['packet_hash'] = evidence_packet.digest(packet)
        self.assertEqual(context.build(packet), packet[0]['content'])

    def test_provenance_removed_but_evidence_time_preserved(self):
        sources = [{"content": "Caroline: I transitioned last year.",
                    "timestamp": 1685020440000, "role": "user",
                    "request_id": "LONG_PRIVATE_ID", "session_id": "SESSION"},
                   {"request_id": "OMITTED_ID", "content_omitted": "duplicate"}]
        memory = {"content": "[event: 2023] identity" + context.SOURCE_MARKER +
                  "OLD_VERBOSE_JSON", "sources": sources}
        original = copy.deepcopy(memory)
        result = context.build([memory])
        for text in ("Caroline:", "last year", "25 May 2023 13:14 UTC", "2023", "User"):
            self.assertIn(text, result)
        for text in ("LONG_PRIVATE_ID", "SESSION", "OMITTED_ID", "OLD_VERBOSE_JSON", "1685020440000"):
            self.assertNotIn(text, result)
        self.assertEqual(memory, original)

    def test_readable_time_accepts_epoch_ms_iso_and_garbage(self):
        self.assertEqual(context.readable_time(1684972800000), "25 May 2023")
        self.assertEqual(context.readable_time("2023-05-25T13:14:00Z"), "25 May 2023 13:14 UTC")
        self.assertEqual(context.readable_time("2023-05-25T15:14:00+02:00"), "25 May 2023 13:14 UTC")
        self.assertIsNone(context.readable_time(None))
        self.assertEqual(context.readable_time("not a date"), "not a date")
        self.assertEqual(context.readable_time("2025"), "2025")
        self.assertEqual(context.readable_time("2025-03"), "March 2025")

    def test_legacy_marker_in_persisted_response_is_still_split(self):
        memory = {"content": "body" + context.LEGACY_SOURCE_MARKER + "OLD",
                  "sources": [{"content": "quoted", "role": "assistant", "timestamp": None}]}
        result = context.build([memory])
        self.assertEqual(result, "body" + context.SOURCE_MARKER + "[Assistant] quoted")

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
