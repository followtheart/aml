"""Offline tests for source mapping, leakage boundaries, chunking and scoring."""
import asyncio
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ["AML_FAKE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import eval_data as data, eval_scoring as scoring, schemas
import local_eval


def msg(text="Remember the red bicycle.", role="user"):
    return {"role": role, "content": text}


def lme():
    return {"question_id": "q1", "question": "What color?", "answer": "GOLD_RED",
            "question_type": "single-session-user", "question_date": "2023/05/30 (Tue) 23:40",
            "answer_session_ids": ["a"], "haystack_session_ids": ["a", "b"],
            "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 02:21"],
            "haystack_sessions": [[msg()], [msg("The bicycle is blue now.")]]}


def beam():
    return {"conversation_id": "beam1", "chat": [[dict(msg(), time_anchor="March-15-2024")]],
            "probing_questions": json.dumps({"information_extraction": [
                {"question": "What color?", "ideal_answer": "GOLD_RED", "rubric": ["GOLD_RUBRIC"]}]})}


def clbench():
    return {"messages": [msg("Use short sentences.", "system"), msg("Reference: red bicycle. What color?")],
            "rubrics": ["GOLD_RUBRIC"], "metadata": {"task_id": "cl1", "context_category": "rules"}}


def script():
    return {"sample_id": "conv-0", "conversation": {"session_1_date_time": "Unknown",
            "session_1": [{"speaker": "Bob", "text": "Red bicycle."}]}, "qa": [
                {"qa_type": "single_choice", "question": "What color?",
                 "option": ["A. Red", "B. Blue"], "answer": "A. Red"}]}


class DatasetTests(unittest.TestCase):
    def test_locomo_refined_raw_mapping(self):
        r = {"sample_id": "conv-26", "conversation": {"speaker_a": "Alice", "speaker_b": "Bob",
             "session_2_date_time": "1:56 pm on 8 May, 2023",
             "session_2": [{"speaker": "Bob", "text": "Hello", "dia_id": "D2:1"}],
             "session_1": [{"speaker": "Alice", "text": "Hi", "dia_id": "D1:1"}]},
             "qa": [{"question": "When?", "answer": [2022, "last year"],
                     "category": 2, "evidence": ["D2:1"], "is_multi_modality": True}],
             "observation": "GOLD_PRIVATE"}
        c = data.validate(data.locomo_refined(r))
        self.assertEqual(c["session_ids"], ["session_1", "session_2"])
        self.assertEqual(c["sessions"][1][0]["role"], "assistant")
        self.assertEqual(c["qa"][0]["id"], "conv-26#q0000")
        self.assertEqual(c["qa"][0]["answer"], ["2022", "last year"])
        self.assertNotIn("GOLD_PRIVATE", json.dumps(c))
        self.assertEqual(c["dialogue_ids"][1], ["D2:1"])

    def test_refined_candidates_are_alternatives(self):
        qa = {"question": "When?", "answer": ["2022", "last year"], "scoring": "refined_binary"}
        with patch.object(scoring.llm, "complete", AsyncMock(return_value="previous year")), \
             patch.object(scoring.llm, "complete_json", AsyncMock(side_effect=[
                 {"label": "WRONG"}, {"label": "CORRECT"}])) as judge:
            result = asyncio.run(scoring.evaluate(qa, []))
            self.assertEqual(result[1], 1)
            self.assertEqual(judge.call_count, 2)
            self.assertNotIn('"reference_answer": [', judge.call_args.args[0])
        with patch.object(scoring.llm, "complete", AsyncMock(return_value="last year")), \
             patch.object(scoring.llm, "complete_json", AsyncMock()) as judge:
            self.assertEqual(asyncio.run(scoring.evaluate(qa, []))[1], 1)
            judge.assert_not_called()

    def test_refined_gold_not_in_answer_prompt(self):
        qa = {"question": "When?", "answer": ["GOLD_SECRET"], "scoring": "refined_binary"}
        self.assertNotIn("GOLD_SECRET", scoring.answer_prompt(qa, []))

    def test_longmemeval_preserves_time_evidence_and_scope(self):
        c = data.validate(data.longmemeval(lme(), "longmemeval-s"))
        self.assertEqual(c["session_ids"], ["a", "b"])
        self.assertEqual(c["sessions"][0][0]["timestamp"], 1684549260000)
        self.assertNotIn("GOLD_RED", json.dumps(c["sessions"]))
        self.assertEqual(c["qa"][0]["question_date"], lme()["question_date"])

    def test_longmemeval_rejects_misaligned_dates(self):
        r = lme(); r["haystack_dates"] = []
        with self.assertRaises(ValueError): data.longmemeval(r, "longmemeval-s")

    def test_abstention_retained(self):
        r = lme(); r["question_id"] = "q_abs"
        self.assertTrue(data.longmemeval(r, "longmemeval-s")["qa"][0]["unanswerable"])

    def test_duplicate_source_ids_and_empty_messages(self):
        r = lme(); r["haystack_session_ids"] = ["same", "same"]
        r["haystack_sessions"][0].append(msg(""))
        c = data.validate(data.longmemeval(r, "longmemeval-s"))
        self.assertEqual(len(set(c["session_ids"])), 2)
        self.assertEqual(c["source_session_ids"], ["same", "same"])
        self.assertEqual(c["preprocessing"]["empty_messages_removed"], 1)

    def test_beam_string_payload_and_no_rubric_leak(self):
        c = data.validate(data.beam(beam()))
        self.assertNotIn("GOLD", json.dumps(c["sessions"]))
        self.assertEqual(c["qa"][0]["scoring"], "rubric_mean")

    def test_clbench_preserves_whole_task_and_system(self):
        c = data.validate(data.clbench(clbench()))
        self.assertEqual(c["qa"][0]["question"], clbench()["messages"][-1]["content"])
        self.assertNotIn("GOLD", json.dumps(c["sessions"]))
        self.assertEqual(c["qa"][0]["system_prompt"], "Use short sentences.")

    def test_clbench_rejects_answer_as_last_turn(self):
        r = clbench(); r["messages"].append(msg("answer", "assistant"))
        with self.assertRaises(ValueError): data.clbench(r)

    def test_scriptmem_choices_and_speakers(self):
        c = data.validate(data.scriptmem(script(), "angry", None))
        self.assertEqual(c["qa"][0]["gold_labels"], ["A"])
        self.assertTrue(c["sessions"][0][0]["content"].startswith("Bob:"))

    def test_scriptmem_omitted_history_rejected(self):
        r = script(); r["conversation"] = {"format_example": r["conversation"]}
        with self.assertRaisesRegex(ValueError, "omits source"):
            data.scriptmem(r, "angry", None)

    def test_persona_history_join_and_stable_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "history.json").write_text(json.dumps({"chat_history": [msg()]}))
            r = {"persona_id": "1", "chat_history_32k_link": "history.json",
                 "user_query": "{'role':'user','content':'What color?'}",
                 "correct_answer": "Red", "incorrect_answers": '["Blue", "Green"]'}
            p = root / "persona.csv"
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(r)); w.writeheader(); w.writerows([r, r])
            a = list(data.convert_records(p, "personamem-v2", root))
            self.assertEqual(a, list(data.convert_records(p, "personamem-v2", root)))
            self.assertEqual(len(a), 1)
            self.assertEqual(len(a[0]["qa"]), 2)
            for q in a[0]["qa"]:
                self.assertTrue(q["options"][ord(q["gold_labels"][0])-65].endswith("Red"))

    def test_history_path_traversal(self):
        with self.assertRaises(ValueError): data.safe_history_path("data", "../secret")

    def test_chunk_limits_order_and_losslessness(self):
        original = [dict(msg("First sentence. " + "word " * 35), timestamp=42), msg("Tail")]
        parts = list(data.chunks(original, 2, 10))
        self.assertTrue(all(len(p) <= 2 and sum(len(m["content"].split()) for m in p) <= 10 for p in parts))
        self.assertEqual("".join(m["content"] for p in parts for m in p), "".join(m["content"] for m in original))
        self.assertTrue(all(m["timestamp"] == 42 for p in parts for m in p if "timestamp" in m))

    def test_twenty_message_boundary(self):
        self.assertEqual([len(p) for p in data.chunks([msg()] * 41)], [20, 20, 1])

    def test_chunk_ids_preserve_source_session(self):
        c = data.longmemeval(lme(), "longmemeval-s")
        c["sessions"][0] *= 21
        self.assertEqual([(s, i) for s, i, _ in data.ingestion_chunks(c)], [("a", 0), ("a", 1), ("b", 0)])

    def test_limit_does_not_read_next_conversation(self):
        def source():
            yield data.longmemeval(lme(), "longmemeval-s")
            raise AssertionError("Read past --limit")
        self.assertEqual(len(list(data.iter_plan(source(), limit=1))), 1)

    def test_streaming_json_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [{"id": 0}, {"id": 1}]
            for name, text in [("a.json", json.dumps(records)), ("b.jsonl", "\n".join(map(json.dumps, records)))]:
                (root/name).write_text(text)
                self.assertEqual(list(data.read_records(root/name)), records)

    def test_duplicate_conversation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"x.jsonl"
            p.write_text((json.dumps(data.longmemeval(lme(), "longmemeval-s"))+"\n")*2)
            with self.assertRaisesRegex(ValueError, "Duplicate conversation"):
                list(data.convert_records(p, "normalized"))

    def test_choice_scoring(self):
        self.assertEqual(scoring.choice_score("(B)", ["B"], "single_choice"), 1)
        self.assertEqual(scoring.choice_score("B,A", ["A", "B"], "multi_select"), 1)
        self.assertEqual(scoring.choice_score("B,A", ["A", "B"], "ordering"), 0)
        for pred in ["A,A", "A,B,C", "I think A", ""]:
            self.assertEqual(scoring.choice_score(pred, ["A"], "single_choice"), 0)

    def test_rubric_fraction_and_all(self):
        p = {"scores": [{"index": 0, "score": 1}, {"index": 1, "score": 0.5}]}
        self.assertEqual(scoring.aggregate_rubrics(p, 2, "rubric_mean"), 0.75)
        p["scores"][1]["score"] = 0
        self.assertEqual(scoring.aggregate_rubrics(p, 2, "rubric_all"), 0)

    def test_missing_duplicate_or_invalid_rubric_scores(self):
        for entries in [[], [{"index": 0, "score": 1}]*2,
                        [{"index": 0, "score": 1}, {"index": 1, "score": 2}]]:
            with self.assertRaises(ValueError): scoring.aggregate_rubrics({"scores": entries}, 2, "rubric_mean")

    def test_answer_does_not_receive_gold_or_rubrics(self):
        for c in [data.beam(beam()), data.clbench(clbench()), data.longmemeval(lme(), "longmemeval-s")]:
            self.assertNotIn("GOLD", scoring.answer_prompt(c["qa"][0], []))

    def test_schema_rejects_bad_score_mode(self):
        c = data.longmemeval(lme(), "longmemeval-s"); c["qa"][0]["scoring"] = "unknown"
        with self.assertRaises(ValueError): data.validate(c)

    def test_choice_skips_judge(self):
        qa = data.scriptmem(script(), "angry", None)["qa"][0]
        with patch.object(scoring.llm, "complete", AsyncMock(return_value="A")), \
             patch.object(scoring.llm, "complete_json", AsyncMock(return_value={"options": []})) as judge:
            self.assertEqual(asyncio.run(scoring.evaluate(qa, []))[1], 1)
            # Choice questions never reach the judge; the only structured call
            # allowed is the pre-answer option/persona-evidence alignment.
            for call in judge.call_args_list:
                self.assertEqual(call.kwargs.get("stage"), "eval.choice_align")

    def test_provider_error_reported(self):
        qa = data.longmemeval(lme(), "longmemeval-s")["qa"][0]
        with patch.object(scoring.llm, "complete", AsyncMock(side_effect=RuntimeError())):
            pred, score, diag = asyncio.run(scoring.evaluate(qa, []))
        self.assertEqual((pred, score, diag["error_stage"]), ("", 0, "answer"))

    def test_rubric_evaluation_calls_judge_only_with_gold(self):
        qa = data.beam(beam())["qa"][0]
        with patch.object(scoring.llm, "complete", AsyncMock(return_value="Red")) as answer, \
             patch.object(scoring.llm, "complete_json", AsyncMock(return_value={
                 "scores": [{"index": 0, "score": 0.5}]})) as judge:
            self.assertEqual(asyncio.run(scoring.evaluate(qa, []))[1], 0.5)
            self.assertNotIn("GOLD", answer.call_args.args[0])
            self.assertIn("GOLD_RUBRIC", judge.call_args.args[0])

    def test_fake_rubric_evaluation_has_no_error(self):
        result = asyncio.run(scoring.evaluate(data.beam(beam())["qa"][0], []))
        self.assertEqual(result[1], 0.0)
        self.assertNotIn("error_stage", result[2])

    def test_local_eval_end_to_end_contract(self):
        c = data.scriptmem(script(), "angry", None)
        c["sessions"][0] *= 21
        adds, searches = [], []
        async def add(st, req): adds.append(req)
        async def search(st, req):
            searches.append(req)
            return schemas.SearchResponse(data=[])
        with tempfile.TemporaryDirectory() as tmp:
            path, out = Path(tmp)/"in.jsonl", Path(tmp)/"out.jsonl"
            path.write_text(json.dumps(c))
            argv = ["local_eval", "--data", str(path), "--limit", "1", "--output", str(out), "--no-progress"]
            with patch.object(sys, "argv", argv), patch.object(local_eval.add_pipeline, "run_add", add), \
                 patch.object(local_eval.search_pipeline, "run_search", search), \
                 patch.object(scoring.llm, "complete", AsyncMock(return_value="A")), redirect_stdout(io.StringIO()):
                asyncio.run(local_eval.main())
            self.assertEqual(len(adds), 2)
            self.assertEqual(adds[0].session_id, adds[1].session_id)
            self.assertNotEqual(adds[0].request_id, adds[1].request_id)
            self.assertEqual(searches[0].user_id, adds[0].user_id)
            self.assertEqual(searches[0].options, c["qa"][0]["options"])
            self.assertEqual(searches[0].query, c["qa"][0]["question"])
            self.assertEqual(searches[0].reference_time,
                             c["qa"][0].get("question_date"))
            result = json.loads(out.read_text())
            self.assertEqual(result["score"], 1)
            self.assertTrue(result["fake"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
