"""Local proxy evaluation harness (design doc v0.2 §10).

Runs LoCoMo-style data through Add -> Search -> Answer -> Judge entirely
in-process, with ablation switches. Uses prompts 06/07 for Answer/Judge.

Data format (LoCoMo-compatible, simplified):
[
  {
    "conversation_id": "conv-0",
    "sessions": [[{"role": "user", "content": "...", "timestamp": ms}, ...], ...],
    "qa": [{"question": "...", "answer": "...", "category": "temporal", ...}]
  }
]

Usage:
  AML_FAKE=1 python scripts/local_eval.py --data data/sample_eval.json
  AML_LLM_MODEL=gpt-4o-mini python scripts/local_eval.py --data data/locomo10.json \
      --limit 50   (needs OPENAI_API_KEY; real scoring)

Ablation flags: --no-graph --no-governance --no-rerank --no-keyexp
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import add_pipeline, prompts, schemas, search_pipeline, store, llm
from app import config, eval_data, eval_scoring


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=0, help="max QA pairs")
    p.add_argument("--convs", type=int, default=0, help="max conversations")
    p.add_argument("--dataset", choices=eval_data.DATASETS, default="normalized")
    p.add_argument("--history-dir", help="PersonaMem/ScriptMem source histories")
    p.add_argument("--size", choices=["32k", "128k"], default="32k")
    p.add_argument("--output", help="Write per-question results as JSONL")
    p.add_argument("--inspect", action="store_true", help="Validate/count only; no model calls")
    p.add_argument("--chunk-messages", type=int, default=20)
    p.add_argument("--chunk-words", type=int, default=2000)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--no-governance", action="store_true")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-keyexp", action="store_true")
    p.add_argument("--no-progress", action="store_true",
                   help="disable the live progress bar")
    return p.parse_args()


class Progress:
    """Dependency-free async progress bar that keeps moving during API waits."""

    def __init__(self, total, enabled=True):
        self.total = max(0, total)
        self.enabled = enabled
        self.done = 0
        self.started = time.monotonic()
        self.label = "starting"
        self._last_width = 0
        self._tty = sys.stdout.isatty()

    @staticmethod
    def _duration(seconds):
        seconds = max(0, int(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _line(self):
        ratio = self.done / self.total if self.total else 1.0
        width = 24
        filled = min(width, int(width * ratio))
        bar = "=" * filled + ">" + "." * max(0, width - filled - 1)
        if self.done >= self.total:
            bar = "=" * width
        elapsed = time.monotonic() - self.started
        eta = "--:--"
        if self.done:
            eta = self._duration(elapsed / self.done *
                                 (self.total - self.done))
        label = self.label.replace("\n", " ")[:58]
        return (f"[{bar}] {self.done}/{self.total} {ratio:6.1%} "
                f"elapsed {self._duration(elapsed)} eta {eta}  {label}")

    def render(self, force=False):
        if not self.enabled:
            return
        line = self._line()
        if self._tty:
            sys.stdout.write("\r" + line.ljust(self._last_width))
            sys.stdout.flush()
            self._last_width = max(self._last_width, len(line))
        elif force:
            print(line, flush=True)

    async def run(self, awaitable, label):
        self.label = label
        self.render(force=True)
        task = asyncio.ensure_future(awaitable)
        while not task.done():
            await asyncio.wait({task}, timeout=1.0)
            if self._tty:
                self.render()
        result = await task
        self.done += 1
        self.render(force=not self._tty)
        return result

    def write(self, text):
        if self.enabled and self._tty:
            sys.stdout.write("\r" + " " * self._last_width + "\r")
        print(text, flush=True)
        if self.enabled and self._tty:
            self.render()

    def finish(self):
        if not self.enabled:
            return
        self.label = "complete"
        if self._tty:
            self.render()
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif self.done < self.total:
            self.render(force=True)


def build_plan(data, args):
    """Select conversations and QA pairs before ingestion.

    This prevents a small global --limit from needlessly ingesting every
    later conversation after the requested QA count has already been met.
    """
    return list(eval_data.iter_plan(data, args.limit, args.convs))


class Ablate:
    """Monkey-patch level ablation, applied once per run."""
    graph = governance = rerank = keyexp = True


_orig_govern = add_pipeline._govern_one
_orig_recall = search_pipeline._recall
_orig_rerank = search_pipeline._filter_rerank


async def _govern_passthrough(st, user_id, fact, vec):
    return "ADD", {"operation": "ADD", "target_id": None}


async def _recall_no_graph(st, req, plan):
    routes = await _orig_recall(st, req, plan)
    return [r for r in routes if r and not _looks_graph(r)] or routes


def _looks_graph(route):  # heuristic: graph route items lack _score/_fused
    return all("_score" not in i for i in route)


async def _rerank_passthrough(req, plan, fused, scored=None):
    # Preserve the requested result capacity during the no-rerank ablation.
    return [dict(c, _final=c.get("_fused", 0)) for c in fused[:req.top_k]]


def apply_ablations(args):
    if args.no_governance:
        add_pipeline._govern_one = _govern_passthrough
    if args.no_graph:
        search_pipeline.graph.ppr_recall = lambda *a, **k: []
    if args.no_rerank:
        search_pipeline._filter_rerank = _rerank_passthrough
    if args.no_keyexp:
        async def _plain(req, anchor=None):
            return {"intent": "fact", "time_scope": None, "entities": [],
                    "sub_queries": [req.query],
                    "expanded_queries": [req.query]}
        search_pipeline._understand = _plain


async def answer_and_judge(question, gold, memories):
    mem_text = eval_scoring.answer_context.build(memories)
    ap = prompts.render("06_eval_answer.txt",
                        speaker_1_user_id="A", speaker_1_memories=mem_text,
                        speaker_2_user_id="B", speaker_2_memories="(n/a)",
                        question=question)
    try:
        pred = (await llm.complete(ap, stage="eval.answer")).strip()
    except Exception:
        pred = "(llm error)"
    jp = prompts.render("07_eval_judge.txt", question=question,
                        gold_answer=gold, generated_answer=pred)
    try:
        verdict = await llm.complete_json(
            jp, '{"label":"CORRECT|WRONG"}',
            schema=llm.STRUCTURED_SCHEMAS["judge"],
            stage="eval.judge")
        correct = verdict.get("label") == "CORRECT"
    except Exception:
        correct = False
    return pred, correct


async def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    apply_ablations(args)
    if args.chunk_messages < 1 or args.chunk_words < 1:
        raise ValueError("Chunk limits must be positive")
    def selected():
        data = eval_data.convert_records(args.data, args.dataset, args.history_dir, args.size)
        return eval_data.iter_plan(data, args.limit, args.convs)

    # Two streaming passes: preflight everything selected before paid calls,
    # then execute without retaining the complete corpus in memory.
    conv_count = total_steps = qa_count = add_count = 0
    for conv, qas in selected():
        conv_count += 1
        qa_count += len(qas)
        add_count += sum(1 for _ in eval_data.ingestion_chunks(
            conv, args.chunk_messages, args.chunk_words))
    total_steps = add_count + 2 * qa_count
    print(f"protocol={eval_scoring.PROTOCOL} fake={config.FAKE} "
          f"conversations={conv_count} add_chunks={add_count} qa={qa_count}")
    if args.inspect:
        return
    if args.output and Path(args.output).resolve() == Path(args.data).resolve():
        raise ValueError("--output must not overwrite input data")
    progress = Progress(total_steps, enabled=not args.no_progress)

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    st = store.Store(tmp.name)

    n_qa, score_sum = 0, 0.0
    per_cat = defaultdict(lambda: [0, 0])
    output = None

    try:
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            output = open(args.output, "w", encoding="utf-8")
        for ci, (conv, qas) in enumerate(selected(), 1):
            cid = conv["conversation_id"]
            dataset = conv.get("dataset", args.dataset)
            uid = f"local:{dataset}:{cid}"
            for sid, chunk_id, session in eval_data.ingestion_chunks(
                    conv, args.chunk_messages, args.chunk_words):
                msgs = [schemas.Message(**m) for m in session]
                req = schemas.AddRequest(
                    request_id=f"{uid}:session:{sid}:chunk:{chunk_id}",
                    messages=msgs, user_id=uid,
                    session_id=f"{uid}:session:{sid}")
                await progress.run(
                    add_pipeline.run_add(st, req),
                    f"Add conv {ci}/{conv_count} session {sid} chunk {chunk_id}\n")
            for qi, qa in enumerate(qas, 1):
                resp = await progress.run(
                    search_pipeline.run_search(
                        st, schemas.SearchRequest(query=qa["question"],
                                                  options=qa.get("options"),
                                                  user_id=uid, top_k=100,
                                                  reference_time=qa.get("question_date"))),
                    f"Search conv {ci}/{conv_count} QA {qi}/{len(qas)}")
                pred, score, diagnostics = await progress.run(
                    eval_scoring.evaluate(
                        qa,
                        [d.dict() for d in resp.data]),
                    f"Answer/Judge conv {ci}/{conv_count} QA {qi}/{len(qas)}")
                n_qa += 1
                score_sum += score
                cat = f"{dataset}/{qa.get('category', 'unknown')}"
                per_cat[cat][0] += score
                per_cat[cat][1] += 1
                if output:
                    result = {"dataset": dataset, "conversation_id": cid,
                              "qa_id": qa.get("id", str(qi - 1)), "prediction": pred,
                              "predicted_answer": pred,
                              "score": score, "scoring": qa.get("scoring", "binary"),
                              "category": qa.get("category", "unknown"),
                              "protocol": eval_scoring.PROTOCOL, "fake": config.FAKE,
                              "model": config.LLM_MODEL, **diagnostics}
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                progress.write(
                    f"[score={score:.2f}] {cat}: "
                    f"{qa['question'][:60]} -> {pred[:60]}")
    finally:
        if output:
            output.close()
        progress.finish()
        st.conn.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    print("\n===== RESULTS =====")
    flags = [k for k, v in [("graph", not args.no_graph),
                            ("governance", not args.no_governance),
                            ("rerank", not args.no_rerank),
                            ("keyexp", not args.no_keyexp)] if v]
    print(f"config: {'+'.join(flags)}")
    for cat, (c, t) in sorted(per_cat.items()):
        print(f"  {cat:20s} {c}/{t} = {c/t:.2%}")
    if n_qa:
        print(f"  {'MEAN SCORE':20s} {score_sum:.2f}/{n_qa} = {score_sum/n_qa:.2%}")
    print("Local proxy results; not official AML leaderboard scores."
          + (" FAKE mode: plumbing only, scores are meaningless." if config.FAKE else ""))


if __name__ == "__main__":
    asyncio.run(main())
