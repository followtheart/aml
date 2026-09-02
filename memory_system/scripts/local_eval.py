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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import add_pipeline, prompts, schemas, search_pipeline, store, llm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=0, help="max QA pairs")
    p.add_argument("--convs", type=int, default=0, help="max conversations")
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
    remaining = args.limit
    plan = []
    for conv in data[: args.convs or None]:
        if args.limit and remaining <= 0:
            break
        qas = list(conv.get("qa", []))
        if args.limit:
            qas = qas[:remaining]
            remaining -= len(qas)
        plan.append((conv, qas))
    return plan


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


async def _rerank_passthrough(req, plan, fused):
    return fused[: config_top()]


def config_top():
    return 40


def apply_ablations(args):
    if args.no_governance:
        add_pipeline._govern_one = _govern_passthrough
    if args.no_graph:
        search_pipeline.graph.ppr_recall = lambda *a, **k: []
    if args.no_rerank:
        search_pipeline._filter_rerank = _rerank_passthrough
    if args.no_keyexp:
        async def _plain(req):
            return {"intent": "fact", "time_scope": None, "entities": [],
                    "sub_queries": [req.query],
                    "expanded_queries": [req.query]}
        search_pipeline._understand = _plain


async def answer_and_judge(question, gold, memories):
    mem_text = "\n".join(m["content"] for m in memories) or "(no memories)"
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
    data = json.load(open(args.data, encoding="utf-8"))
    plan = build_plan(data, args)
    total_steps = sum(len(conv["sessions"]) + 2 * len(qas)
                      for conv, qas in plan)
    progress = Progress(total_steps, enabled=not args.no_progress)

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    st = store.Store(tmp.name)

    n_qa, n_correct = 0, 0
    per_cat = defaultdict(lambda: [0, 0])

    try:
        for ci, (conv, qas) in enumerate(plan, 1):
            cid = conv["conversation_id"]
            uid = f"local:{cid}"
            session_count = len(conv["sessions"])
            for si, session in enumerate(conv["sessions"], 1):
                msgs = [schemas.Message(**m) for m in session]
                req = schemas.AddRequest(
                    request_id=f"local:{cid}:chunk-{si - 1}",
                    messages=msgs, user_id=uid,
                    session_id=f"local:{cid}:s{si - 1}")
                await progress.run(
                    add_pipeline.run_add(st, req),
                    f"Add conv {ci}/{len(plan)} session {si}/{session_count}")
            for qi, qa in enumerate(qas, 1):
                resp = await progress.run(
                    search_pipeline.run_search(
                        st, schemas.SearchRequest(query=qa["question"],
                                                  user_id=uid, top_k=100)),
                    f"Search conv {ci}/{len(plan)} QA {qi}/{len(qas)}")
                pred, correct = await progress.run(
                    answer_and_judge(
                        qa["question"], qa["answer"],
                        [d.dict() for d in resp.data]),
                    f"Answer/Judge conv {ci}/{len(plan)} QA {qi}/{len(qas)}")
                n_qa += 1
                n_correct += int(correct)
                cat = qa.get("category", "unknown")
                per_cat[cat][0] += int(correct)
                per_cat[cat][1] += 1
                progress.write(
                    f"[{'OK' if correct else 'XX'}] {cat}: "
                    f"{qa['question'][:60]} -> {pred[:60]}")
    finally:
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
        print(f"  {'OVERALL':20s} {n_correct}/{n_qa} = {n_correct/n_qa:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
