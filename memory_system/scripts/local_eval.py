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
import os
import sys
import tempfile
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
    return p.parse_args()


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
        pred = (await llm.complete(ap)).strip()
    except Exception:
        pred = "(llm error)"
    jp = prompts.render("07_eval_judge.txt", question=question,
                        gold_answer=gold, generated_answer=pred)
    try:
        verdict = await llm.complete_json(
            jp, '{"label":"CORRECT|WRONG"}',
            schema=llm.STRUCTURED_SCHEMAS["judge"])
        correct = verdict.get("label") == "CORRECT"
    except Exception:
        correct = False
    return pred, correct


async def main():
    args = parse_args()
    apply_ablations(args)
    data = json.load(open(args.data, encoding="utf-8"))

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    st = store.Store(tmp.name)

    n_qa, n_correct = 0, 0
    per_cat = defaultdict(lambda: [0, 0])

    for conv in data[: args.convs or None]:
        cid = conv["conversation_id"]
        uid = f"local:{cid}"
        for si, session in enumerate(conv["sessions"]):
            msgs = [schemas.Message(**m) for m in session]
            req = schemas.AddRequest(
                request_id=f"local:{cid}:chunk-{si}",
                messages=msgs, user_id=uid,
                session_id=f"local:{cid}:s{si}")
            await add_pipeline.run_add(st, req)
        for qa in conv.get("qa", []):
            if args.limit and n_qa >= args.limit:
                break
            resp = await search_pipeline.run_search(
                st, schemas.SearchRequest(query=qa["question"],
                                          user_id=uid, top_k=100))
            pred, correct = await answer_and_judge(
                qa["question"], qa["answer"],
                [d.dict() for d in resp.data])
            n_qa += 1
            n_correct += int(correct)
            cat = qa.get("category", "unknown")
            per_cat[cat][0] += int(correct)
            per_cat[cat][1] += 1
            print(f"[{'OK' if correct else 'XX'}] {cat}: {qa['question'][:60]}"
                  f" -> {pred[:60]}")

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
    st.conn.close()
    try:
        os.unlink(tmp.name)
    except OSError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
