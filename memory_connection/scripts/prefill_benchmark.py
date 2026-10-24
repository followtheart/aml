"""Prepare another isolated conversation while the main run evaluates the first.

Publish complete SQLite backups only if destinations do not yet exist. The main
runner will replay the same idempotent requests; no request semantics are changed.
"""
import argparse
import asyncio
import json
import os
import sqlite3
from pathlib import Path
from .benchmark_metrics import prepare
from .real_benchmark import load_env, configuration, execute, save


def publish(output, root, conversation):
    status = json.loads((output/"status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        raise RuntimeError("prefill ingestion is not complete")
    databases = list(output.glob("*.db"))
    if any((root/db.name).exists() for db in databases):
        raise RuntimeError("main run already created a destination; refusing to overwrite")
    for db in databases:
        target = root / db.name
        if target.exists():
            raise RuntimeError("main run already created destination; refusing to overwrite")
        temporary = root / (db.name + ".prefill-transfer")
        source, dest = sqlite3.connect(db), sqlite3.connect(temporary)
        try:
            source.backup(dest)
        finally:
            source.close()
            dest.close()
        temporary.rename(target)
    save(output/"published.json", {"conversation_id": conversation, "published": True,
                                  "note": "Include this folder's calls in run cost and index-building totals."})
    print("prefill published", conversation, flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--conversation", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--publish-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.run)
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    output = root / ("prefill-" + args.conversation)
    if args.publish_only:
        publish(output, root, args.conversation)
        return
    output.mkdir(exist_ok=args.resume)
    load_env(args.config)
    os.environ["AML_SEARCH_DEADLINE_SECONDS"] = "120"
    settings, _ = configuration()
    plan = prepare("memory_system/data/locomo_eval.json", "memory_system/data/locomo10.json",
                   len(manifest["selected"]), len(manifest["selected"][0]["qa_ids"]), manifest["seed"])
    conv = next(c for c in plan if c["conversation_id"] == args.conversation)
    conv["qa"] = []
    options = argparse.Namespace(search_seconds=120, expansion_seconds=30, verify_seconds=80,
                                 evidence_bytes=manifest["evidence_bytes"], chunk_messages=manifest["chunk_messages"],
                                 top_k=manifest["top_k"], seed=manifest["seed"], add_concurrency=4, query_concurrency=1)
    prices = json.loads((root/"prices.snapshot.json").read_text(encoding="utf-8"))
    await execute(options, [conv], settings, {}, output, prices)
    publish(output, root, args.conversation)


if __name__ == "__main__":
    asyncio.run(main())
