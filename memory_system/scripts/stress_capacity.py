#!/usr/bin/env python3
"""Single-user capacity stress test for BEAM-scale histories (in-process, no HTTP).

Ingests a synthetic multi-session history through the real Add pipeline, then
runs searches, and reports Add/Search latency percentiles, memory counts,
SQLite size and vector-cache load time. Default AML_FAKE=1 measures storage,
indexing and packing cost without provider calls; unset it to include model
latency (paid). Example:

    AML_FAKE=1 python scripts/stress_capacity.py --chunks 200 --searches 20
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time

os.environ.setdefault("AML_FAKE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import add_pipeline, config, schemas, search_pipeline, store  # noqa: E402

TOPICS = ["photography", "marathon training", "sourdough baking", "guitar lessons", "job interview",
          "apartment hunting", "cat Milo", "trip to Lisbon", "budget spreadsheet", "night shifts",
          "garden tomatoes", "chess club", "thesis defence", "knee surgery", "learning Spanish"]
VERBS = ["started", "finished", "postponed", "enjoyed", "struggled with", "planned", "cancelled", "resumed"]
TIMES = ["last week", "yesterday", "in March", "two months ago", "next Friday", "this summer", "on Monday"]


def synthetic_messages(rng, count, base_ts):
    messages = []
    for i in range(count):
        topic, verb, when = rng.choice(TOPICS), rng.choice(VERBS), rng.choice(TIMES)
        if i % 2 == 0:
            text = (f"I {verb} {topic} {when}. " + " ".join(rng.choice(TOPICS).split()[0] for _ in range(rng.randint(20, 60)))
                    + f". It cost {rng.randint(5, 900)} euros and took {rng.randint(1, 12)} weeks.")
            role = "user"
        else:
            text = f"That sounds great. How is the {topic} going now?"
            role = "assistant"
        messages.append(schemas.Message(role=role, content=text, timestamp=base_ts + i * 60000))
    return messages


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))], 3)


def summary(values):
    return dict(count=len(values), p50=percentile(values, .5), p95=percentile(values, .95),
                max=round(max(values), 3) if values else None,
                mean=round(statistics.fmean(values), 3) if values else None)


async def run(args):
    db_path = args.db or str(Path(tempfile.mkdtemp(prefix="aml-stress-")) / "stress.db")
    st = store.Store(db_path)
    rng = random.Random(args.seed)
    user = f"stress:{args.seed}"
    add_latency, words = [], 0
    started = time.monotonic()
    base_ts = 1_700_000_000_000
    for chunk in range(args.chunks):
        session = f"{user}:s{chunk // args.chunks_per_session}"
        messages = synthetic_messages(rng, args.messages_per_chunk, base_ts + chunk * 86_400_000)
        words += sum(len(m.content.split()) for m in messages)
        req = schemas.AddRequest(request_id=f"{user}:c{chunk}", user_id=user, session_id=session, messages=messages)
        t = time.monotonic()
        await add_pipeline.run_add(st, req)
        add_latency.append(time.monotonic() - t)
        if (chunk + 1) % max(1, args.chunks // 10) == 0:
            print(f"add {chunk + 1}/{args.chunks} p50={percentile(add_latency, .5)}s p95={percentile(add_latency, .95)}s", flush=True)
    ingest_seconds = time.monotonic() - started
    counts = dict(st.conn.execute("SELECT type, COUNT(*) FROM amu WHERE user_id=? GROUP BY type", (user,)).fetchall())
    search_latency, returned = [], []
    for i in range(args.searches):
        topic = rng.choice(TOPICS)
        req = schemas.SearchRequest(user_id=user, query=f"When did I last work on {topic} and how much did it cost?", top_k=100)
        t = time.monotonic()
        response = await search_pipeline.run_search(st, req)
        search_latency.append(time.monotonic() - t)
        returned.append(len(response.data))
        if i == 0:
            print(f"first search {search_latency[0]:.2f}s (includes vector cache load), returned {returned[0]}", flush=True)
    report = dict(db_path=db_path, fake=config.FAKE, chunks=args.chunks, messages=args.chunks * args.messages_per_chunk,
                  approx_words=words, ingest_seconds=round(ingest_seconds, 1), memories_by_type=counts,
                  db_bytes=Path(db_path).stat().st_size, add_latency_s=summary(add_latency),
                  search_latency_s=summary(search_latency), search_returned=summary([float(x) for x in returned]),
                  first_search_s=round(search_latency[0], 3) if search_latency else None,
                  vector_approximate=config.VECTOR_APPROXIMATE, vector_exact_limit=config.VECTOR_EXACT_LIMIT)
    st.conn.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=int, default=100, help="Add requests (platform-sized chunks)")
    parser.add_argument("--messages-per-chunk", type=int, default=20)
    parser.add_argument("--chunks-per-session", type=int, default=5)
    parser.add_argument("--searches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--db", help="SQLite path (default: temporary file)")
    parser.add_argument("--output", type=Path, help="Write the JSON report here as well")
    args = parser.parse_args()
    if min(args.chunks, args.messages_per_chunk, args.chunks_per_session, args.searches) <= 0:
        parser.error("Counts must be positive")
    report = asyncio.run(run(args))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
