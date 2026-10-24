"""Five-arm real-model pilot. Core memory_connection never imports the legacy system.

Preflight is the default; --run requires explicit real model configuration.
All artifacts are private local files under runs/, never credentials.
"""
import argparse
import asyncio
import contextvars
import json
import logging
import os
import random
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit
from memory_connection.app.config import Settings
from memory_connection.app.providers import HTTPProvider
from memory_connection.app.schemas import AddRequest, SearchRequest
from memory_connection.app.service import MemoryService
from memory_connection.app.store import Store
from .benchmark_metrics import prepare, common_packet, quote_metrics, summarize, file_hash, call_cost

CONTEXT = contextvars.ContextVar("benchmark_context", default={})


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


class LegacyUsage(logging.Handler):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def emit(self, record):
        text = record.getMessage()
        if not text.startswith("provider_call "):
            return
        fields = dict(re.findall(r"(\w+)=([^ ]+)", text))
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            fields[key] = int(fields.get(key, 0))
        # Old logger uses zero for missing usage. Do not claim free calls.
        fields["usage_reported"] = fields["total_tokens"] > 0
        fields["duration_s"] = float(fields.get("duration_s", 0))
        self.sink(fields)


def load_env(path):
    if path:
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if sep:
                os.environ[key.strip()] = value.strip().strip('"').strip("'")


def configuration():
    # Evaluation adapter explicitly aligns model choices; neither server changes the other.
    mapping = {"MODEL": "LLM_MODEL", "PROVIDER_API_KEY": "LLM_API_KEY", "API_BASE": "LLM_API_BASE",
               "EMBED_MODEL": "EMBED_MODEL", "EMBED_API_KEY": "EMBED_API_KEY",
               "EMBED_API_BASE": "EMBED_API_BASE", "EMBED_DIMENSION": "EMBED_DIM"}
    for mc, aml in mapping.items():
        if os.environ.get("MC_" + mc):
            value = os.environ["MC_" + mc]
            if mc in {"MODEL", "EMBED_MODEL"} and not value.startswith("openai/"):
                value = "openai/" + value
            os.environ["AML_" + aml] = value
    if os.environ.get("MC_PROVIDER_API_KEY"):
        os.environ.setdefault("AML_EMBED_API_KEY", os.environ.get("MC_EMBED_API_KEY") or os.environ["MC_PROVIDER_API_KEY"])
        os.environ.setdefault("AML_EMBED_API_BASE", os.environ.get("MC_EMBED_API_BASE") or os.environ.get("MC_API_BASE", "https://api.openai.com/v1"))
    os.environ["AML_FAKE"] = "0"
    os.environ["AML_EVAL_CHOICE_POLICY"] = "platform"
    from memory_system.app import config
    key = config.LLM_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    embed_key = config.EMBED_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    base = config.LLM_API_BASE or "https://api.openai.com/v1"
    embed_base = config.EMBED_API_BASE or "https://api.openai.com/v1"
    return Settings(model=config.LLM_MODEL.removeprefix("openai/"), provider_api_key=key,
                    api_base=base, embed_model=config.EMBED_MODEL.removeprefix("openai/"),
                    embed_api_base=embed_base, embed_api_key=embed_key, embed_dimension=config.EMBED_DIM,
                    embed_batch_size=min(config.EMBED_BATCH_SIZE, 10),
                    embed_send_dimensions=("text-embedding-3" in config.EMBED_MODEL or
                                           "text-embedding-v" in config.EMBED_MODEL or "qwen3-embedding" in config.EMBED_MODEL)), config


async def execute(args, plan, settings, manifest, output, prices):
    from memory_system.app import add_pipeline, search_pipeline, schemas as old_schemas, store as old_store, eval_scoring
    def previous(name):
        path = output / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    calls, rows = previous("calls.jsonl"), previous("results.jsonl")
    ingestions = json.loads((output / "ingestions.json").read_text(encoding="utf-8")) if (output / "ingestions.json").exists() else []
    completed = {(r["arm"], r["conversation_id"], r["qa_id"]) for r in rows}

    def sink(record):
        entry = {**CONTEXT.get(), **record}
        calls.append(entry)
        append(output / "calls.jsonl", entry)

    handler = LegacyUsage(sink)
    logger = logging.getLogger("aml.metrics")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    # Raise budgets for quality experiments equally across connection arms; record all overrides.
    settings = settings.model_copy(update={"search_seconds": args.search_seconds,
        "expansion_seconds": args.expansion_seconds, "recall_seconds": 30,
        "verify_seconds": args.verify_seconds, "add_seconds": 180, "evidence_bytes": args.evidence_bytes})
    save(output / "status.json", {"status": "running", "completed_questions": 0})
    try:
        for conv in plan:
            cid = conv["conversation_id"]
            prefill = output / ("prefill-" + cid)
            while prefill.exists() and not (prefill/"published.json").exists():
                state = json.loads((prefill/"status.json").read_text(encoding="utf-8")) if (prefill/"status.json").exists() else {}
                if state.get("status") == "complete":
                    from .prefill_benchmark import publish
                    publish(prefill, output, cid)
                    break
                if state.get("status") == "interrupted":
                    raise RuntimeError("independent prefill interrupted; resume it before continuing")
                await asyncio.sleep(5)
            stores, services = {}, {}
            legacy = None
            try:
                for family in ("C0", "C1", "legacy"):
                    CONTEXT.set({"arm": family, "phase": "add", "conversation_id": cid})
                    before = len(calls)
                    started = time.perf_counter()
                    if family == "legacy":
                        legacy = old_store.Store(str(output / f"{cid}-legacy.db"))
                    else:
                        provider = HTTPProvider(settings, sink)
                        stores[family] = Store(str(output / f"{cid}-{family}.db"), provider.space_id)
                        services[family] = MemoryService(stores[family], provider,
                            settings.model_copy(update={"memory_expansion": family == "C1"}))
                    pending_payloads = []
                    for si, session in enumerate(conv["sessions"]):
                        for start in range(0, len(session), args.chunk_messages):
                            payload = {"request_id": f"{cid}-{si}-{start}", "user_id": cid,
                                       "session_id": str(si), "messages": session[start:start+args.chunk_messages]}
                            pending_payloads.append(payload)
                    semaphore = asyncio.Semaphore(args.add_concurrency if family != "legacy" else 1)
                    async def ingest(payload):
                        async with semaphore:
                            if family == "legacy":
                                await add_pipeline.run_add(legacy, old_schemas.AddRequest(**payload))
                            else:
                                response = await services[family].add(AddRequest(**payload))
                                if response["degradation"]:
                                    append(output / "ingestion_degradation.jsonl", {"arm": family, "conversation_id": cid,
                                           "request_id": payload["request_id"], "degradation": response["degradation"]})
                    # Keep all results observed; one failed Add aborts this conversation.
                    if family == "legacy":
                        # Governance depends on history order. Never skip a failed chunk
                        # and apply later chunks before a successful retry.
                        for payload in pending_payloads:
                            await ingest(payload)
                        outcomes = []
                    else:
                        outcomes = await asyncio.gather(*(ingest(p) for p in pending_payloads), return_exceptions=True)
                    failure = next((x for x in outcomes if isinstance(x, BaseException)), None)
                    if failure:
                        raise failure
                    ingestions.append({"family": family, "conversation_id": cid,
                                       "seconds": time.perf_counter()-started, "calls": len(calls)-before,
                                       "reported_prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in calls[before:]),
                                       "reported_completion_tokens": sum(c.get("completion_tokens") or 0 for c in calls[before:]),
                                       "cost": call_cost(calls[before:], prices)})
                    save(output / "ingestions.json", ingestions)
                    print(f"ingested {cid} {family}", flush=True)
                rng = random.Random(args.seed)
                for qa in conv["qa"]:
                    arms = ["C00", "C10", "C01", "C11", "legacy"]
                    rng.shuffle(arms)
                    async def evaluate_arm(arm):
                        if (arm, cid, qa["qa_id"]) in completed:
                            return
                        CONTEXT.set({"arm": arm, "phase": "search", "conversation_id": cid, "qa_id": qa["qa_id"]})
                        started = time.perf_counter()
                        errors, response, items = [], {}, []
                        try:
                            if arm == "legacy":
                                response = (await search_pipeline.run_search(legacy, old_schemas.SearchRequest(
                                    user_id=cid, query=qa["question"], top_k=args.top_k,
                                    evidence_token_budget=args.evidence_bytes))).model_dump()
                            else:
                                base_service = services[arm[:2]]
                                service = MemoryService(base_service.store, base_service.provider,
                                    base_service.settings.model_copy(update={"query_expansion": arm[-1] == "1"}))
                                response = await service.search(SearchRequest(user_id=cid, query=qa["question"],
                                                                               top_k=args.top_k, debug=True))
                            items = response["data"]
                        except Exception as exc:
                            errors.append({"phase": "search", "type": type(exc).__name__})
                        search_time = time.perf_counter()-started
                        packet = common_packet(items, args.evidence_bytes)
                        retrieval = quote_metrics(packet, qa, conv["source_texts"])
                        CONTEXT.set({"arm": arm, "phase": "answer_judge", "conversation_id": cid, "qa_id": qa["qa_id"]})
                        eval_start = time.perf_counter()
                        # Identical model, prompt, context serializer and judge for every arm.
                        prediction, score, diagnostics = await eval_scoring.evaluate(
                            qa, [{"id": str(i), "content": json.dumps(p, ensure_ascii=False)}
                                 for i, p in enumerate(packet)])
                        if diagnostics.get("error_stage"):
                            errors.append({"phase": diagnostics["error_stage"], "type": diagnostics.get("error_type")})
                        row = {"arm": arm, "conversation_id": cid, "qa_id": qa["qa_id"],
                               "category": qa.get("category"), "question": qa["question"],
                               "gold": qa["answer"], "prediction": prediction, "score": 0.0 if errors else score,
                               "raw_judge_score": score,
                               "retrieval": retrieval, "search_seconds": search_time,
                               "answer_judge_seconds": time.perf_counter()-eval_start, "errors": errors,
                               "evidence_status": response.get("evidence_status"),
                               "degradation": response.get("degradation", []),
                               "verification_errors": response.get("verification_errors", [])}
                        rows.append(row)
                        append(output / "results.jsonl", row)
                        append(output / "evidence.jsonl", {"arm": arm, "conversation_id": cid, "qa_id": qa["qa_id"],
                               "packet": packet, "response": response})
                        save(output / "summary.json", summarize(rows, calls, prices, args.seed))
                        save(output / "status.json", {"status": "running", "completed_questions": len(rows)})
                        print(f"evaluated {cid}/{qa['qa_id']} {arm} score={score} search={search_time:.2f}s", flush=True)
                    query_gate = asyncio.Semaphore(args.query_concurrency)
                    async def bounded(arm):
                        async with query_gate:
                            await evaluate_arm(arm)
                    await asyncio.gather(*(bounded(arm) for arm in arms))
            finally:
                for service in services.values():
                    if hasattr(service.provider, "aclose"):
                        await service.provider.aclose()
                for store in stores.values():
                    store.close()
                if legacy:
                    legacy.conn.close()
        save(output / "status.json", {"status": "complete", "completed_questions": len(rows)})
    except BaseException as exc:
        save(output / "status.json", {"status": "interrupted", "error_type": type(exc).__name__, "completed_questions": len(rows)})
        raise
    finally:
        logger.removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="memory_system/data/locomo_eval.json")
    parser.add_argument("--raw", default="memory_system/data/locomo10.json")
    parser.add_argument("--config", help="local env file; secrets never copied to artifacts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--conversations", type=int, default=2)
    parser.add_argument("--questions", type=int, default=12, help="per conversation, no history truncation")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--chunk-messages", type=int, default=6)
    parser.add_argument("--add-concurrency", type=int, default=4)
    parser.add_argument("--query-concurrency", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--evidence-bytes", type=int, default=24000)
    parser.add_argument("--search-seconds", type=float, default=120)
    parser.add_argument("--expansion-seconds", type=float, default=30)
    parser.add_argument("--verify-seconds", type=float, default=80)
    parser.add_argument("--prices", help="JSON {model:{input_per_million,output_per_million}}, one explicit currency")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="replay only unfinished requests/questions in the same frozen sample")
    args = parser.parse_args()
    if min(args.conversations, args.questions, args.chunk_messages, args.top_k, args.add_concurrency, args.query_concurrency) < 1 or args.top_k > 100:
        parser.error("positive sample sizes and top_k <=100 required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    load_env(args.config)
    os.environ["AML_SEARCH_DEADLINE_SECONDS"] = str(args.search_seconds)
    # Disable hidden second clipping in the shared reader; common_packet already caps bytes.
    os.environ["AML_ANSWER_CONTEXT_MAX_CHARS"] = str(args.evidence_bytes)
    os.environ["AML_ANSWER_CONTEXT_ITEM_MAX_CHARS"] = str(args.evidence_bytes)
    plan = prepare(args.data, args.raw, args.conversations, args.questions, args.seed)
    settings, legacy_config = configuration()
    missing = [name for name, value in (("chat_credentials", settings.provider_api_key), ("embedding_credentials", settings.embed_api_key)) if not value]
    safe_settings = settings.model_dump(exclude={"api_key", "provider_api_key", "embed_api_key", "api_base", "embed_api_base",
                                               "rerank_api_key", "rerank_api_url"})
    safe_settings["chat_host"] = urlsplit(settings.api_base).hostname
    safe_settings["embedding_host"] = urlsplit(settings.embed_api_base).hostname
    manifest = {"protocol": "five-arm-real-pilot-v1", "seed": args.seed,
        "dataset_hash": file_hash(args.data), "raw_hash": file_hash(args.raw),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {str(p): file_hash(p) for root in ("memory_connection", "memory_system/app", "prompts")
                          for p in Path(root).rglob("*") if p.is_file() and p.suffix in {".py", ".txt"} and "runs" not in p.parts},
        "settings": safe_settings, "connection_budget_overrides": {"search": args.search_seconds, "expansion": args.expansion_seconds, "verify": args.verify_seconds},
        "legacy_budget": {k: getattr(legacy_config, k, None) for k in ("SEARCH_DEADLINE_SECONDS", "SEARCH_MAX_CALLS", "SEARCH_MAX_TOKENS", "CE_MODEL")},
        "top_k": args.top_k, "evidence_bytes": args.evidence_bytes, "chunk_messages": args.chunk_messages,
        "connection_add_concurrency": args.add_concurrency, "legacy_add_concurrency": 1,
        "query_concurrency": args.query_concurrency,
        "selected": [{"conversation_id": c["conversation_id"], "messages": sum(map(len,c["sessions"])),
                      "qa_ids": [q["qa_id"] for q in c["qa"]]} for c in plan],
        "missing_configuration": missing,
        "note": "C00/C01 share C0 index; C10/C11 share C1 index. Add cost reported per family, do not double-count actual spend."}
    if args.resume:
        old = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        for key in ("dataset_hash", "raw_hash", "selected", "top_k", "evidence_bytes", "chunk_messages", "settings", "connection_budget_overrides"):
            if old[key] != manifest[key]:
                raise ValueError(f"resume changes frozen configuration: {key}")
        if (output/"results.jsonl").exists() and old.get("query_concurrency", 1) != args.query_concurrency:
            raise ValueError("cannot change query concurrency after scoring has begun")
        append(output / "resumes.jsonl", {"time": time.time(), "source_hashes": manifest["source_hashes"],
                                         "query_concurrency": args.query_concurrency})
        old["query_concurrency"] = args.query_concurrency
        save(output/"manifest.json", old)
    else:
        save(output / "manifest.json", manifest)
    prices = json.loads(Path(args.prices).read_text(encoding="utf-8")) if args.prices else {}
    if prices:
        save(output / "prices.snapshot.json", prices)
    print(json.dumps({"selected": manifest["selected"], "missing_configuration": missing, "output": str(output)}, ensure_ascii=False))
    if missing or not args.run:
        save(output / "status.json", {"status": "blocked_missing_credentials" if missing else "preflight_complete", "real_calls": 0})
        return 2 if args.run and missing else 0
    asyncio.run(execute(args, plan, settings, manifest, output, prices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
