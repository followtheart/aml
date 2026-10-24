"""No model calls: frozen sampling, quote coverage and paired summaries."""
import hashlib
import json
import random
import re
import statistics
from pathlib import Path


def canonical(text):
    return re.sub(r"\s+", " ", text.casefold()).strip()


def prepare(data_path, raw_path, conversations=2, questions=12, seed=20261023, exclude=()):
    records = json.loads(Path(data_path).read_text(encoding="utf-8"))
    records = [r for r in records if r["conversation_id"] not in set(exclude)]
    raw = {r["sample_id"]: r for r in json.loads(Path(raw_path).read_text(encoding="utf-8"))}
    rng = random.Random(seed)
    selected = rng.sample(records, min(conversations, len(records)))
    plan = []
    for record in selected:
        source = raw[record["conversation_id"]]["conversation"]
        sessions, gold = [], {}
        for index, session in enumerate(record["sessions"], 1):
            original = source[f"session_{index}"]
            if len(original) != len(session):
                raise ValueError("raw and normalized histories do not align")
            messages = []
            for message, origin in zip(session, original):
                if message["content"] != f'{origin["speaker"]}: {origin["text"]}':
                    raise ValueError("raw and normalized source text mismatch")
                mid = origin["dia_id"]
                if mid in gold:
                    raise ValueError("duplicate source id")
                gold[mid] = origin["text"]
                messages.append({**message, "message_id": mid, "speaker_id": origin["speaker"]})
            sessions.append(messages)
        # Sample before consulting answers/evidence; preserve full histories.
        indices = sorted(rng.sample(range(len(record["qa"])), min(questions, len(record["qa"]))))
        qa = [{**record["qa"][i], "qa_id": str(i)} for i in indices]
        plan.append({"conversation_id": record["conversation_id"], "sessions": sessions, "qa": qa, "source_texts": gold})
    return plan


def common_packet(items, byte_budget):
    """Same model-visible content and attributed sources for every arm; whole-item cap."""
    result, size = [], 2
    for item in items:
        entry = {}
        if "evidence_spans" in item:
            entry["evidence_spans"] = item["evidence_spans"]
        entry.update(content=item["content"], sources=item.get("sources", []))
        if "source_annotations" in item:
            entry["source_annotations"] = item["source_annotations"]
        encoded = json.dumps(entry, ensure_ascii=False)
        needed = len(encoded.encode()) + (2 if result else 0)
        if size + needed <= byte_budget:
            result.append(entry)
            size += needed
    return result


def quote_metrics(packet, qa, sources):
    gold = set(qa.get("evidence", []))
    if not gold or not gold.issubset(sources):
        return {"eligible": False, "gold_quote_recall": None, "all_gold_quotes": None,
                "reason": "missing_or_unmapped_gold_sources"}
    # Measure only text actually visible to the common answer model.
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for v in value.values():
                yield from strings(v)
        elif isinstance(value, list):
            for v in value:
                yield from strings(v)
    visible = [canonical(text) for text in strings(packet)]
    found = {mid for mid in gold if any(canonical(sources[mid]) in text for text in visible)}
    return {"eligible": True, "gold_quote_recall": len(found)/len(gold),
            "all_gold_quotes": float(found == gold), "gold_count": len(gold), "found_ids": sorted(found)}


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    x = (len(ordered)-1)*p
    lo = int(x)
    return ordered[lo] + (ordered[min(lo+1, len(ordered)-1)]-ordered[lo])*(x-lo)


def call_cost(records, prices):
    total, complete = 0.0, True
    for record in records:
        model = record.get("model", "").removeprefix("openai/")
        rate = prices.get(model)
        if not rate or not record.get("usage_reported") or record.get("status") != "ok":
            complete = False
            continue
        if record.get("prompt_tokens") is None or record.get("completion_tokens") is None:
            complete = False
            continue
        total += (record["prompt_tokens"]*rate["input_per_million"] + record["completion_tokens"]*rate["output_per_million"])/1e6
    return {"priced_reported_usage_cost": total if records and prices else None,
            "cost_complete": complete and bool(records) and bool(prices)}


def summarize(rows, calls, prices, seed=20261023):
    result = {"arms": {}, "paired_differences": {}, "note": "quote coverage is a conservative proxy, not semantic source recall or precision"}
    for arm in sorted({r["arm"] for r in rows}):
        selected = [r for r in rows if r["arm"] == arm]
        recall = [r["retrieval"]["gold_quote_recall"] for r in selected if r["retrieval"]["eligible"]]
        latency = [r["search_seconds"] for r in selected]
        usage = [c for c in calls if c["arm"] == arm and c.get("phase") != "add"]
        result["arms"][arm] = {"questions": len(selected), "recall_eligible": len(recall),
            "accuracy_including_errors_as_zero": statistics.mean(r["score"] for r in selected),
            "gold_quote_recall": statistics.mean(recall) if recall else None,
            "search_p50_s": percentile(latency, .5), "search_p95_s": percentile(latency, .95),
            "error_questions": sum(bool(r.get("errors")) for r in selected),
            "degraded_questions": sum(bool(r.get("degradation") or r.get("verification_errors")) for r in selected),
            "provider_calls": len(usage), "reported_prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in usage if c.get("usage_reported")),
            "reported_completion_tokens": sum(c.get("completion_tokens") or 0 for c in usage if c.get("usage_reported")),
            "uncertain_usage_calls": sum(not c.get("usage_reported") for c in usage),
            "cost": call_cost(usage, prices),
            "phase_costs": {phase: call_cost([c for c in usage if c.get("phase") == phase], prices)
                            for phase in ("search", "answer_judge")}}
    # Cluster bootstrap over conversations, preserving paired questions.
    by_key = {(r["arm"], r["conversation_id"], r["qa_id"]): r for r in rows}
    for baseline in ("C00", "C10", "C01", "legacy"):
        groups = {}
        for r in rows:
            key = (baseline, r["conversation_id"], r["qa_id"])
            if r["arm"] == "C11" and key in by_key:
                groups.setdefault(r["conversation_id"], []).append(r["score"]-by_key[key]["score"])
        if not groups:
            continue
        values = [d for group in groups.values() for d in group]
        rng = random.Random(seed)
        estimates = []
        if len(groups) >= 2:
            for _ in range(2000):
                sampled = [d for key in rng.choices(list(groups), k=len(groups)) for d in groups[key]]
                estimates.append(statistics.mean(sampled))
        result["paired_differences"][baseline] = {"accuracy_delta_C11_minus_baseline": statistics.mean(values),
            "conversations": len(groups), "pairs": len(values),
            "cluster_bootstrap_95": [percentile(estimates, .025), percentile(estimates, .975)] if estimates else None,
            "exploratory_only": len(groups) < 10}
    return result


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
