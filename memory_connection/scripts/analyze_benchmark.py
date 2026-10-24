"""Offline analysis of saved real runs, including amortized index costs."""
import argparse
import collections
import json
import sqlite3
import statistics
from pathlib import Path
from .benchmark_metrics import prepare, summarize, call_cost, percentile


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def analyze(root, prices):
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    rows, calls = load_rows(root/"results.jsonl"), load_rows(root/"calls.jsonl")
    for prefill in root.glob("prefill-*"):
        if (prefill/"published.json").exists():
            calls.extend(load_rows(prefill/"calls.jsonl"))
    # The legacy reranker logger substitutes a byte upper bound when usage is absent.
    # Its logs do not distinguish this from actual usage: never call it billed tokens.
    for call in calls:
        if call.get("kind") == "rerank":
            call["usage_reported"] = False
            call["usage_source"] = "provider_tokens_or_byte_upper_bound_not_distinguishable"
    evidence = load_rows(root/"evidence.jsonl")
    summary = summarize(rows, calls, prices, manifest["seed"])
    summary["status"] = json.loads((root/"status.json").read_text(encoding="utf-8"))
    summary["actual_run_cost"] = call_cost(calls, prices)
    summary["uncertain_usage_calls"] = sum(not c.get("usage_reported") for c in calls)
    summary["index_families"] = {}
    for family in ("C0", "C1", "legacy"):
        subset = [c for c in calls if c["arm"] == family and c["phase"] == "add"]
        summary["index_families"][family] = {"calls": len(subset),
            "reported_prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in subset),
            "reported_completion_tokens": sum(c.get("completion_tokens") or 0 for c in subset),
            "cost": call_cost(subset, prices)}
    event_lookup, expansion_counts = {}, collections.Counter()
    for path in root.glob("*-C[01].db"):
        with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True) as conn:
            for (payload,) in conn.execute("SELECT payload FROM events"):
                event = json.loads(payload)
                event_lookup[event["event_id"]] = event
                if path.stem.endswith("C1"):
                    expansion_counts["events"] += 1
                    expansion_counts["derived_views"] += len(event["views"])-1
                    expansion_counts["zero_derived_events"] += len(event["views"]) == 1
                    expansion_counts["failed_events"] += event.get("expansion_status") == "failed"
    summary["memory_expansion_audit"] = dict(expansion_counts)
    by_row = {(r["arm"],r["conversation_id"],r["qa_id"]):r for r in rows}
    expected = {(a, c["conversation_id"], q) for c in manifest["selected"]
                for q in c["qa_ids"] for a in ("C00", "C10", "C01", "C11", "legacy")}
    summary["integrity"] = {"expected_rows": len(expected), "actual_rows": len(rows),
        "duplicate_rows": len(rows)-len(by_row), "missing_rows": sorted(expected-set(by_row)),
        "unexpected_rows": sorted(set(by_row)-expected)}
    evidence_keys = {(r["arm"],r["conversation_id"],r["qa_id"]) for r in evidence}
    summary["integrity"].update(evidence_rows=len(evidence),
        duplicate_evidence=len(evidence)-len(evidence_keys),
        results_without_evidence=sorted(set(by_row)-evidence_keys),
        evidence_without_results=sorted(evidence_keys-set(by_row)))
    detail = collections.defaultdict(list)
    source_scores = collections.defaultdict(list)
    source_sets, question_details, pools = {}, {}, {}
    planned = prepare("memory_system/data/locomo_eval.json", "memory_system/data/locomo10.json",
                      len(manifest["selected"]), len(manifest["selected"][0]["qa_ids"]), manifest["seed"])
    conversations = {c["conversation_id"]: c for c in planned}
    raw_records = {c["sample_id"]: c for c in json.loads(
        Path("memory_system/data/locomo10.json").read_text(encoding="utf-8"))}
    category_names = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
    raw_categories = {}
    for cid, conv in conversations.items():
        raw_qa = [q for q in raw_records[cid]["qa"] if q.get("answer") is not None]
        for qa in conv["qa"]:
            original = raw_qa[int(qa["qa_id"])]
            if original["question"] != qa["question"]:
                raise ValueError("raw and normalized QA do not align")
            raw_categories[(cid,qa["qa_id"])] = original["category"]
    summary["category_audit"] = {"raw_id_counts": dict(collections.Counter(raw_categories.values())),
        "report_names": category_names,
        "note": "Legacy converter swaps names for raw IDs 3 and 4. Saved results preserve those labels; report categories use raw IDs. Scores and selected questions are unchanged."}
    def returned_ids(sources, conv):
        found = set()
        for source in sources:
            mid = source.get("message_id")
            if mid in conv["source_texts"]:
                found.add(mid)
                continue
            request = source.get("request_id", "")
            prefix = conv["conversation_id"] + "-"
            if request.startswith(prefix) and isinstance(source.get("message_index"), int):
                try:
                    session, start = map(int, request[len(prefix):].split("-"))
                    index = start + source["message_index"]
                    if session >= 0 and index >= 0:
                        found.add(conv["sessions"][session][index]["message_id"])
                except (ValueError, IndexError):
                    pass
        return found
    for item in evidence:
        key = (item["arm"], item["conversation_id"], item["qa_id"])
        # A live run can append evidence after we read the result snapshot.
        # Expose that mismatch in integrity instead of mixing snapshot generations.
        if key not in by_row:
            continue
        conv = conversations[item["conversation_id"]]
        qa = next(q for q in conv["qa"] if q["qa_id"] == item["qa_id"])
        gold = set(qa.get("evidence", []))
        if gold and gold.issubset(conv["source_texts"]):
            found = returned_ids([s for p in item["packet"] for s in p.get("sources", [])], conv)
            source_scores[item["arm"]].append(len(gold & found)/len(gold))
            source_sets[key] = {"gold": gold, "final": found}
            question_details[key] = {"arm": item["arm"], "conversation_id": item["conversation_id"],
                "qa_id": item["qa_id"], "source_id_recall": len(gold & found)/len(gold),
                "gold_ids": sorted(gold), "returned_gold_ids": sorted(gold & found)}
        response = item["response"]
        if item["arm"] != "legacy":
            trace = response.get("trace", {})
            decisions = trace.get("decisions", [])
            pools[key] = [c["event_id"] for c in trace.get("candidates", [])[:16]]
            detail[item["arm"]].append({"query_views": len(trace.get("query_views", [])),
                "candidates": len(trace.get("candidates", [])),
                "rejected": sum(d["verdict"] == "irrelevant" for d in decisions),
                "verification_errors": len(response.get("verification_errors", [])),
                "returned": len(response.get("data", [])),
                "expansion_ms": response.get("metrics", {}).get("query_expansion_ms", 0),
                "recall_ms": response.get("metrics", {}).get("recall_ms", 0),
                "verification_ms": response.get("metrics", {}).get("verification_ms", 0),
                "dual_bridge_candidates": sum(any(p["a_view"] > 0 and p["b_view"] > 0 for p in c["paths"])
                                              for c in trace.get("candidates", []))})
            if gold and gold.issubset(conv["source_texts"]):
                candidates = trace.get("candidates", [])
                candidate_sources = [s for c in candidates
                                     for s in event_lookup.get(c["event_id"], {}).get("sources", [])]
                source_sets[key]["candidate"] = returned_ids(candidate_sources, conv)
                detail[item["arm"]][-1]["candidate_source_recall"] = len(gold & returned_ids(candidate_sources, conv))/len(gold)
                pool_sources = [s for c in candidates[:16] for s in event_lookup.get(c["event_id"], {}).get("sources", [])]
                accepted_sources = [s for d in decisions if d["verdict"] in {"relevant", "partial", "contradicting"}
                                    for s in event_lookup.get(d["event_id"], {}).get("sources", [])]
                detail[item["arm"]][-1]["verification_pool_source_recall"] = len(gold & returned_ids(pool_sources, conv))/len(gold)
                detail[item["arm"]][-1]["accepted_source_recall"] = len(gold & returned_ids(accepted_sources, conv))/len(gold)
                question_details[key].update(detail[item["arm"]][-1])
    for arm, values in summary["arms"].items():
        family = "legacy" if arm == "legacy" else arm[:2]
        # Logical standalone deployment cost; shared experimental spend is counted once above.
        model_calls = [c for c in calls if (c["arm"] == family and c["phase"] == "add") or
                       (c["arm"] == arm and c["phase"] == "search")]
        values["logical_memory_cost_add_plus_search"] = call_cost(model_calls, prices)
        values["logical_memory_cost_per_query"] = (
            values["logical_memory_cost_add_plus_search"]["priced_reported_usage_cost"] / values["questions"]
            if values["logical_memory_cost_add_plus_search"]["priced_reported_usage_cost"] is not None else None)
        values["answer_judge_p50_s"] = percentile([r["answer_judge_seconds"] for r in rows if r["arm"] == arm], .5)
        values["final_source_id_recall"] = statistics.mean(source_scores[arm]) if source_scores[arm] else None
        online_calls = [c for c in calls if c["arm"] == arm and c["phase"] != "add"]
        values["provider_error_calls"] = sum(c.get("status") != "ok" for c in online_calls)
        values["provider_error_stages"] = dict(collections.Counter(
            c.get("stage", c.get("kind", "unknown")) for c in online_calls if c.get("status") != "ok"))
        failed_queries = {(c["conversation_id"], c["qa_id"]) for c in online_calls
                          if c["phase"] == "search" and c.get("status") != "ok"}
        values["search_provider_error_questions"] = len(failed_queries)
        values["search_latency_by_provider_failure"] = {}
        for failed in (False, True):
            times = [r["search_seconds"] for r in rows if r["arm"] == arm and
                     ((r["conversation_id"], r["qa_id"]) in failed_queries) == failed]
            values["search_latency_by_provider_failure"][str(failed)] = {
                "questions": len(times), "p50_s": percentile(times, .5), "p95_s": percentile(times, .95)}
        values["verification_error_reasons"] = dict(collections.Counter(
            e["reason"] for r in rows if r["arm"] == arm for e in r.get("verification_errors", [])))
        values["evidence_status_counts"] = dict(collections.Counter(
            r.get("evidence_status") for r in rows if r["arm"] == arm))
        if detail[arm]:
            keys = set().union(*(v.keys() for v in detail[arm]))
            values["pipeline_means"] = {k: statistics.mean(v[k] for v in detail[arm] if k in v) for k in keys}
        values["categories_dataset_labels"] = {str(category): {"questions": len(group),
            "accuracy": statistics.mean(r["score"] for r in group)}
            for category in sorted({r["category"] for r in rows if r["arm"] == arm}, key=str)
            if (group := [r for r in rows if r["arm"] == arm and r["category"] == category])}
        values["categories"] = {category_names[category]: {"raw_id": category, "questions": len(group),
            "accuracy": statistics.mean(r["score"] for r in group)}
            for category in sorted(set(raw_categories.values()))
            if (group := [r for r in rows if r["arm"] == arm and
                          raw_categories[(r["conversation_id"],r["qa_id"])] == category])}
    for baseline, comparison in summary["paired_differences"].items():
        pairs = [(r, by_row[(baseline, r["conversation_id"], r["qa_id"])]) for r in rows
                 if r["arm"] == "C11" and (baseline, r["conversation_id"], r["qa_id"]) in by_row]
        comparison["wins"] = sum(a["score"] > b["score"] for a, b in pairs)
        comparison["losses"] = sum(a["score"] < b["score"] for a, b in pairs)
        comparison["ties"] = sum(a["score"] == b["score"] for a, b in pairs)
        deltas = [question_details[("C11", a["conversation_id"], a["qa_id"])]["source_id_recall"] -
                  question_details[(baseline, a["conversation_id"], a["qa_id"])]["source_id_recall"]
                  for a, _ in pairs if ("C11", a["conversation_id"], a["qa_id"]) in question_details
                  and (baseline, a["conversation_id"], a["qa_id"]) in question_details]
        comparison["source_id_recall_delta"] = statistics.mean(deltas) if deltas else None
    novel, interactions = [], []
    for key, sources in source_sets.items():
        arm, cid, qid = key
        if arm != "C11" or not all((a, cid, qid) in source_sets for a in ("C00", "C10", "C01")):
            continue
        singles = [source_sets[(a, cid, qid)] for a in ("C10", "C01")]
        novel.append({"conversation_id": cid, "qa_id": qid,
            "candidate_gold_unique_to_dual": sorted(sources["gold"] & (sources["candidate"] - singles[0]["candidate"] - singles[1]["candidate"])),
            "final_gold_unique_to_dual": sorted(sources["gold"] & (sources["final"] - singles[0]["final"] - singles[1]["final"])),
            "same_ordered_verification_pool_as_C10": pools[key] == pools[("C10",cid,qid)],
            "verification_pool_overlap_with_C10": len(set(pools[key]) & set(pools[("C10",cid,qid)]))/max(1,len(set(pools[key]))),
            "accuracy_C11_minus_C10": by_row[key]["score"]-by_row[("C10",cid,qid)]["score"]})
        interactions.append(by_row[key]["score"] - by_row[("C10",cid,qid)]["score"] -
                            by_row[("C01",cid,qid)]["score"] + by_row[("C00",cid,qid)]["score"])
    summary["dual_expansion_audit"] = {"per_question": novel,
        "accuracy_factorial_interaction": statistics.mean(interactions) if interactions else None,
        "note": "Unique-source comparisons are observational; a dual-derived path alone does not prove incremental benefit."}
    summary["question_pipeline"] = list(question_details.values())
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--prices", default="memory_connection/benchmark-prices-usd.json")
    args = parser.parse_args()
    root = Path(args.run)
    summary = analyze(root, json.loads(Path(args.prices).read_text(encoding="utf-8")))
    (root/"analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
