from .verification import byte_size
from .temporal import source_annotations


def pack(candidates, decisions, top_k, budget):
    useful = [c for c in candidates if decisions.get(c["event"]["event_id"], {}).get("verdict") in
              {"relevant", "partial", "contradicting"}]
    # Preserve evidence ranking/diversity. A useful partial premise must not be
    # automatically crowded out by individually complete but redundant facts.
    # Reserve a counterexample if present; it cannot be silently crowded out.
    counter = next((c for c in useful if decisions[c["event"]["event_id"]]["verdict"] == "contradicting"), None)
    if counter and counter not in useful[:top_k]:
        useful.remove(counter)
        useful.insert(max(0, top_k-1), counter)
    data, omitted, used = [], [], 0
    for candidate in useful:
        event = candidate["event"]
        decision = decisions[event["event_id"]]
        item = {"id": event["event_id"], "content": event["text"], "sources": event["sources"],
                "source_annotations": source_annotations(event["sources"]),
                "score": candidate["score"], "score_kind": "original_evidence_rank_signal" if candidate.get("semantic_score") is not None else "coarse_rank_signal",
                "verdict": decision["verdict"], "relation_type": decision["relation_type"],
                "evidence_spans": decision["evidence_spans"], "version": event["version"],
                "valid_from": event["valid_from"], "valid_to": event["valid_to"]}
        if event.get("memory_type"):
            item["memory_type"] = event["memory_type"]
            item["experience_status"] = event.get("experience_status")
        size = byte_size(item)
        if len(data) >= top_k or used + size > budget:
            omitted.append({"event_id": event["event_id"], "reason": "top_k" if len(data) >= top_k else "evidence_bytes"})
            continue
        data.append(item)
        used += size
    return data, omitted, bool(counter)
