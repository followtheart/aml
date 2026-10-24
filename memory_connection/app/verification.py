import asyncio
import json
import time
from collections import Counter
from .schemas import Decision
from .event_builder import original_packet, valid_spans, resolve_spans


def byte_size(value):
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


async def verify_candidates(provider, query, candidates, settings, deadline):
    batches, current, decisions, errors = [], [], {}, []
    measure = getattr(provider, "verification_size", lambda q, es: byte_size({"query": q, "events": es}))
    for candidate in candidates[:settings.verify_limit]:
        event = candidate["event"]
        packet = original_packet(event)
        if measure(query, [packet]) > settings.batch_bytes:
            errors.append({"event_id": event["event_id"], "reason": "context_overflow"})
            continue
        if current and (len(current) >= settings.batch_size or
                        measure(query, current + [packet]) > settings.batch_bytes):
            batches.append(current)
            current = []
        current.append(packet)
    if current:
        batches.append(current)
    for batch in batches[settings.max_batches:]:
        errors.extend({"event_id": e["event_id"], "reason": "batch_budget"} for e in batch)
    batches = batches[:settings.max_batches]
    events = {c["event"]["event_id"]: c["event"] for c in candidates}
    calls = 0
    for batch in batches:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            errors.extend({"event_id": e["event_id"], "reason": "verification_deadline"} for e in batch)
            continue
        calls += 1
        expected = {e["event_id"] for e in batch}
        try:
            raw = await asyncio.wait_for(provider.verify(query, batch), timeout=remaining)
            rows = raw.get("decisions") if isinstance(raw, dict) else None
            if not isinstance(rows, list) or len(rows) > 2*settings.batch_size:
                raise ValueError("invalid decision envelope")
        except Exception as exc:
            errors.extend({"event_id": eid, "reason": "verifier_failed", "error_type": type(exc).__name__} for eid in expected)
            continue
        ids = [r.get("event_id") for r in rows if isinstance(r, dict) and isinstance(r.get("event_id"), str)]
        counts = Counter(ids)
        for eid in expected:
            if counts[eid] != 1:
                errors.append({"event_id": eid, "reason": "missing_decision" if not counts[eid] else "duplicate_decision"})
        if set(ids)-expected or len(ids) != len(rows):
            errors.append({"reason": "unknown_verifier_identity"})
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("event_id"), str):
                continue
            eid = row["event_id"]
            if eid not in expected or counts[eid] != 1:
                continue
            try:
                decision = Decision.model_validate(row)
            except ValueError:
                errors.append({"event_id": eid, "reason": "invalid_decision"})
                continue
            useful = decision.verdict in {"relevant", "partial", "contradicting"}
            resolved = resolve_spans(decision.evidence_spans, events[decision.event_id])
            if resolved:
                decision = decision.model_copy(update={"evidence_spans": resolved})
            if (useful or decision.evidence_spans) and not valid_spans(decision.evidence_spans, events[decision.event_id]):
                errors.append({"event_id": decision.event_id, "reason": "invalid_evidence"})
                continue
            if decision.verdict == "uncertain":
                errors.append({"event_id": decision.event_id, "reason": "uncertain"})
            decisions[decision.event_id] = decision.model_dump()
    return decisions, errors, calls
