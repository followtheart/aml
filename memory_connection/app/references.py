"""Server-owned exact source references; model never computes character offsets."""
import re
from typing import Literal
from pydantic import Field
from .schemas import Strict, Decision
from .temporal import source_time


class ReferenceDecision(Strict):
    event_id: str
    verdict: Literal["relevant", "partial", "contradicting", "irrelevant", "uncertain"]
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)
    reason: str = Field(default="", max_length=800)


class ReferenceVerdicts(Strict):
    decisions: list[ReferenceDecision] = Field(max_length=8)


def reference_packet(events):
    sources, refs, aliases, source_aliases = [], {}, {}, {}
    packets = []
    for index, event in enumerate(events):
        alias = f"e{index}"
        aliases[alias] = event["event_id"]
        for source in event["sources"]:
            sid = source["source_id"]
            if sid in source_aliases:
                continue
            source_aliases[sid] = f"s{len(source_aliases)}"
            units = []
            # Keep the entire source, divided only at sentence/line boundaries.
            for match in re.finditer(r"\S.*?(?:[.!?](?=\s|$)|\n|$)", source["content"], re.S):
                start, end = match.span()
                while end > start and source["content"][end-1].isspace():
                    end -= 1
                quote = source["content"][start:end]
                if not quote:
                    continue
                ref = f"r{len(refs)}"
                refs[ref] = {"source_id": sid, "start": start, "end": end, "quote": quote}
                units.append({"ref": ref, "text": quote})
            sources.append({"source_id": source_aliases[sid], "speaker": source.get("speaker_id"),
                            "role": source.get("role"), **source_time(source), "units": units})
        event_ids = {s["source_id"] for s in event["sources"]}
        local_aliases = {source_aliases[sid] for sid in event_ids}
        packets.append({"event_id": alias, "primary_source": source_aliases[event["primary_source_id"]],
                        "source_ids": [source_aliases[s["source_id"]] for s in event["sources"]],
                        "sources": [s for s in sources if s["source_id"] in local_aliases],
                        "allowed_evidence_refs": [ref for ref,span in refs.items() if span["source_id"] in event_ids],
                        "valid_from": event["valid_from"], "valid_to": event["valid_to"]})
    return {"events": packets}, aliases, refs


def decode_references(raw, aliases, refs):
    if not isinstance(raw, dict) or not isinstance(raw.get("decisions"), list):
        raise ValueError("missing decisions")
    result = []
    for row in raw["decisions"]:
        eid = aliases.get(row.get("event_id")) if isinstance(row, dict) else None
        if eid is None:
            result.append({"event_id": "__unknown__"})
            continue
        try:
            parsed = ReferenceDecision.model_validate(row)
            spans = [refs[ref] for ref in parsed.evidence_refs]
            decision = Decision(event_id=eid, verdict=parsed.verdict, relation_type="other",
                subject_match="unknown", time_match="unknown", polarity="unknown",
                evidence_spans=spans, reason=parsed.reason)
            result.append(decision.model_dump())
        except (ValueError, KeyError):
            # Keep identity so the verifier can report one failed item, not drop its neighbors.
            result.append({"event_id": eid})
    return {"decisions": result}
