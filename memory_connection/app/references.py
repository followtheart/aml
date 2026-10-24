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


def _compact_refs(row, refs, event):
    """Coalesce excess adjacent exact references, never truncate evidence.

    Some providers copy every sentence ref despite maxItems=8. Accept that
    representation only if ALL refs belong to this event and can be expressed
    in at most eight verbatim spans. Other schema errors still fail closed.
    """
    values = row.get('evidence_refs')
    if event is None or not isinstance(values, list) or len(values) <= 8 or len(values) > 256:
        return None
    parsed = ReferenceDecision.model_validate({**row, 'evidence_refs': []})
    sources = {s['source_id']: s['content'] for s in event['sources']}
    spans = []
    for ref in values:
        if not isinstance(ref, str) or ref not in refs:
            raise ValueError('unknown evidence reference')
        span = refs[ref]
        if span['source_id'] not in sources:
            raise ValueError('reference outside this event')
        content = sources[span['source_id']]
        if content[span['start']:span['end']] != span['quote']:
            raise ValueError('reference is not verbatim')
        spans.append(dict(span))
    merged = []
    for span in sorted(spans, key=lambda s: (s['source_id'], s['start'], s['end'])):
        previous = merged[-1] if merged else None
        if previous and previous['source_id'] == span['source_id']:
            content = sources[span['source_id']]
            gap = content[previous['end']:span['start']]
            if not gap.strip():
                previous['end'] = max(previous['end'], span['end'])
                previous['quote'] = content[previous['start']:previous['end']]
                continue
        merged.append(span)
    if len(merged) > 8:
        raise ValueError('too many nonadjacent evidence spans')
    return parsed, merged


def decode_references(raw, aliases, refs, originals=()):
    if not isinstance(raw, dict) or not isinstance(raw.get("decisions"), list):
        raise ValueError("missing decisions")
    result = []
    events = {e['event_id']: e for e in originals}
    for row in raw["decisions"]:
        eid = aliases.get(row.get("event_id")) if isinstance(row, dict) else None
        if eid is None:
            result.append({"event_id": "__unknown__"})
            continue
        try:
            compacted = _compact_refs(row, refs, events.get(eid))
            if compacted is not None:
                parsed, spans = compacted
            else:
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
