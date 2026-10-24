"""Lossless message events; bounded preceding context within an Add request."""
import uuid
from datetime import datetime, timezone


def build_events(req):
    events, sources, seen = [], [], set()
    for pos, message in enumerate(req.messages):
        logical_id = message.message_id or f"{req.request_id}:{pos}"
        if logical_id in seen:
            raise ValueError("duplicate message_id in Add")
        seen.add(logical_id)
        source = {**message.model_dump(), "source_id": uuid.uuid4().hex,
                  "session_id": req.session_id, "message_id": logical_id}
        sources.append(source)
        # Context is preserved as separate attributed sources, never flattened into facts.
        context = sources[max(0, pos - 2):pos + 1]
        event_id = uuid.uuid4().hex
        events.append({"event_id": event_id, "logical_id": logical_id,
                       "session_id": req.session_id, "sources": context,
                       "primary_source_id": source["source_id"], "text": message.content,
                       "valid_from": getattr(message, "valid_from", None) or datetime.now(timezone.utc).isoformat(),
                       "valid_to": None, "version": 1})
    return events


def original_packet(event):
    return {key: event[key] for key in ("event_id", "sources", "primary_source_id",
                                        "valid_from", "valid_to", "version")}


def valid_spans(spans, event):
    sources = {s["source_id"]: s["content"] for s in event["sources"]}
    return bool(spans) and all(
        s.source_id in sources and 0 <= s.start < s.end <= len(sources[s.source_id])
        and sources[s.source_id][s.start:s.end] == s.quote for s in spans)


def resolve_spans(spans, event):
    """Locate an exact unique quote when the model miscounts character offsets.

    Never fuzzy match, substitute a source, or disambiguate repeated quotations.
    """
    sources = {s["source_id"]: s["content"] for s in event["sources"]}
    resolved = []
    for span in spans:
        content = sources.get(span.source_id, "")
        if valid_spans([span], event):
            resolved.append(span)
        elif content.count(span.quote) == 1:
            start = content.index(span.quote)
            resolved.append(span.model_copy(update={"start": start, "end": start + len(span.quote)}))
        else:
            return None
    return resolved
