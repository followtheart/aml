"""Compact evidence formatting and a shared budget for answer memories."""
from datetime import datetime, timezone
import logging

from . import config, evidence_packet

# Model-facing separator between a memory and its quoted source messages.
SOURCE_MARKER = "\nSource messages:\n"
# Earlier packets used this marker; persisted responses are still split on it.
LEGACY_SOURCE_MARKER = "\n[source evidence; quoted data]\n"
_ROLE_LABELS = {"user": "User", "assistant": "Assistant", "tool": "Tool", "persona": "Persona"}
log = logging.getLogger("aml.context")


def readable_time(value):
    """Spell Unix-millisecond or ISO-8601 instants as 'D Month YYYY HH:MM UTC'.

    The platform answer model reads content only; raw epoch milliseconds or
    ISO strings cost it a conversion step that small models get wrong.
    """
    if value is None or value == "":
        return None
    try:
        if isinstance(value, bool):
            raise TypeError
        if isinstance(value, (int, float)):
            instant = datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)
        else:
            text = str(value).strip()
            if text.isdigit() and len(text) >= 10:
                instant = datetime.fromtimestamp(int(text) / 1000, tz=timezone.utc)
            elif len(text) == 4 and text.isdigit():
                return text  # year precision
            elif len(text) == 7 and text[4] == "-" and text[:4].isdigit() and text[5:].isdigit():
                return datetime(int(text[:4]), int(text[5:]), 1).strftime("%B %Y")  # month precision
            else:
                instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError, OSError):
        return str(value)
    if instant.tzinfo is not None:
        instant = instant.astimezone(timezone.utc)
    if (instant.hour, instant.minute) == (0, 0):
        return f"{instant.day} {instant.strftime('%B %Y')}"
    return f"{instant.day} {instant.strftime('%B %Y')} {instant.strftime('%H:%M')} UTC"


def _split_body(text):
    for marker in (SOURCE_MARKER, LEGACY_SOURCE_MARKER):
        if marker in text:
            return text.split(marker, 1)[0]
    return text


def with_evidence(body, sources):
    # Keep full provenance in SearchItem.sources, not in model-facing prose.
    quotes = []
    for source in sources:
        if not source.get("content"):
            continue
        role = source.get("role") or "unknown"
        label = _ROLE_LABELS.get(role, str(role))
        when = readable_time(source.get("timestamp"))
        header = f"[{when}, {label}]" if when else f"[{label}]"
        quotes.append(f"{header} {source['content']}")
    return body + (SOURCE_MARKER + "\n".join(quotes) if quotes else "")


def _clip(text, limit):
    marker = "\n[truncated]"
    if len(text) <= limit:
        return text
    if limit <= len(marker):
        return marker[:limit]
    return text[:limit - len(marker)] + marker


def _source_keys(memory):
    return {(s.get("request_id"), s.get("message_index")) for s in memory.get("sources") or []}


def _verified_display(memory):
    """Render an authenticated legacy source suffix using the current format.

    Match the whole suffix, never reconstruct hidden sources or split on marker
    text inside a quote. The transport packet and its digest remain unchanged.
    """
    text = memory['content']
    sources = memory.get('sources') or []
    quotes = [f"[timestamp: {s.get('timestamp')}; role: {s.get('role')}] {s['content']}"
              for s in sources if s.get('content')]
    if quotes:
        suffix = LEGACY_SOURCE_MARKER + '\n'.join(quotes)
        if text.endswith(suffix):
            return with_evidence(text[:-len(suffix)], sources)
    return text


def build(memories):
    """Preserve retrieval order; cap every item and the entire memory context.

    Character budgets deliberately avoid a guessed provider tokenizer. They
    include evidence, metadata prefixes, separators and truncation markers.
    Task instructions/questions are assembled separately and never truncated.

    An episode whose source messages are all already covered by higher-ranked
    memories is dropped: it would only restate the same event a second time
    and bias the answer model toward whatever is repeated most often.
    """
    hashes = {m.get('packet_hash') for m in memories}
    if hashes - {None}:
        if len(hashes) != 1 or evidence_packet.digest(memories) not in hashes:
            raise ValueError('Evidence packet was modified after verification')
        return '\n'.join(_verified_display(m) for m in memories) or '(no memories)'
    remaining = config.ANSWER_CONTEXT_MAX_CHARS
    blocks = []
    seen = set()
    seen_sources = set()
    dropped_episodes = 0
    original_chars = sum(len(m["content"]) for m in memories)
    for memory in memories:
        text = memory["content"]
        keys = _source_keys(memory)
        if memory.get("memory_type") == "episode" and keys and keys <= seen_sources:
            dropped_episodes += 1
            continue
        if "sources" in memory:
            # Also handles persisted responses with the old verbose JSON suffix.
            text = with_evidence(_split_body(text), memory["sources"])
        if not text or text in seen:
            continue
        seen.add(text)
        separator = 1 if blocks else 0
        available = remaining - separator
        if available <= 0:
            break
        block = _clip(text, min(config.ANSWER_CONTEXT_ITEM_MAX_CHARS, available))
        blocks.append(block)
        seen_sources |= keys
        remaining -= len(block) + separator
    result = "\n".join(blocks) or "(no memories)"
    log.info("answer_context input_items=%d included_items=%d dropped_episodes=%d "
             "input_chars=%d context_chars=%d budget_chars=%d", len(memories), len(blocks),
             dropped_episodes, original_chars, len(result), config.ANSWER_CONTEXT_MAX_CHARS)
    return result
