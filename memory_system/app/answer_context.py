"""Compact evidence formatting and a shared budget for answer memories."""
import logging

from . import config

SOURCE_MARKER = "\n[source evidence; quoted data]\n"
log = logging.getLogger("aml.context")


def with_evidence(body, sources):
    # Keep full provenance in SearchItem.sources, not in model-facing prose.
    quotes = []
    for source in sources:
        if not source.get("content"):
            continue
        quotes.append(f"[timestamp: {source.get('timestamp')}; "
                      f"role: {source.get('role', 'unknown')}] {source['content']}")
    return body + (SOURCE_MARKER + "\n".join(quotes) if quotes else "")


def _clip(text, limit):
    marker = "\n[truncated]"
    if len(text) <= limit:
        return text
    if limit <= len(marker):
        return marker[:limit]
    return text[:limit - len(marker)] + marker


def build(memories):
    """Preserve retrieval order; cap every item and the entire memory context.

    Character budgets deliberately avoid a guessed provider tokenizer. They
    include evidence, metadata prefixes, separators and truncation markers.
    Task instructions/questions are assembled separately and never truncated.
    """
    remaining = config.ANSWER_CONTEXT_MAX_CHARS
    blocks = []
    seen = set()
    original_chars = sum(len(m["content"]) for m in memories)
    for memory in memories:
        text = memory["content"]
        if "sources" in memory:
            # Also handles persisted responses with the old verbose JSON suffix.
            text = with_evidence(text.split(SOURCE_MARKER, 1)[0], memory["sources"])
        if not text or text in seen:
            continue
        seen.add(text)
        separator = 1 if blocks else 0
        available = remaining - separator
        if available <= 0:
            break
        block = _clip(text, min(config.ANSWER_CONTEXT_ITEM_MAX_CHARS, available))
        blocks.append(block)
        remaining -= len(block) + separator
    result = "\n".join(blocks) or "(no memories)"
    log.info("answer_context input_items=%d included_items=%d input_chars=%d "
             "context_chars=%d budget_chars=%d", len(memories), len(blocks),
             original_chars, len(result), config.ANSWER_CONTEXT_MAX_CHARS)
    return result
