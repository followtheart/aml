"""Compact evidence formatting and a shared budget for answer memories."""
import logging

from . import config, evidence_packet

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


def _source_keys(memory):
    return {(s.get("request_id"), s.get("message_index")) for s in memory.get("sources") or []}


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
        return '\n'.join(m['content'] for m in memories) or '(no memories)'
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
        seen_sources |= keys
        remaining -= len(block) + separator
    result = "\n".join(blocks) or "(no memories)"
    log.info("answer_context input_items=%d included_items=%d dropped_episodes=%d "
             "input_chars=%d context_chars=%d budget_chars=%d", len(memories), len(blocks),
             dropped_episodes, original_chars, len(result), config.ANSWER_CONTEXT_MAX_CHARS)
    return result
