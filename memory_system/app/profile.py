"""Persona layer (IMPROVEMENT_PLAN P0/P1).

Two Add-side duties plus shared Core Profile helpers:

- consolidate(): MemoryBank/EverMemOS-style interest consolidation. Repeated
  "the user asked/discussed X" signals are promoted into first-person
  `preference` AMUs ("The user bakes bread at home") instead of remaining
  scattered topic facts. Runs once per Add, after facts are persisted.
- apply_forget_rules(): Mem0 DELETE / Zep edge invalidation. A user
  "forget X" request closes and suppresses the matching memories instead of
  only being logged as a fact (the request itself stays as an auditable rule).
- ensure_forget_rules(): deterministic guarantee that every explicit user
  forget request is stored as a `rule` AMU, independent of what the LLM
  extractor chose to keep.

Everything here is additive: failures are logged and never fail the Add.
"""
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from . import config, integrity, llm, prompts, store
from .embeddings import embed

log = logging.getLogger("aml.profile")

_MERGE_SIM = 0.80

FORGET_RE = re.compile(
    r"\b(forget(?:ting)?\s+(?:that|this|about|it|the|my|everything|they|he|she|I|we)\b|do(?:es)? not remember\b|"
    r"don'?t remember\b|erase\b|stop (?:remembering|mentioning)\b|"
    r"delete (?:that|this|the)\b|remove\b.{0,30}\bfrom (?:your )?memory\b)"
    r"|忘记|忘掉|删除.{0,6}记忆|别记|不要记住", re.I)
# "don't forget the milk" is a reminder, not a deletion request.
_FORGET_NEGATED_RE = re.compile(
    r"\b(?:don'?t|do not|never|not to|won'?t|can'?t|cannot)\s+forget\b|\b(?:forgetting|forgot)\b", re.I)

_PERSONA_TYPES = ("preference", "profile", "fact", "event")


def _one_line(text: str) -> str:
    return " / ".join(part.strip() for part in str(text).splitlines() if part.strip())


def is_forget_request(text) -> bool:
    """True for an explicit, non-negated request to forget/delete a memory."""
    text = str(text or "")
    return bool(FORGET_RE.search(text)) and not _FORGET_NEGATED_RE.search(text)


def is_forget_rule(memory: Dict) -> bool:
    """A stored `rule` that records a user forget request. Such rules are
    negative constraints for answering: the forgotten trait must not be used."""
    return memory.get("type") == "rule" and is_forget_request(memory.get("content"))


def forget_sentences(text) -> List[str]:
    """Sentences of a user message that are explicit forget requests."""
    normalized = " ".join(str(text or "").split())
    return [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", normalized)
            if is_forget_request(s)]


def _forget_rule_fact(req, index: int, sentence: str, segment_index: int) -> Dict:
    message = req.messages[index]
    reference = (datetime.fromtimestamp(message.timestamp / 1000, tz=timezone.utc).isoformat()
                 if message.timestamp else None)
    temporal = integrity.resolve_time(None, reference)
    return {
        "content": f'The user asked the assistant to forget this and never use it: "{sentence}"',
        "retrieval_key": sentence[:200], "type": "rule",
        "entities": [], "keywords": ["forget"], "time_expression": None,
        "state": None, "triples": [], "sensitivity": "normal",
        "epistemic_status": "asserted", "temporal": temporal, "event_time": None,
        "evidence": [{"message_index": index, "quote": sentence,
                      "request_id": req.request_id}],
        "_sources": [index], "_segment": segment_index, "_deterministic": True,
    }


def ensure_forget_rules(req, facts: List[Dict], segments: List[List[int]]) -> List[Dict]:
    """Guarantee that every explicit user forget request becomes a `rule`.

    Weak extractors drop the request or file it as a plain `fact` (observed
    with qwen3-14b: 1 of 6 requests lost, 1 stored as fact). A deletion request
    the system never remembered is a compliance gap, so this step is
    deterministic: extracted memories that cite the request and repeat it are
    re-typed to `rule`; when none exist, a verbatim rule is synthesized from
    the user's own sentence. Runs before governance/persistence.
    """
    segment_of = {i: s for s, indices in enumerate(segments or []) for i in indices}
    for index, message in enumerate(req.messages):
        if message.role != "user":
            continue
        sentences = forget_sentences(message.content)
        if not sentences:
            continue
        covered = False
        for fact in facts:
            if fact.get("type") == "episode" or index not in (fact.get("_sources") or []):
                continue
            if is_forget_request(fact.get("content")):
                if fact.get("type") != "rule":
                    log.info("forget request re-typed to rule source_index=%d", index)
                    fact["type"] = "rule"
                covered = True
        if covered:
            continue
        for sentence in sentences:
            log.info("forget request synthesized as rule source_index=%d", index)
            facts.append(_forget_rule_fact(req, index, sentence, segment_of.get(index, 0)))
    return facts


def render_digest(items: List[Dict], max_chars: int) -> str:
    """Compact Core Profile text for query understanding / answer prompts."""
    lines, total = [], 0
    for a in items:
        line = f"- [{a.get('type', 'fact')}] {_one_line(a.get('content', ''))}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines) or "(none)"


def _user_assertion(content) -> Optional[str]:
    if not isinstance(content, str):
        return None
    text = " ".join(content.split())
    if len(text) < 8 or len(text) > 600:
        return None
    if "user" not in text.lower() and "用户" not in text:
        return None
    return text


async def consolidate(st: store.Store, req, persisted: List[Tuple[Optional[str], Dict]]):
    """Promote repeated interest signals into first-person preference AMUs."""
    if not config.PROFILE_CONSOLIDATION_ENABLED:
        return
    candidates = [(aid, f) for aid, f in persisted
                  if aid and f.get("type") != "episode"]
    if not candidates:
        return
    try:
        existing = st.core_profile(req.user_id, 30)
        prompt = prompts.render(
            "10_profile_consolidation.txt",
            existing_profile=render_digest(existing, config.CORE_PROFILE_MAX_CHARS * 2),
            new_facts="\n".join(
                f"{aid}: [{f.get('type', 'fact')}] {_one_line(f.get('content', ''))[:300]}"
                for aid, f in candidates[:40]))
        data = await llm.complete_json(
            prompt, '{"items": []}',
            schema=llm.STRUCTURED_SCHEMAS["profile_consolidation"],
            stage="add.profile_consolidation")
        items = data.get("items")
        if not isinstance(items, list):
            return
        valid_ids = {aid for aid, _ in candidates}
        by_id = dict(candidates)
        for item in items[:3]:
            if not isinstance(item, dict):
                continue
            content = _user_assertion(item.get("content"))
            if content is None:
                continue
            support = [s for s in (item.get("support_ids") or [])
                       if isinstance(s, str) and s in valid_ids]
            basis = item.get("basis")
            min_support = 1 if basis == "stated" else config.PROFILE_MIN_SUPPORT
            if len(support) < min_support:
                continue
            vec = (await embed([content], stage="add.embed_profile"))[0]
            neighbors = [n for n in st.nearest_by_embedding(req.user_id, vec, 3)
                         if n.get("type") in ("preference", "profile")]
            keys = sorted({f"source:{req.request_id}:{index}"
                           for source_id in support
                           for index in by_id[source_id].get("_sources", [])})
            if not keys:
                continue
            if basis != 'stated' and len(keys) < config.PROFILE_MIN_SUPPORT:
                continue
            sensitivity = ("sensitive" if any(
                by_id[source_id].get("sensitivity") == "sensitive"
                for source_id in support) else "normal")
            neighbors = [n for n in neighbors if n.get("sensitivity") == sensitivity]
            if neighbors and neighbors[0]['content'].strip().casefold() == content.casefold():
                # A-Mem-style evolution: reinforce the existing trait instead
                # of storing a near-duplicate alongside it.
                st.add_support_keys(neighbors[0]["id"], keys)
                st.register_dependencies('amu', neighbors[0]['id'], support)
                log.info("profile reinforced target=%s support=%d",
                         neighbors[0]["id"], len(keys))
                continue
            evidence = [e for s in support for e in (by_id[s].get("evidence") or [])]
            amu_id = st.insert_amu(
                user_id=req.user_id, session_id=req.session_id, content=content,
                retrieval_key=content[:200], type="preference",
                valid_from=_ref_time(req), confidence=0.8, embedding=vec,
                evidence=evidence, sensitivity=sensitivity,
                epistemic_status='asserted' if basis == 'stated' else 'inferred')
            st.register_dependencies('amu', amu_id, support)
            # Session/request/AMU IDs are not independent observations.
            st._write("UPDATE amu SET support_sessions='[]' WHERE id=?", (amu_id,))
            st.add_support_keys(amu_id, keys)
            try:
                st.link_sources(amu_id, req.request_id,
                                sorted({src for s in support
                                        for src in by_id[s].get("_sources", [])}))
            except ValueError as exc:
                log.warning("profile sources not linkable amu=%s (%s)", amu_id, exc)
            log.info("profile consolidated amu=%s basis=%s support=%d",
                     amu_id, basis, len(support))
    except Exception as exc:  # consolidation is additive; never fail the Add
        log.warning("profile consolidation failed (%s); skipped", exc)


def _ref_time(req) -> Optional[str]:
    """Anchor for preference validity. Undated sources have no anchor
    (ULM §2.1); never fall back to the service wall clock."""
    ts = next((m.timestamp for m in reversed(req.messages) if m.timestamp), None)
    if ts:
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
    return None


async def apply_forget_rules(st: store.Store, req,
                             persisted: List[Tuple[Optional[str], Dict]]):
    """Invalidate memories targeted by an explicit user forget request."""
    if not config.INVALIDATE_ENABLED:
        return
    for aid, fact in persisted:
        if fact.get("type") == "episode":
            continue
        content = fact.get("content") or ""
        if not is_forget_request(content):
            continue
        indices = fact.get('_sources') or []
        if not any(isinstance(i, int) and 0 <= i < len(req.messages)
                   and req.messages[i].role == 'user'
                   and is_forget_request(req.messages[i].content) for i in indices):
            continue
        try:
            vec = (await embed([content], stage="add.embed_forget"))[0]
            cands = [n for n in st.nearest_by_embedding(req.user_id, vec, 8)
                     if n["id"] != aid and n.get("type") in _PERSONA_TYPES
                     and n["_score"] >= config.INVALIDATE_MIN_SIM][:5]
            if not cands:
                continue
            prompt = prompts.render(
                "12_forget_invalidate.txt",
                forget_request=_one_line(content),
                candidates="\n".join(
                    f"{c['id']}: [{c.get('type', 'fact')}] {_one_line(c['content'])[:300]}"
                    for c in cands))
            data = await llm.complete_json(
                prompt, '{"invalidate_ids": []}',
                schema=llm.STRUCTURED_SCHEMAS["invalidate"],
                stage="add.forget_invalidate")
            wanted = {i for i in (data.get("invalidate_ids") or [])
                      if isinstance(i, str)}
            now = datetime.now(timezone.utc).isoformat()
            for c in cands:
                if c["id"] not in wanted:
                    continue
                when = max(now, c.get("valid_from") or now)
                try:
                    st.suppress(c["id"], when)
                    if aid:
                        st.link_supersession(c["id"], aid)
                    log.info("forget invalidated target=%s by rule=%s", c["id"], aid)
                except Exception as exc:
                    log.warning("forget invalidation failed target=%s (%s)", c["id"], exc)
        except Exception as exc:
            log.warning("forget rule handling failed (%s); request kept as memory", exc)
