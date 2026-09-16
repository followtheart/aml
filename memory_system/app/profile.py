"""Persona layer (IMPROVEMENT_PLAN P0/P1).

Two Add-side duties plus shared Core Profile helpers:

- consolidate(): MemoryBank/EverMemOS-style interest consolidation. Repeated
  "the user asked/discussed X" signals are promoted into first-person
  `preference` AMUs ("The user bakes bread at home") instead of remaining
  scattered topic facts. Runs once per Add, after facts are persisted.
- apply_forget_rules(): Mem0 DELETE / Zep edge invalidation. A user
  "forget X" request closes and suppresses the matching memories instead of
  only being logged as a fact (the request itself stays as an auditable rule).

Everything here is additive: failures are logged and never fail the Add.
"""
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from . import config, llm, prompts, store
from .embeddings import embed

log = logging.getLogger("aml.profile")

_MERGE_SIM = 0.80

FORGET_RE = re.compile(
    r"\b(forget(?:ting)?\s+(?:that|this|about|it)\b|do(?:es)? not remember\b|"
    r"don'?t remember\b|erase\b|stop (?:remembering|mentioning)\b|"
    r"delete (?:that|this|the)\b|remove\b.{0,30}\bfrom (?:your )?memory\b)"
    r"|忘记|忘掉|删除.{0,6}记忆|别记|不要记住", re.I)

_PERSONA_TYPES = ("preference", "profile", "fact", "event")


def _one_line(text: str) -> str:
    return " / ".join(part.strip() for part in str(text).splitlines() if part.strip())


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
            keys = [req.session_id, req.request_id] + support
            if neighbors and neighbors[0]["_score"] >= _MERGE_SIM:
                # A-Mem-style evolution: reinforce the existing trait instead
                # of storing a near-duplicate alongside it.
                st.add_support_keys(neighbors[0]["id"], keys)
                log.info("profile reinforced target=%s support=%d",
                         neighbors[0]["id"], len(keys))
                continue
            evidence = [e for s in support for e in (by_id[s].get("evidence") or [])]
            amu_id = st.insert_amu(
                user_id=req.user_id, session_id=req.session_id, content=content,
                retrieval_key=content[:200], type="preference",
                valid_from=_ref_time(req), confidence=0.8, embedding=vec,
                evidence=evidence)
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


def _ref_time(req) -> str:
    ts = next((m.timestamp for m in reversed(req.messages) if m.timestamp), None)
    if ts:
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


async def apply_forget_rules(st: store.Store, req,
                             persisted: List[Tuple[Optional[str], Dict]]):
    """Invalidate memories targeted by an explicit user forget request."""
    if not config.INVALIDATE_ENABLED:
        return
    for aid, fact in persisted:
        if fact.get("type") == "episode":
            continue
        content = fact.get("content") or ""
        if not FORGET_RE.search(content):
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
