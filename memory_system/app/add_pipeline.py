"""Add pipeline (design doc v0.2 §4):
normalize -> rolling summary -> AMU extraction -> item-level governance ->
multi-index write. Synchronous from the caller's point of view.
"""
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from . import config, llm, memory_debug, prompts, schemas, store
from .embeddings import embed

log = logging.getLogger("aml.add")


def _format_messages(msgs: List[schemas.Message]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in msgs)


def _ref_time(msgs: List[schemas.Message]) -> str:
    ts = next((m.timestamp for m in reversed(msgs) if m.timestamp), None)
    if ts:
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _embed_text(fact: Dict) -> str:
    parts = [fact.get("content", ""), fact.get("retrieval_key", ""),
             " ".join(fact.get("keywords") or []),
             " ".join(fact.get("entities") or [])]
    return " ".join(p for p in parts if p)


def _is_grounded_fact(fact: Dict, source: str) -> bool:
    """Reject empty/schema-placeholder facts emitted by weak small models."""
    if not isinstance(fact, dict):
        return False
    content = str(fact.get("content") or "").strip()
    if len(content) < 4:
        return False
    source_lower = source.lower()
    tokens = [t for t in re.findall(r"\w+", content.lower()) if len(t) >= 3]
    if any(t in source_lower for t in tokens):
        return True
    # For CJK text, require at least one source-grounded character bigram.
    cjk = "".join(re.findall(r"[\u3400-\u9fff]", content))
    return any(cjk[i:i + 2] in source for i in range(len(cjk) - 1))


async def _extract(st: store.Store, req: schemas.AddRequest) -> Dict:
    summary = st.get_summary(req.user_id, req.session_id)
    batch_size = config.EXTRACT_BATCH_MESSAGES
    facts, triples = [], []

    for start in range(0, len(req.messages), batch_size):
        batch = req.messages[start:start + batch_size]
        prior = req.messages[max(0, start - 4):start]
        prompt = prompts.render(
            "01_extract_amu.txt",
            session_summary=summary or "(none yet)",
            recent_messages=_format_messages(prior) or "(none)",
            chunk_messages="\n".join(f"[{i}] {m.role}: {m.content}"
                                     for i, m in enumerate(batch)),
            reference_time=_ref_time(batch),
        )
        try:
            data = await llm.complete_json(
                prompt,
                '{"facts":[{"source_message_indices":[0],"content":"...","retrieval_key":"...",'
                '"type":"fact","entities":[],"keywords":[],'
                '"event_time":null,"sensitivity":"normal"}],'
                '"triples":[{"fact_index":0,"subject":"...","relation":"...",'
                '"object":"..."}]}',
                schema=llm.STRUCTURED_SCHEMAS["extraction"],
                stage=f"add.extract.batch_{start // batch_size + 1}")
            if not isinstance(data, dict):
                raise ValueError("extraction result is not a JSON object")
            raw_facts = data.get("facts") or []
            grounded = []
            index_map = {}
            for i, raw in enumerate(raw_facts):
                if not _is_grounded_fact(raw, _format_messages(batch)):
                    continue
                fact = dict(raw)
                indices = fact.get("source_message_indices")
                if not isinstance(indices, list) or not indices or any(
                        type(j) is not int or not 0 <= j < len(batch) for j in indices):
                    # Legacy model output: retain honest batch-level provenance.
                    indices = list(range(len(batch)))
                fact["_sources"] = [start + j for j in indices]
                index_map[i] = len(facts) + len(grounded)
                grounded.append(fact)
            facts.extend(grounded)
            for t in data.get("triples") or []:
                if not isinstance(t, dict):
                    continue
                index = t.get("fact_index")
                if type(index) is int and index in index_map and all(
                        isinstance(t.get(k), str) and t[k].strip()
                        for k in ("subject", "relation", "object")):
                    triples.append(dict(t, fact_index=index_map[index]))
            if raw_facts and not grounded:
                log.warning(
                    "extraction batch %d-%d returned no grounded facts; "
                    "storing that batch as an episode",
                    start, start + len(batch) - 1)
                facts.append({
                    "content": _format_messages(batch),
                    "retrieval_key": "Conversation episode",
                    "type": "episode",
                    "_sources": list(range(start, start + len(batch))),
                    "entities": [],
                    "keywords": [],
                    "event_time": None,
                    "sensitivity": "normal",
                })
        except Exception as e:
            log.warning(
                "extraction batch %d-%d failed (%s); storing that batch as "
                "an episode", start, start + len(batch) - 1, e)
            facts.append({
                "content": _format_messages(batch),
                "retrieval_key": "Conversation episode",
                "type": "episode",
                "_sources": list(range(start, start + len(batch))),
                "entities": [],
                "keywords": [],
                "event_time": None,
                "sensitivity": "normal",
            })

    return {"facts": facts, "triples": triples}


async def _govern_one(st: store.Store, user_id: str, fact: Dict,
                      vec) -> str:
    """Item-level governance: ADD / UPDATE / SUPERSEDE / NOOP."""
    neighbors = st.nearest_by_embedding(user_id, vec,
                                        config.GOVERNANCE_NEIGHBORS)
    entity_neighbors = st.get_by_entities(
        user_id, fact.get("entities") or [], config.GOVERNANCE_NEIGHBORS)
    by_id = {item["id"]: item for item in neighbors}
    by_id.update({item["id"]: item for item in entity_neighbors})
    # restrict to confident near-duplicates / same-entity items
    entity_ids = {item["id"] for item in entity_neighbors}
    cand = [n for n in by_id.values()
            if n["id"] in entity_ids or n["_score"] > 0.55]
    if not cand:
        op = {"operation": "ADD", "target_id": None, "merged_content": None}
    else:
        cand_text = "\n".join(f"{n['id']}: {n['content']}" for n in cand)
        prompt = prompts.render(
            "02_governance_decision.txt",
            new_memory=fact["content"],
            neighbor_memories=cand_text)
        try:
            op = await llm.complete_json(
                prompt,
                '{"operation":"ADD|UPDATE|SUPERSEDE|NOOP","target_id":null,'
                '"merged_content":null,"reason":"..."}',
                schema=llm.STRUCTURED_SCHEMAS["governance"],
                stage="add.governance")
        except Exception:
            op = {"operation": "ADD", "target_id": None,
                  "merged_content": None}
    return op.get("operation", "ADD"), op


async def _persist_fact(st: store.Store, req: schemas.AddRequest,
                        fact: Dict, vec) -> Optional[str]:
    op, detail = await _govern_one(st, req.user_id, fact, vec)
    now = datetime.now(timezone.utc).isoformat()
    target = detail.get("target_id")
    if op in ("UPDATE", "SUPERSEDE", "NOOP") and target:
        candidates = st.get_amus_by_ids([target])
        if not candidates or candidates[0]["user_id"] != req.user_id:
            raise ValueError("Governance target is not an active memory of this user")
    if op == "NOOP":
        if target:
            st.link_sources(target, req.request_id, fact.get("_sources", []))
        return None
    if op == "UPDATE" and target:
        final = dict(fact, content=detail.get("merged_content") or fact["content"])
        # Metadata and graph must describe the merged body, not just the incoming fact.
        refresh_req = schemas.AddRequest(
            request_id=req.request_id, user_id=req.user_id, session_id=req.session_id,
            messages=[schemas.Message(role="user", content=final["content"])])
        refreshed = await _extract(st, refresh_req)
        parts = [part for part in refreshed["facts"]
                 if part.get("type") != "episode"]
        if not parts:
            raise ValueError("Cannot rebuild derived indexes for merged memory")
        final["entities"] = sorted({e for f in parts for e in f.get("entities", [])})
        final["keywords"] = sorted({e for f in parts for e in f.get("keywords", [])})
        final["retrieval_key"] = " ".join(f.get("retrieval_key", "") for f in parts)
        final_vec = (await embed([_embed_text(final)], stage="add.embed_update"))[0]
        st.replace_fact(target, final, final_vec)
        for t in refreshed["triples"]:
            st.insert_triple(req.user_id, t["subject"], t["relation"], t["object"], target)
        st.link_sources(target, req.request_id, fact.get("_sources", []))
        return None  # refreshed triples already persisted against final content
    supersedes = None
    if op == "SUPERSEDE" and detail.get("target_id"):
        st.close_validity(detail["target_id"],
                          fact.get("event_time") or now)
        supersedes = detail["target_id"]
    return st.insert_amu(
        user_id=req.user_id, session_id=req.session_id,
        content=fact["content"],
        retrieval_key=fact.get("retrieval_key", ""),
        type=fact.get("type", "fact"),
        entities=fact.get("entities") or [],
        keywords=fact.get("keywords") or [],
        event_time=fact.get("event_time"),
        valid_from=fact.get("event_time") or now,
        supersedes=supersedes,
        confidence=0.9,
        sensitivity=fact.get("sensitivity", "normal"),
        embedding=vec)


async def _update_summary(st: store.Store, req: schemas.AddRequest):
    prompt = prompts.render(
        "03_session_summary.txt",
        current_summary=st.get_summary(req.user_id, req.session_id)
        or "(empty)",
        chunk_messages=_format_messages(req.messages))
    summary = (await llm.complete(prompt, stage="add.summary")).strip()
    if not summary:
        raise ValueError("Summary generation returned empty text")
    st.set_summary(req.user_id, req.session_id, summary)


async def run_add(st: store.Store, req: schemas.AddRequest) -> None:
    """Publish a complete Add atomically; failures leave the live store unchanged."""
    debug_record = None
    async with st.add_lock(req.user_id):
        if st.request_seen(req.request_id):
            return
        with st.staged(req.user_id) as work:
            await _run_add(work, req)
            if config.MEMORY_DEBUG_LOG:
                try:
                    debug_record = memory_debug.capture(work, req)
                except Exception:
                    log.warning("Could not capture memory debug snapshot", exc_info=True)
    if debug_record is not None:
        memory_debug.append(debug_record)


async def _run_add(st: store.Store, req: schemas.AddRequest) -> None:
    st.save_messages(req)
    data = await _extract(st, req)
    facts = data.get("facts") or [{
        "content": _format_messages(req.messages), "type": "episode",
        "_sources": list(range(len(req.messages)))}]
    vecs = await embed([_embed_text(f) for f in facts], stage="add.embed_facts")
    if len(vecs) != len(facts):
        raise ValueError("Embedding count does not match extracted facts")
    for i, (fact, vec) in enumerate(zip(facts, vecs)):
        aid = await _persist_fact(st, req, fact, vec)
        if not aid:
            continue
        st.link_sources(aid, req.request_id,
                        fact.get("_sources", list(range(len(req.messages)))))
        for t in data.get("triples", []):
            if t.get("fact_index") == i:
                st.insert_triple(req.user_id, t["subject"], t["relation"], t["object"], aid)
    await _update_summary(st, req)
    st.record_request(req.request_id, req.user_id, req.session_id)
