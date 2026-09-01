"""Add pipeline (design doc v0.2 §4):
normalize -> rolling summary -> AMU extraction -> item-level governance ->
multi-index write. Synchronous from the caller's point of view.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List

from . import config, llm, prompts, schemas, store
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


async def _extract(st: store.Store, req: schemas.AddRequest) -> Dict:
    summary = st.get_summary(req.user_id, req.session_id)
    recent = _format_messages(req.messages[-4:])
    prompt = prompts.render(
        "01_extract_amu.txt",
        session_summary=summary or "(none yet)",
        recent_messages=recent,
        chunk_messages=_format_messages(req.messages),
        reference_time=_ref_time(req.messages),
    )
    try:
        data = llm.extract_json(await llm.complete(prompt))
        data.setdefault("facts", [])
        data.setdefault("triples", [])
        return data
    except Exception as e:
        log.warning("extraction failed (%s); falling back to episode", e)
        return {"facts": [], "triples": [], "_fallback": True}


async def _govern_one(st: store.Store, user_id: str, fact: Dict,
                      vec) -> str:
    """Item-level governance: ADD / UPDATE / SUPERSEDE / NOOP."""
    neighbors = st.nearest_by_embedding(user_id, vec,
                                        config.GOVERNANCE_NEIGHBORS)
    # restrict to confident near-duplicates / same-entity items
    cand = [n for n in neighbors if n["_score"] > 0.55]
    if not cand:
        op = {"operation": "ADD", "target_id": None, "merged_content": None}
    else:
        cand_text = "\n".join(f"{n['id']}: {n['content']}" for n in cand)
        prompt = prompts.render(
            "02_governance_decision.txt",
            new_memory=fact["content"],
            neighbor_memories=cand_text)
        try:
            op = llm.extract_json(await llm.complete(prompt))
        except Exception:
            op = {"operation": "ADD", "target_id": None,
                  "merged_content": None}
    return op.get("operation", "ADD"), op


async def _persist_fact(st: store.Store, req: schemas.AddRequest,
                        fact: Dict, vec) -> str:
    op, detail = await _govern_one(st, req.user_id, fact, vec)
    now = datetime.now(timezone.utc).isoformat()
    if op == "NOOP":
        return "noop"
    if op == "UPDATE" and detail.get("target_id"):
        st.update_amu_content(detail["target_id"],
                              detail.get("merged_content") or fact["content"])
        return "update"
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
    try:
        summary = (await llm.complete(prompt)).strip()
        if summary:
            st.set_summary(req.user_id, req.session_id, summary)
    except Exception as e:
        log.warning("summary update failed: %s", e)


async def run_add(st: store.Store, req: schemas.AddRequest) -> None:
    """Full synchronous Add. Raises nothing; always persists something."""
    if st.request_seen(req.request_id):
        return  # idempotent replay

    data = await _extract(st, req)
    facts: List[Dict] = data.get("facts", [])
    triples: List[Dict] = data.get("triples", [])

    if not facts:
        # fallback: store raw chunk as episode AMU (recall floor)
        vec = (await embed([_format_messages(req.messages)]))[0]
        st.insert_amu(user_id=req.user_id, session_id=req.session_id,
                      content=_format_messages(req.messages),
                      type="episode", embedding=vec)
    else:
        vecs = await embed([_embed_text(f) for f in facts])
        # governance sequentially per fact (shared store), extraction parallel
        results = []
        for f, v in zip(facts, vecs):
            results.append(await _persist_fact(st, req, f, v))
        amu_ids = [r for r in results if r.startswith("amu_")]
        # triples -> link each triple to the first AMU created from its fact
        for t in triples:
            for aid in amu_ids[:1] or [None]:
                if aid:
                    st.insert_triple(req.user_id, t.get("subject", ""),
                                     t.get("relation", ""),
                                     t.get("object", ""), aid)
                    break

    await _update_summary(st, req)
    st.record_request(req.request_id, req.user_id, req.session_id)
