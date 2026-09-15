"""Task-level experience distillation for ULM design section 6."""
import json
import re
from typing import List

from . import llm, prompts, schemas, store
from .embeddings import embed


def _signature(req: schemas.FeedbackRequest) -> str:
    if req.task_signature:
        return req.task_signature.strip()
    return " ".join(re.findall(r"[\w./-]+", req.task.casefold()))[:240]


async def run_feedback(st: store.Store, req: schemas.FeedbackRequest) -> List[str]:
    """Distill explicit outcome evidence; ordinary chat never enters this path."""
    signature = _signature(req)
    prompt = prompts.render(
        "09_experience_distill.txt", task=req.task, outcome=req.outcome,
        trace=req.trace, task_signature=signature,
        environment_verified=str(req.environment_verified).lower())
    result = await llm.complete_json(
        prompt, '{"items":[{"type":"strategy","title":"...",'
        '"description":"...","content":"...","keywords":[]}]}',
        schema=llm.STRUCTURED_SCHEMAS["experience"], stage="feedback.distill")
    items = result.get("items") or []
    if not isinstance(items, list) or len(items) > 3:
        raise ValueError("Experience distiller returned an invalid item list")
    texts = []
    for item in items:
        if item.get("type") not in ("strategy", "workflow", "skill", "playbook"):
            raise ValueError("Invalid experience memory type")
        if not all(isinstance(item.get(key), str) and item[key].strip()
                   for key in ("title", "description", "content")):
            raise ValueError("Incomplete experience memory")
        texts.append(" ".join((signature, item["title"], item["description"], item["content"])))
    if not texts:
        return []
    vectors = await embed(texts, stage="feedback.embed")
    ids = []
    with st.staged(req.user_id) as work:
        for item, vector in zip(items, vectors):
            duplicates = [candidate for candidate in work.nearest_by_embedding(
                req.user_id, vector, 5) if candidate.get("type") == item["type"]
                and candidate.get("task_signature") == signature
                and candidate.get("_score", 0.0) >= 0.97]
            if duplicates:
                target = duplicates[0]["id"]
                work.record_experience_feedback(
                    target, req.outcome == "success", req.session_id)
                ids.append(target)
                continue
            content = f"{item['title']}: {item['description']}\n{item['content']}"
            ids.append(work.insert_amu(
                user_id=req.user_id, session_id=req.session_id, content=content,
                retrieval_key=signature, type=item["type"],
                keywords=item.get("keywords") or [], embedding=vector,
                polarity=req.outcome, helpful=int(req.outcome == "success"),
                harmful=int(req.outcome == "failure"),
                verified=bool(item["type"] == "skill" and req.environment_verified),
                task_signature=signature, evidence=[{"kind": "task_feedback"}]))
    return ids