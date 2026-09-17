"""Task-level experience distillation for ULM design section 6."""
import json
import hashlib
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
    epoch = st.user_state(req.user_id)["epoch"]
    payload = req.model_dump(exclude={'feedback_event_id'})
    payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    event_id = req.feedback_event_id or payload_hash
    existing = st.get_feedback(req.user_id, event_id)
    if existing:
        if existing['payload_hash'] != payload_hash:
            raise ValueError('Feedback event was reused with different content')
        return existing['memory_ids']
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
    vectors = await embed(texts, stage="feedback.embed") if texts else []
    ids = []
    with st.staged(req.user_id, expected_epoch=epoch) as work:
        existing = work.get_feedback(req.user_id, event_id)
        if existing:
            if existing['payload_hash'] != payload_hash:
                raise ValueError('Feedback event was reused with different content')
            return existing['memory_ids']
        used = work.get_amus_by_ids(list(set(req.used_memory_ids)))
        used = [a for a in used if a['user_id'] == req.user_id]
        if used and not req.attribution_reason:
            raise ValueError('Attributed feedback requires an attribution reason')
        for memory in used:
            work.record_experience_feedback(memory['id'], req.outcome == 'success', event_id)
        work.record_recall([a['id'] for a in used], sorted({a['scene_id'] for a in used if a.get('scene_id')}))
        for item, vector in zip(items, vectors):
            duplicates = [candidate for candidate in work.nearest_by_embedding(
                req.user_id, vector, 5) if candidate.get("type") == item["type"]
                and candidate.get("task_signature") == signature
                and candidate.get("_score", 0.0) >= 0.97]
            if duplicates:
                target = duplicates[0]["id"]
                ids.append(target)
                continue
            content = f"{item['title']}: {item['description']}\n{item['content']}"
            ids.append(work.insert_amu(
                user_id=req.user_id, session_id=req.session_id, content=content,
                retrieval_key=signature, type=item["type"],
                keywords=item.get("keywords") or [], embedding=vector,
                polarity=req.outcome, helpful=0, harmful=0,
                verified=bool(item['type'] == 'skill' and req.verification_result == 'passed'
                              and req.environment_signature and req.code_version and req.dependency_versions),
                epistemic_status='inferred',
                task_signature=signature, evidence=[{'kind': 'task_feedback', 'event_id': event_id,
                    'environment_signature': req.environment_signature, 'code_version': req.code_version,
                    'dependency_versions': req.dependency_versions, 'verification_result': req.verification_result}]))
        work.save_feedback(req.user_id, event_id, payload_hash, ids, payload)
    return ids
