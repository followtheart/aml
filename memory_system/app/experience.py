"""Task-level experience distillation for ULM design section 6."""
import json
import hashlib
import re
from typing import List
from datetime import datetime, timezone

from . import llm, prompts, schemas, store
from .embeddings import embed


def _signature(req: schemas.FeedbackRequest) -> str:
    if req.task_signature:
        return req.task_signature.strip()
    return " ".join(re.findall(r"[\w./-]+", req.task.casefold()))[:240]


def feedback_identity(req):
    payload = req.model_dump(exclude={'feedback_event_id'})
    # Preserve hashes of requests created before structured trajectories existed.
    if req.execution_kind == 'observed':
        payload.pop('execution_kind', None)
    if req.trajectory is None:
        payload.pop('trajectory', None)
    if req.distill:
        payload.pop('distill', None)
    payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    event_id = req.feedback_event_id or payload_hash
    return payload, payload_hash, event_id


async def run_feedback(st: store.Store, req: schemas.FeedbackRequest) -> List[str]:
    """Distill explicit outcome evidence; ordinary chat never enters this path."""
    epoch = st.user_state(req.user_id)["epoch"]
    payload, payload_hash, event_id = feedback_identity(req)
    existing = st.get_feedback(req.user_id, event_id)
    if existing:
        if existing['payload_hash'] != payload_hash:
            raise ValueError('Feedback event was reused with different content')
        return existing['memory_ids']
    signature = _signature(req)
    prompt = prompts.render(
        "09_experience_distill.txt", task=req.task, outcome=req.outcome,
        trace=req.trace, task_signature=signature,
        environment_verified=str(req.environment_verified).lower(),
        execution_kind=req.execution_kind,
        trajectory=json.dumps(req.trajectory.model_dump() if req.trajectory else {}, ensure_ascii=False))
    result = await llm.complete_json(
        prompt, '{"items":[{"type":"strategy","title":"...",'
        '"description":"...","content":"...","keywords":[],"applicability":[],"exclusions":[], '
        '"validation_plan":"...","evidence_quotes":["exact trace quote"]}]}',
        schema=llm.STRUCTURED_SCHEMAS["experience"], stage="feedback.distill") if req.distill else {'items': []}
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
        quotes = item.get('evidence_quotes', [])
        if not isinstance(quotes, list) or not quotes or any(not isinstance(q, str) or not q.strip() or q not in req.trace for q in quotes):
            raise ValueError('Experience requires exact quotes from the feedback trace')
        if item['type'] == 'skill' and not req.environment_verified:
            raise ValueError('Skill requires a verified environment')
        for key in ('applicability', 'exclusions'):
            if not isinstance(item.get(key), list) or any(not isinstance(v, str) for v in item[key]):
                raise ValueError('Experience conditions must be string lists')
        if not isinstance(item.get('validation_plan'), str) or not item['validation_plan'].strip():
            raise ValueError('Experience requires a validation plan')
        texts.append(" ".join((signature, item["title"], item["description"], item["content"])))
    vectors = await embed(texts, stage="feedback.embed") if texts else []
    ids = []
    with st.staged(req.user_id, expected_epoch=epoch) as work:
        existing = work.get_feedback(req.user_id, event_id)
        if existing:
            if existing['payload_hash'] != payload_hash:
                raise ValueError('Feedback event was reused with different content')
            return existing['memory_ids']
        used_ids = set(req.used_memory_ids) | {mid for step in (req.trajectory.steps if req.trajectory else []) for mid in step.used_memory_ids}
        used = work.get_amus_by_ids(list(used_ids))
        if {a['id'] for a in used if a['user_id'] == req.user_id} != used_ids:
            raise ValueError('Missing or cross-user attributed memory')
        used = [a for a in used if a['user_id'] == req.user_id]
        if used and not req.attribution_reason:
            raise ValueError('Attributed feedback requires an attribution reason')
        for memory in used if req.execution_kind == 'observed' and req.outcome in ('success', 'failure') else []:
            work.record_experience_feedback(memory['id'], req.outcome == 'success', event_id)
        if req.execution_kind == 'observed':
            work.record_recall([a['id'] for a in used], sorted({a['scene_id'] for a in used if a.get('scene_id')}))
        for item, vector in zip(items, vectors):
            content = f"{item['title']}: {item['description']}\n{item['content']}"
            scope = dict(environment_signature=req.environment_signature, code_version=req.code_version,
                         dependency_versions=req.dependency_versions, execution_kind=req.execution_kind,
                         task_type=req.trajectory.task_type if req.trajectory else None,
                         dataset_version=req.trajectory.dataset_version if req.trajectory else None,
                         model=req.trajectory.model if req.trajectory else None,
                         constraints=req.trajectory.constraints if req.trajectory else [],
                         acceptance_criteria=req.trajectory.acceptance_criteria if req.trajectory else [],
                         settings=req.trajectory.settings if req.trajectory else {})
            content += ('\nApplicability: ' + '; '.join(item['applicability']) +
                        '\nExclusions: ' + '; '.join(item['exclusions']) +
                        '\nEnvironment: ' + str(req.environment_signature or 'unspecified') +
                        '; code version: ' + str(req.code_version or 'unspecified'))
            case_key = hashlib.sha256(json.dumps(dict(task=signature, trace=req.trace, scope=scope),
                                                sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            links = [dict(kind='task_feedback', event_id=event_id, payload_hash=payload_hash,
                          quote=q, relation='context', case_key=case_key, outcome=req.outcome, **scope)
                     for q in item['evidence_quotes']]
            lifecycle = dict(schema_version=1, status='hypothesis', scope=scope,
                             applicability=item['applicability'], exclusions=item['exclusions'],
                             validation_plan=item['validation_plan'], history=[])
            duplicates = [c for c in work.nearest_by_embedding(req.user_id, vector, 5)
                          if c.get('type') == item['type'] and c.get('task_signature') == signature
                          and c.get('content') == content and c.get('polarity') == req.outcome
                          and (c.get('state') or {}).get('experience', {}).get('scope') == scope
                          and (c.get('state') or {}).get('experience', {}).get('applicability') == item['applicability']
                          and (c.get('state') or {}).get('experience', {}).get('exclusions') == item['exclusions']
                          and c.get('_score', 0.0) >= 0.97]
            if duplicates:
                target = duplicates[0]
                update_metadata(work, target, target['state'], (target.get('evidence') or []) + links,
                                'experience_evidence_added')
                ids.append(target['id'])
                continue
            ids.append(work.insert_amu(
                user_id=req.user_id, session_id=req.session_id, content=content,
                retrieval_key=signature, type=item['type'], keywords=item.get('keywords') or [],
                embedding=vector, polarity=req.outcome, helpful=0, harmful=0, verified=False,
                epistemic_status='inferred', task_signature=signature,
                state={'experience': lifecycle}, evidence=links))
        work.save_feedback(req.user_id, event_id, payload_hash, ids, payload)
    return ids


def update_metadata(st, memory, state, evidence, reason):
    """Called only inside a user-scoped staged transaction."""
    st._archive_claim(memory['id'], reason)
    st.set_metadata(memory['id'], dict(temporal=memory.get('temporal'), state=state, evidence=evidence))
    status = state['experience']['status']
    st._write('UPDATE amu SET verified=? WHERE id=?',
              (int(memory['type'] == 'skill' and status in ('validated', 'adopted')), memory['id']))


def inspect(st, user_id, memory_id):
    with st.snapshot(user_id) as view:
        rows = view.get_amus_by_ids([memory_id])
        if not rows or rows[0]['user_id'] != user_id or not (rows[0].get('state') or {}).get('experience'):
            raise ValueError('Experience memory not found')
        row = rows[0]
        links = [e for e in row.get('evidence') or [] if e.get('kind') == 'task_feedback']
        return dict(memory_id=memory_id, version=row['version'], content=row['content'],
                    experience=row['state']['experience'], evidence=links,
                    independent_cases=len({e['case_key'] for e in links if e.get('case_key')}))


def review(st, req: schemas.ExperienceReviewRequest):
    """Explicit evidence-backed lifecycle review, never a model-driven promotion.

    A verified replay establishes only its recorded environment/scope. Quotes
    are authenticated to stored feedback, not treated as causal proof.
    """
    transitions = {'hypothesis': {'validated', 'rejected', 'stale'},
                   'validated': {'adopted', 'rejected', 'stale'},
                   'adopted': {'rejected', 'stale'}, 'rejected': {'hypothesis'}, 'stale': {'hypothesis'}}
    with st.staged(req.user_id) as work:
        rows = work.get_amus_by_ids([req.memory_id])
        if not rows or rows[0]['user_id'] != req.user_id:
            raise ValueError('Experience memory not found')
        row = rows[0]
        state = row.get('state') or {}
        current = state.get('experience')
        if not current or row['version'] != req.expected_version:
            raise ValueError('Experience version changed or lifecycle missing')
        if req.status not in transitions[current['status']]:
            raise ValueError('Invalid experience state transition')
        verified = False
        links = list(row.get('evidence') or [])
        for ref in req.evidence:
            event = work.get_feedback(req.user_id, ref.event_id)
            if not event or ref.quote not in event['payload']['trace'] or not ref.quote.strip():
                raise ValueError('Review evidence must quote a stored feedback event')
            payload = event['payload']
            scope = current['scope']
            signature = (payload.get('task_signature') or '').strip() or ' '.join(re.findall(r'[\w./-]+', payload['task'].casefold()))[:240]
            compatible = signature == row['task_signature'] and all(payload.get(k) == scope.get(k) for k in
                             ('environment_signature', 'code_version', 'dependency_versions'))
            trajectory = payload.get('trajectory') or {}
            compatible = compatible and all(trajectory.get(k, {} if k == 'settings' else [] if k in ('constraints', 'acceptance_criteria') else None) == scope.get(k)
                                             for k in ('task_type', 'dataset_version', 'model', 'constraints', 'acceptance_criteria', 'settings'))
            verified |= bool(ref.relation == 'support' and compatible
                             and payload.get('execution_kind', 'observed') != 'simulation'
                             and payload.get('verification_result') == 'passed'
                             and payload.get('environment_verified')
                             and payload.get('environment_signature') and payload.get('code_version')
                             and payload.get('dependency_versions'))
            links.append(dict(kind='task_feedback', **ref.model_dump(), payload_hash=event['payload_hash'],
                              execution_kind=payload.get('execution_kind', 'observed'), review_status=req.status))
        if req.status in ('validated', 'adopted') and not verified:
            raise ValueError('Promotion requires compatible verified execution evidence, not simulation')
        if req.status in ('validated', 'adopted') and current['scope']['execution_kind'] == 'simulation':
            raise ValueError('Simulation hypotheses require a separately observed or replayed lesson')
        current['history'].append(dict(previous=current['status'], status=req.status,
                                       reviewer=req.reviewer, rationale=req.rationale,
                                       evidence=[r.model_dump() for r in req.evidence],
                                       at=datetime.now(timezone.utc).isoformat()))
        current['status'] = req.status
        update_metadata(work, row, state, links, 'experience_review')
    return inspect(st, req.user_id, req.memory_id)
