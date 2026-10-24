"""Independent persistence for the compatible feedback/experience API."""
import json
import re
import uuid
from datetime import datetime, timezone
from .store import Store, Conflict, digest


def now():
    return datetime.now(timezone.utc).isoformat()


def signature(payload):
    return (payload.get('task_signature') or '').strip() or ' '.join(
        re.findall(r'[\w./-]+', payload['task'].casefold()))[:240]


def scope(payload):
    result = {k: payload.get(k) for k in
              ('environment_signature', 'code_version', 'dependency_versions')}
    trajectory = payload.get('trajectory') or {}
    result.update({k: trajectory.get(k, {} if k == 'settings' else [] if k in
        ('constraints', 'acceptance_criteria') else None) for k in
        ('task_type', 'dataset_version', 'model', 'constraints', 'acceptance_criteria', 'settings')})
    result['execution_kind'] = payload.get('execution_kind', 'observed')
    return result


class HTTPSStore(Store):
    def __init__(self, path, space_id):
        super().__init__(path, space_id)
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS feedback_events(
                user_id TEXT, event_id TEXT, payload TEXT NOT NULL,
                PRIMARY KEY(user_id,event_id));
            CREATE TABLE IF NOT EXISTS experiences(
                user_id TEXT, memory_id TEXT, payload TEXT NOT NULL,
                PRIMARY KEY(user_id,memory_id));
        ''')

    def feedback(self, user, event_id):
        with self.lock:
            row = self.db.execute('SELECT payload FROM feedback_events WHERE user_id=? AND event_id=?',
                                  (user, event_id)).fetchone()
            return json.loads(row[0]) if row else None

    def experience(self, user, memory_id):
        with self.lock:
            row = self.db.execute('SELECT payload FROM experiences WHERE user_id=? AND memory_id=?',
                                  (user, memory_id)).fetchone()
            if not row:
                raise ValueError('Experience memory not found')
            return json.loads(row[0])

    def save_feedback(self, req, event_id, payload_hash, events, expected_epoch):
        payload = req.model_dump(exclude={'feedback_event_id'})
        with self.lock, self.db:
            self.db.execute('BEGIN IMMEDIATE')
            existing = self.feedback(req.user_id, event_id)
            if existing:
                if existing['payload_hash'] != payload_hash:
                    raise ValueError('Feedback event was reused with different content')
                return existing['memory_ids']
            revision, epoch = self.state(req.user_id)
            if epoch != expected_epoch:
                raise Conflict('user purged while feedback was preparing')
            for used in req.used_memory_ids:
                if not self.db.execute('SELECT 1 FROM events WHERE user_id=? AND event_id=?',
                                       (req.user_id, used)).fetchone():
                    raise ValueError('Missing or cross-user attributed memory')
            if req.used_memory_ids and not req.attribution_reason:
                raise ValueError('Attributed feedback requires an attribution reason')
            ids = []
            for event in events:
                memory_id = event['event_id']
                ids.append(memory_id)
                event['revision'] = revision + 1
                event['feedback_event_id'] = event_id
                event['memory_type'] = 'workflow'
                event['experience_status'] = 'hypothesis'
                self.db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)',
                    (req.user_id, memory_id, req.session_id, event['logical_id'],
                     event['valid_from'], None, json.dumps(event, ensure_ascii=False)))
                record = dict(memory_id=memory_id, version=1, content=event['text'],
                    experience=dict(schema_version=1, status='hypothesis', scope=scope(payload),
                        applicability=[req.task], exclusions=[],
                        validation_plan='Review against compatible verified execution evidence.', history=[]),
                    evidence=[dict(kind='task_feedback', event_id=event_id, quote=req.trace,
                                   payload_hash=payload_hash, relation='context', case_key=payload_hash,
                                   outcome=req.outcome, **scope(payload))], independent_cases=1,
                    task_signature=signature(payload))
                self.db.execute('INSERT INTO experiences VALUES (?,?,?)',
                                (req.user_id, memory_id, json.dumps(record, ensure_ascii=False)))
            record = dict(user_id=req.user_id, event_id=event_id, payload_hash=payload_hash,
                          payload=payload, memory_ids=ids, created_at=now())
            self.db.execute('INSERT INTO feedback_events VALUES (?,?,?)',
                            (req.user_id, event_id, json.dumps(record, ensure_ascii=False)))
            self.db.execute('INSERT INTO tenants VALUES (?,?,?) ON CONFLICT(user_id) DO UPDATE SET revision=excluded.revision',
                            (req.user_id, revision + 1, epoch))
            return ids

    def review(self, req):
        transitions = {'hypothesis': {'validated', 'rejected', 'stale'},
                       'validated': {'adopted', 'rejected', 'stale'},
                       'adopted': {'rejected', 'stale'}, 'rejected': {'hypothesis'}, 'stale': {'hypothesis'}}
        with self.lock, self.db:
            self.db.execute('BEGIN IMMEDIATE')
            record = self.experience(req.user_id, req.memory_id)
            lifecycle = record['experience']
            if record['version'] != req.expected_version:
                raise ValueError('Experience version changed')
            if req.status not in transitions[lifecycle['status']]:
                raise ValueError('Invalid experience state transition')
            verified = False
            for ref in req.evidence:
                event = self.feedback(req.user_id, ref.event_id)
                if not event or not ref.quote.strip() or ref.quote not in event['payload']['trace']:
                    raise ValueError('Review evidence must quote a stored feedback event')
                payload = event['payload']
                target, original = scope(payload), lifecycle['scope']
                compatible = signature(payload) == record['task_signature'] and all(
                    target[k] == original[k] for k in original if k != 'execution_kind')
                verified |= bool(ref.relation == 'support' and compatible
                    and payload['execution_kind'] != 'simulation'
                    and payload['verification_result'] == 'passed' and payload['environment_verified']
                    and payload['environment_signature'] and payload['code_version'] and payload['dependency_versions'])
                record['evidence'].append(dict(kind='task_feedback', **ref.model_dump(),
                    payload_hash=event['payload_hash'], execution_kind=payload['execution_kind'], review_status=req.status))
            if req.status in {'validated', 'adopted'} and (not verified or lifecycle['scope']['execution_kind'] == 'simulation'):
                raise ValueError('Promotion requires compatible verified execution evidence, not simulation')
            lifecycle['history'].append(dict(previous=lifecycle['status'], status=req.status,
                reviewer=req.reviewer, rationale=req.rationale, evidence=[r.model_dump() for r in req.evidence], at=now()))
            lifecycle['status'] = req.status
            record['version'] += 1
            self.db.execute('UPDATE experiences SET payload=? WHERE user_id=? AND memory_id=?',
                            (json.dumps(record, ensure_ascii=False), req.user_id, req.memory_id))
            row = self.db.execute('SELECT payload FROM events WHERE user_id=? AND event_id=?',
                                 (req.user_id, req.memory_id)).fetchone()
            event = json.loads(row[0])
            event['experience_status'] = req.status
            self.db.execute('UPDATE events SET payload=? WHERE user_id=? AND event_id=?',
                            (json.dumps(event, ensure_ascii=False), req.user_id, req.memory_id))
            self.db.execute('UPDATE tenants SET revision=revision+1 WHERE user_id=?', (req.user_id,))
            return record

    def snapshot(self, req):
        revision, epoch, events = super().snapshot(req)
        return revision, epoch, [e for e in events if e.get('experience_status') not in {'rejected', 'stale'}]

    def purge(self, user):
        with self.lock, self.db:
            self.db.execute('BEGIN IMMEDIATE')
            revision, epoch = self.state(user)
            counts = {}
            for table in ('events', 'requests', 'feedback_events', 'experiences'):
                counts[table] = self.db.execute(f'DELETE FROM {table} WHERE user_id=?', (user,)).rowcount
            self.db.execute('INSERT INTO tenants VALUES (?,?,?) ON CONFLICT(user_id) DO UPDATE SET revision=excluded.revision,epoch=excluded.epoch',
                            (user, revision + 1, epoch + 1))
            return dict(success=True, receipt_id=uuid.uuid4().hex, user_id=user, deleted_at=now(),
                        row_counts=counts, write_revision=revision + 1, scope_epoch=epoch + 1)
