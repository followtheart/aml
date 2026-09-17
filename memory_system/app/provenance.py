"""Versioned claims, source identity and consistent historical read projections."""
import hashlib
import json
from contextlib import contextmanager
from . import integrity

SCHEMA = '''
CREATE TABLE IF NOT EXISTS claim_versions (
 user_id TEXT NOT NULL, amu_id TEXT NOT NULL, version INTEGER NOT NULL,
 recorded_from TEXT, recorded_to TEXT NOT NULL, payload TEXT NOT NULL,
 sources TEXT NOT NULL, triples TEXT NOT NULL, reason TEXT NOT NULL,
 PRIMARY KEY(amu_id,version));
CREATE INDEX IF NOT EXISTS idx_claim_history ON claim_versions(user_id,recorded_from,recorded_to);
CREATE TABLE IF NOT EXISTS view_dependencies (
 user_id TEXT NOT NULL, view_kind TEXT NOT NULL, view_id TEXT NOT NULL,
 source_id TEXT NOT NULL, source_version INTEGER NOT NULL,
 PRIMARY KEY(view_kind,view_id,source_id));
CREATE TABLE IF NOT EXISTS feedback_events (
 user_id TEXT NOT NULL,event_id TEXT NOT NULL,payload_hash TEXT NOT NULL,
 memory_ids TEXT NOT NULL,payload TEXT NOT NULL,
 PRIMARY KEY(user_id,event_id));
'''


class ProvenanceStore:
    def _init_provenance(self):
        self.conn.executescript(SCHEMA)
        additions = {
            'amu': {'version': 'INTEGER NOT NULL DEFAULT 1', 'recorded_from': 'TEXT',
                    'recorded_to': 'TEXT', 'knowledge_status': "TEXT NOT NULL DEFAULT 'legacy_unknown'",
                    'epistemic_status': "TEXT NOT NULL DEFAULT 'asserted'",
                    'resolution_status': "TEXT NOT NULL DEFAULT 'accepted'",
                    'valid_from_status': "TEXT NOT NULL DEFAULT 'unknown'",
                    'valid_to_status': "TEXT NOT NULL DEFAULT 'unknown'",
                    'view_status': "TEXT NOT NULL DEFAULT 'ready'",
                    'trust_scope': "TEXT NOT NULL DEFAULT '[]'"},
            'scenes': {'view_status': "TEXT NOT NULL DEFAULT 'stale'"},
            'source_messages': {'message_id': 'TEXT', 'source_event_id': 'TEXT',
                    'speaker_id': 'TEXT', 'source_kind': "TEXT NOT NULL DEFAULT 'unknown'",
                    'trust_scope': "TEXT NOT NULL DEFAULT '[]'", 'ingested_at': 'TEXT',
                    'sensitivity': "TEXT NOT NULL DEFAULT 'normal'"},
        }
        for table, fields in additions.items():
            existing = {r['name'] for r in self.conn.execute(f'PRAGMA table_info({table})')}
            for field, declaration in fields.items():
                if field not in existing:
                    self.conn.execute(f'ALTER TABLE {table} ADD COLUMN {field} {declaration}')

    def _archive_claim(self, amu_id, reason):
        from .store import _now
        row = self.conn.execute('SELECT * FROM amu WHERE id=?', (amu_id,)).fetchone()
        if not row:
            raise ValueError('Missing claim to revise')
        payload = dict(row)
        if payload.get('embedding') is not None:
            payload['embedding'] = {'hex': payload['embedding'].hex()}
        at = _now()
        sources = [dict(r) for r in self.conn.execute('SELECT * FROM amu_sources WHERE amu_id=?', (amu_id,))]
        triples = [dict(r) for r in self.conn.execute('SELECT * FROM triples WHERE amu_id=?', (amu_id,))]
        self._write('INSERT INTO claim_versions VALUES (?,?,?,?,?,?,?,?,?)',
                    (row['user_id'], amu_id, row['version'], row['recorded_from'], at,
                     json.dumps(payload), json.dumps(sources), json.dumps(triples), reason))
        self._write('UPDATE amu SET version=version+1,recorded_from=?,recorded_to=NULL,knowledge_status=? WHERE id=?',
                    (at, 'known', amu_id))
        self.invalidate_dependencies(amu_id)

    def invalidate_dependencies(self, source_id):
        """Synchronously withdraw derived evidence before a source revision publishes."""
        queue, seen = [source_id], set()
        while queue:
            source = queue.pop()
            if source in seen:
                continue
            seen.add(source)
            rows = self.conn.execute('SELECT * FROM view_dependencies WHERE source_id=?', (source,)).fetchall()
            for dep in rows:
                if dep['view_kind'] == 'amu':
                    self._write("UPDATE amu SET view_status='stale' WHERE id=?", (dep['view_id'],))
                    queue.append(dep['view_id'])
                elif dep['view_kind'] == 'scene':
                    self._write("UPDATE scenes SET view_status='stale' WHERE id=?", (dep['view_id'],))
            # Existing scenes predate explicit dependency registration.
            self._write("UPDATE scenes SET view_status='stale' WHERE id IN (SELECT scene_id FROM amu WHERE id=?)", (source,))

    def register_dependencies(self, view_kind, view_id, source_ids, expected_versions=None):
        """Rebuild hook: validate versions, inherit restrictions, then mark ready."""
        if view_kind not in ('amu', 'scene') or not source_ids:
            raise ValueError('Derived views require a supported kind and sources')
        table = 'amu' if view_kind == 'amu' else 'scenes'
        target = self.conn.execute(f'SELECT user_id FROM {table} WHERE id=?', (view_id,)).fetchone()
        sources = [self.conn.execute('SELECT * FROM amu WHERE id=?', (aid,)).fetchone() for aid in dict.fromkeys(source_ids)]
        if target is None or any(r is None or r['user_id'] != target[0] or r['view_status'] != 'ready' for r in sources):
            raise ValueError('Missing, stale or cross-user dependency')
        if expected_versions and any(expected_versions.get(r['id']) != r['version'] for r in sources):
            raise ValueError('Source version changed during rebuild')
        if view_id in source_ids:
            raise ValueError('A view cannot depend on itself')
        scope = set(json.loads(sources[0]['trust_scope']))
        for r in sources[1:]:
            scope.intersection_update(json.loads(r['trust_scope']))
        sensitivity = max((r['sensitivity'] for r in sources), key=lambda s: {'normal': 0, 'sensitive': 1, 'suppressed': 2}.get(s, 2))
        if view_kind == 'scene' and sensitivity != 'normal':
            raise ValueError('Restricted sources cannot build a shared scene')
        self._write('DELETE FROM view_dependencies WHERE view_kind=? AND view_id=?', (view_kind, view_id))
        for r in sources:
            self._write('INSERT INTO view_dependencies VALUES (?,?,?,?,?)', (target[0], view_kind, view_id, r['id'], r['version']))
        if view_kind == 'amu':
            current = self.conn.execute('SELECT sensitivity FROM amu WHERE id=?', (view_id,)).fetchone()[0]
            sensitivity = max((current, sensitivity), key=lambda s: {'normal': 0, 'sensitive': 1, 'suppressed': 2}.get(s, 2))
            self._write("UPDATE amu SET sensitivity=?,trust_scope=?,view_status='ready' WHERE id=?", (sensitivity, json.dumps(sorted(scope)), view_id))
        else:
            self._write("UPDATE scenes SET view_status='ready' WHERE id=?", (view_id,))
        self._touch_user(target[0])
        self.conn.commit()

    def support_events(self, amu_id):
        """Only original observations count; derived views inherit their event union."""
        found, visited, queue = set(), set(), [amu_id]
        while queue:
            aid = queue.pop()
            if aid in visited:
                continue
            visited.add(aid)
            for source in self.sources_for_amu(aid):
                found.add(source.get('source_event_id') or f"legacy:{source['request_id']}:{source['message_index']}")
            queue.extend(r[0] for r in self.conn.execute("SELECT source_id FROM view_dependencies WHERE view_kind='amu' AND view_id=?", (aid,)))
        return sorted(found)

    def dependencies_for(self, view_id, view_kind='amu'):
        return [dict(r) for r in self.conn.execute(
            'SELECT source_id,source_version FROM view_dependencies WHERE view_kind=? AND view_id=?', (view_kind, view_id))]

    def get_feedback(self, user_id, event_id):
        row = self.conn.execute('SELECT * FROM feedback_events WHERE user_id=? AND event_id=?', (user_id, event_id)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result['memory_ids'] = json.loads(result['memory_ids'])
        result['payload'] = json.loads(result['payload'])
        return result

    def save_feedback(self, user_id, event_id, payload_hash, memory_ids, payload):
        existing = self.get_feedback(user_id, event_id)
        if existing:
            if existing['payload_hash'] != payload_hash:
                raise ValueError('Feedback event already has different content')
            return existing
        self._write('INSERT INTO feedback_events VALUES (?,?,?,?,?)',
                    (user_id, event_id, payload_hash, json.dumps(memory_ids), json.dumps(payload, ensure_ascii=False)))
        self._touch_user(user_id)
        self.conn.commit()
        return self.get_feedback(user_id, event_id)

    def mark_resolution(self, amu_id, status, reason='dispute'):
        if status not in ('accepted', 'disputed', 'retracted'):
            raise ValueError('Invalid resolution status')
        self._archive_claim(amu_id, reason)
        self._write('UPDATE amu SET resolution_status=? WHERE id=?', (status, amu_id))
        user = self.conn.execute('SELECT user_id FROM amu WHERE id=?', (amu_id,)).fetchone()[0]
        self._touch_user(user)
        self.conn.commit()

    @contextmanager
    def snapshot(self, user_id, min_revision=None, as_of=None):
        """An isolated read projection; live Store.assert_epoch guards final delivery."""
        from .store import Store, _LOCK, _now
        if as_of and integrity.instant(as_of) > integrity.instant(_now()):
            raise ValueError('as_of cannot be in the future')
        work = Store(':memory:')
        try:
            with _LOCK:
                self.conn.execute('BEGIN')
                try:
                    state = self.user_state(user_id)
                    if min_revision is not None and state['revision'] < min_revision:
                        raise ValueError('Requested revision is not committed')
                    self._copy_user_to(work, user_id)
                finally:
                    self.conn.rollback()
            work.read_revision, work.read_epoch = state['revision'], state['epoch']
            work.read_as_of = as_of
            if as_of:
                work._project_as_of(user_id, as_of)
            work.conn.execute('PRAGMA query_only=ON')
            yield work
        finally:
            work.conn.close()

    def _project_as_of(self, user_id, as_of):
        """Reconstruct claims plus their exact source/graph dependencies at knowledge time."""
        at = integrity.instant(as_of)
        chosen = {}
        suppressed = {r[0] for r in self.conn.execute("SELECT id FROM amu WHERE user_id=? AND sensitivity='suppressed'", (user_id,))}
        for row in self.conn.execute('SELECT * FROM amu WHERE user_id=?', (user_id,)):
            if row['recorded_from'] and integrity.instant(row['recorded_from']) <= at:
                chosen[row['id']] = (dict(row), None, None)
        for row in self.conn.execute('SELECT * FROM claim_versions WHERE user_id=?', (user_id,)):
            if row['recorded_from'] and integrity.instant(row['recorded_from']) <= at < integrity.instant(row['recorded_to']):
                payload = json.loads(row['payload'])
                if isinstance(payload.get('embedding'), dict):
                    payload['embedding'] = bytes.fromhex(payload['embedding']['hex'])
                payload['recorded_to'] = row['recorded_to']
                chosen[row['amu_id']] = (payload, json.loads(row['sources']), json.loads(row['triples']))
        self.conn.execute('DELETE FROM amu WHERE user_id=?', (user_id,))
        self.conn.execute('DELETE FROM amu_fts WHERE user_id=?', (user_id,))
        for aid, (payload, sources, triples) in chosen.items():
            if aid in suppressed:
                continue
            names = list(payload)
            self.conn.execute(f"INSERT INTO amu ({','.join(names)}) VALUES ({','.join('?' for _ in names)})", tuple(payload.values()))
            self.conn.execute('INSERT INTO amu_fts VALUES (?,?,?,?)', (aid,user_id,payload['content'],payload['retrieval_key']))
            if sources is not None:
                self.conn.execute('DELETE FROM amu_sources WHERE amu_id=?', (aid,))
                for s in sources:
                    self.conn.execute('INSERT INTO amu_sources VALUES (?,?,?)', (aid,s['request_id'],s['message_index']))
            if triples is not None:
                self.conn.execute('DELETE FROM triples WHERE amu_id=?', (aid,))
                for t in triples:
                    names = list(t)
                    self.conn.execute(f"INSERT INTO triples ({','.join(names)}) VALUES ({','.join('?' for _ in names)})", tuple(t.values()))
        self.conn.execute('DELETE FROM triples WHERE amu_id NOT IN (SELECT id FROM amu)')
        self.conn.execute('DELETE FROM amu_sources WHERE amu_id NOT IN (SELECT id FROM amu)')
        self.conn.execute('DELETE FROM source_messages WHERE ingested_at IS NULL OR julianday(ingested_at)>julianday(?)', (as_of,))
        self.conn.execute("UPDATE scenes SET view_status='stale'")
        self.conn.execute("UPDATE amu SET view_status='stale' WHERE id IN (SELECT view_id FROM view_dependencies WHERE view_kind='amu')")
        self.conn.execute('DELETE FROM sessions')
        self.conn.execute('DELETE FROM sessions_fts')
        self.conn.commit()


def source_identity(req, message, index):
    """Identity is supplied by the ingestion boundary, never an extraction model."""
    identity = message.message_id or f'{req.request_id}:{index}'
    # No event key means repeats cannot establish independent observations.
    event = message.source_event_id or 'observation_' + hashlib.sha256(
        json.dumps([req.user_id, message.speaker_id or message.role, message.content], ensure_ascii=False).encode()).hexdigest()
    return identity, event
