"""SQLite storage layer.

Tables:
  amu          -- atomic memory units (with embedding blob, validity interval,
                  MemScene id, recall strength / tier for ULM forgetting)
  scenes       -- MemScene clusters (centroid, keywords, heat counters)
  triples      -- (subject, relation, object, amu_id) knowledge graph edges
  sessions     -- rolling session summaries
  requests     -- Add idempotency ledger
  amu_fts      -- FTS5 mirror of amu.content + retrieval_key (BM25)

Swap target for production: Postgres + pgvector (same method surface).
"""
import json
import asyncio
from contextlib import contextmanager
import re
import sqlite3
import threading
import uuid
import weakref
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

from . import budget, config, graph, integrity, vector_index
from .provenance import ProvenanceStore, source_identity

_LOCK = threading.RLock()


class MemoryDeleted(RuntimeError):
    """The operation predates a deletion and must not be retried automatically."""

_FORGET_RULE_RE = re.compile(r"\b(forget|forgot|erase|delete|remove|stop (?:remembering|mentioning))\b"
                             r"|忘记|忘掉|删除|别记|不要记住", re.I)

# Invalidated ("forgotten") memories are excluded from every retrieval route
# unconditionally, independently of the sensitive opt-in (Zep edge invalidation).
_SUPPRESSED = " AND sensitivity!='suppressed' AND view_status='ready' AND resolution_status!='retracted'"

SCHEMA = """
CREATE TABLE IF NOT EXISTS amu (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  content TEXT NOT NULL,
    compressed_content TEXT,
  retrieval_key TEXT,
  type TEXT NOT NULL DEFAULT 'fact',
  entities TEXT NOT NULL DEFAULT '[]',
  keywords TEXT NOT NULL DEFAULT '[]',
  event_time TEXT,
  temporal TEXT,
  state TEXT,
  evidence TEXT,
  valid_from TEXT,
  valid_to TEXT,
  supersedes TEXT,
    superseded_by TEXT,
  confidence REAL DEFAULT 0.9,
  sensitivity TEXT DEFAULT 'normal',
  embedding BLOB,
  embedding_space TEXT,
  created_at TEXT NOT NULL,
  scene_id TEXT,
  cell_id TEXT,
  recall_count INTEGER NOT NULL DEFAULT 0,
  last_recalled TEXT,
  strength REAL NOT NULL DEFAULT 1.0,
  tier TEXT NOT NULL DEFAULT 'hot',
    support_sessions TEXT NOT NULL DEFAULT '[]',
    profile_status TEXT,
    expires_at TEXT,
    polarity TEXT,
    helpful INTEGER NOT NULL DEFAULT 0,
    harmful INTEGER NOT NULL DEFAULT 0,
    verified INTEGER NOT NULL DEFAULT 0,
    task_signature TEXT
);
CREATE INDEX IF NOT EXISTS idx_amu_user ON amu(user_id);
CREATE INDEX IF NOT EXISTS idx_amu_user_type ON amu(user_id, type);
CREATE INDEX IF NOT EXISTS idx_amu_scene ON amu(scene_id);

CREATE TABLE IF NOT EXISTS scenes (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  summary TEXT NOT NULL DEFAULT '',
  keywords TEXT NOT NULL DEFAULT '[]',
  centroid BLOB,
  embedding_space TEXT,
  cell_count INTEGER NOT NULL DEFAULT 0,
  visit_count INTEGER NOT NULL DEFAULT 0,
  interaction_count INTEGER NOT NULL DEFAULT 0,
  surprise REAL NOT NULL DEFAULT 0,
  last_access TEXT,
  tier TEXT NOT NULL DEFAULT 'hot',
  promoted_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scenes_user ON scenes(user_id);

CREATE TABLE IF NOT EXISTS triples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  subject TEXT NOT NULL,
  relation TEXT NOT NULL,
  object TEXT NOT NULL,
    amu_id TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT
);
CREATE INDEX IF NOT EXISTS idx_triples_user ON triples(user_id);
CREATE INDEX IF NOT EXISTS idx_triples_ent ON triples(user_id, subject);
CREATE INDEX IF NOT EXISTS idx_triples_candidate ON triples(user_id, amu_id, id);

CREATE TABLE IF NOT EXISTS sessions (
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  summary TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (user_id, session_id)
);

CREATE TABLE IF NOT EXISTS requests (
  request_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_revisions (
  user_id TEXT PRIMARY KEY,
  revision INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS deletion_epochs (
  user_id TEXT PRIMARY KEY,
  epoch INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS purge_receipts (
    receipt_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    deleted_at TEXT NOT NULL,
    row_counts TEXT NOT NULL
);
"""

SOURCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_messages (
 request_id TEXT NOT NULL, message_index INTEGER NOT NULL,
 user_id TEXT NOT NULL, session_id TEXT NOT NULL,
 role TEXT NOT NULL, content TEXT NOT NULL, timestamp INTEGER,
 PRIMARY KEY(request_id, message_index)
);
CREATE TABLE IF NOT EXISTS amu_sources (
 amu_id TEXT NOT NULL, request_id TEXT NOT NULL, message_index INTEGER NOT NULL,
 PRIMARY KEY(amu_id, request_id, message_index)
);
CREATE INDEX IF NOT EXISTS idx_sources_amu ON amu_sources(amu_id);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS amu_fts USING fts5(
  amu_id UNINDEXED, user_id UNINDEXED, content, retrieval_key
);
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
 user_id UNINDEXED, session_id UNINDEXED, summary
);
CREATE VIRTUAL TABLE IF NOT EXISTS source_fts USING fts5(
 request_id UNINDEXED, message_index UNINDEXED, user_id UNINDEXED, content
);
INSERT INTO sessions_fts(user_id,session_id,summary)
 SELECT s.user_id,s.session_id,s.summary FROM sessions s
 WHERE NOT EXISTS (SELECT 1 FROM sessions_fts f
 WHERE f.user_id=s.user_id AND f.session_id=s.session_id);
"""


def _now() -> str:
    return _now_dt().isoformat()


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


class Store(ProvenanceStore):
    def __init__(self, path: Optional[str] = None):
        self.path = path or config.DB_PATH
        self._lock = _LOCK
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._add_locks = weakref.WeakValueDictionary()
        with self._lock:
            self.conn.execute("PRAGMA busy_timeout=5000")
            if self.path != ":memory:":
                self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            # Forward-compatible migration for databases created before vector
            # spaces were recorded. NULL means "unknown" and is not searched.
            columns = {r["name"] for r in self.conn.execute("PRAGMA table_info(amu)")}
            if "compressed_content" not in columns:
                self.conn.execute("ALTER TABLE amu ADD COLUMN compressed_content TEXT")
            if "embedding_space" not in columns:
                self.conn.execute("ALTER TABLE amu ADD COLUMN embedding_space TEXT")
            if "superseded_by" not in columns:
                self.conn.execute("ALTER TABLE amu ADD COLUMN superseded_by TEXT")
            for column in ("temporal", "state", "evidence"):
                if column not in columns:
                    self.conn.execute(f"ALTER TABLE amu ADD COLUMN {column} TEXT")
            for column, decl in (("scene_id", "TEXT"), ("cell_id", "TEXT"),
                                 ("recall_count", "INTEGER NOT NULL DEFAULT 0"),
                                 ("last_recalled", "TEXT"),
                                 ("strength", "REAL NOT NULL DEFAULT 1.0"),
                                 ("tier", "TEXT NOT NULL DEFAULT 'hot'"),
                                 ("support_sessions", "TEXT NOT NULL DEFAULT '[]'"),
                                 ("profile_status", "TEXT"), ("expires_at", "TEXT"),
                                 ("polarity", "TEXT"),
                                 ("helpful", "INTEGER NOT NULL DEFAULT 0"),
                                 ("harmful", "INTEGER NOT NULL DEFAULT 0"),
                                 ("verified", "INTEGER NOT NULL DEFAULT 0"),
                                 ("task_signature", "TEXT")):
                if column not in columns:
                    self.conn.execute(f"ALTER TABLE amu ADD COLUMN {column} {decl}")
            self.conn.executescript(FTS_SCHEMA)
            self.conn.executescript(SOURCE_SCHEMA)
            self._init_provenance()
            self.conn.executescript('''
                CREATE TABLE IF NOT EXISTS search_database_identity (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS vector_generations (user_id TEXT PRIMARY KEY, generation INTEGER NOT NULL DEFAULT 0);
                CREATE TRIGGER IF NOT EXISTS vector_insert AFTER INSERT ON amu BEGIN
                  INSERT INTO vector_generations VALUES (NEW.user_id,1)
                  ON CONFLICT(user_id) DO UPDATE SET generation=generation+1;
                END;
                CREATE TRIGGER IF NOT EXISTS vector_delete AFTER DELETE ON amu BEGIN
                  INSERT INTO vector_generations VALUES (OLD.user_id,1)
                  ON CONFLICT(user_id) DO UPDATE SET generation=generation+1;
                END;
                CREATE TRIGGER IF NOT EXISTS vector_update AFTER UPDATE OF embedding,embedding_space,user_id ON amu BEGIN
                  INSERT INTO vector_generations VALUES (OLD.user_id,1)
                  ON CONFLICT(user_id) DO UPDATE SET generation=generation+1;
                  INSERT INTO vector_generations VALUES (NEW.user_id,1)
                  ON CONFLICT(user_id) DO UPDATE SET generation=generation+1;
                END;
            ''')
            self.conn.execute('INSERT OR IGNORE INTO search_database_identity VALUES (1,?)', (uuid.uuid4().hex,))
            self._cache_identity = self.conn.execute('SELECT value FROM search_database_identity WHERE id=1').fetchone()[0]
            # Source text is immutable. Backfill older databases without changing
            # their claims, privacy labels, vectors or recorded revisions.
            self.conn.execute("""INSERT INTO source_fts(request_id,message_index,user_id,content)
                SELECT s.request_id,s.message_index,s.user_id,s.content FROM source_messages s
                WHERE NOT EXISTS (SELECT 1 FROM source_fts f
                  WHERE f.request_id=s.request_id AND f.message_index=s.message_index)""")
            triple_columns = {r["name"] for r in self.conn.execute("PRAGMA table_info(triples)")}
            for column in ("valid_from", "valid_to"):
                if column not in triple_columns:
                    self.conn.execute(f"ALTER TABLE triples ADD COLUMN {column} TEXT")
            self.conn.commit()

    def _write(self, sql, params=()):
        result = self.conn.execute(sql, params)
        if hasattr(self, "_pending"):
            self._pending.append((sql, params))
        return result

    def _touch_user(self, user_id):
        """Advance a live user's revision; staged stores advance on publish."""
        self._check_user(user_id)
        if hasattr(self, "_pending"):
            return
        self.conn.execute(
            "INSERT INTO user_revisions(user_id,revision) VALUES (?,1) "
            "ON CONFLICT(user_id) DO UPDATE SET revision=revision+1", (user_id,))

    def _copy_user_to(self, work, user_id):
        """Copy only one user's state; avoids an O(database) Add snapshot."""
        self._check_user(user_id)
        def copy(table, where, params):
            budget.check()
            cursor = self.conn.execute(f"SELECT * FROM {table} WHERE {where}", params)
            try:
                while rows := cursor.fetchmany(512):
                    budget.check()
                    columns = rows[0].keys()
                    placeholders = ",".join("?" for _ in columns)
                    names = ",".join(columns)
                    work.conn.executemany(f"INSERT INTO {table}({names}) VALUES ({placeholders})",
                                          ([row[name] for name in columns] for row in rows))
            finally:
                cursor.close()

        copy("amu", "user_id=?", (user_id,))
        copy("scenes", "user_id=?", (user_id,))
        copy("triples", "user_id=?", (user_id,))
        copy("sessions", "user_id=?", (user_id,))
        copy("requests", "user_id=?", (user_id,))
        copy("source_messages", "user_id=?", (user_id,))
        copy("source_fts", "user_id=?", (user_id,))
        copy("claim_versions", "user_id=?", (user_id,))
        copy("view_dependencies", "user_id=?", (user_id,))
        copy("feedback_events", "user_id=?", (user_id,))
        copy("amu_sources", "amu_id IN (SELECT id FROM amu WHERE user_id=?)", (user_id,))
        copy("amu_fts", "user_id=?", (user_id,))
        copy("sessions_fts", "user_id=?", (user_id,))
        work.conn.commit()

    def user_state(self, user_id):
        self._check_user(user_id)
        with self._lock:
            revision = self.conn.execute(
                "SELECT revision FROM user_revisions WHERE user_id=?", (user_id,)).fetchone()
            epoch = self.conn.execute(
                "SELECT epoch FROM deletion_epochs WHERE user_id=?", (user_id,)).fetchone()
        return {"revision": revision[0] if revision else 0, "epoch": epoch[0] if epoch else 0}

    def assert_epoch(self, user_id, expected_epoch):
        self._check_user(user_id)
        if hasattr(self, '_origin'):
            return self._origin.assert_epoch(user_id, expected_epoch)
        if self.user_state(user_id)["epoch"] != expected_epoch:
            raise MemoryDeleted("Memory deleted while request was in progress")

    def _scope_clause(self, alias=''):
        user = getattr(self, 'read_user_id', None)
        return (f' AND {alias}user_id=?', [user]) if user is not None else ('', [])

    def _check_user(self, user_id):
        if getattr(self, 'read_user_id', user_id) != user_id:
            raise ValueError('Read snapshot belongs to another user')

    @contextmanager
    def staged(self, user_id, expected_epoch=None):
        """Prepare one user's Add privately and publish it in a short transaction.

        Different users can prepare and commit independently. A same-user change
        is detected through a per-user revision and must be retried.
        """
        self._check_user(user_id)
        work = Store(":memory:")
        try:
            with self._lock:
                self.conn.execute("BEGIN")
                try:
                    epoch = self.user_state(user_id)["epoch"]
                    if expected_epoch is not None and epoch != expected_epoch:
                        raise MemoryDeleted("Request predates memory deletion")
                    revision_row = self.conn.execute(
                        "SELECT revision FROM user_revisions WHERE user_id=?", (user_id,)
                    ).fetchone()
                    revision = revision_row[0] if revision_row else 0
                    self._copy_user_to(work, user_id)
                finally:
                    self.conn.rollback()
            work._pending = []
            yield work
            with self._lock:
                self.conn.execute("BEGIN IMMEDIATE")
                try:
                    self.assert_epoch(user_id, epoch)
                    current_row = self.conn.execute(
                        "SELECT revision FROM user_revisions WHERE user_id=?", (user_id,)
                    ).fetchone()
                    current = current_row[0] if current_row else 0
                    if current != revision:
                        raise RuntimeError("Memory changed during Add; retry request")
                    for sql, params in work._pending:
                        self.conn.execute(sql, params)
                    self.conn.execute(
                        "INSERT INTO user_revisions(user_id,revision) VALUES (?,1) "
                        "ON CONFLICT(user_id) DO UPDATE SET revision=revision+1",
                        (user_id,))
                    self.conn.commit()
                    work.published_revision = revision + 1
                except BaseException:
                    self.conn.rollback()
                    raise
        finally:
            work.conn.close()

    def save_messages(self, req):
        for i, message in enumerate(req.messages):
            identity, _ = source_identity(req, message, i)
            old = self.conn.execute('SELECT content,role,speaker_id FROM source_messages WHERE user_id=? AND message_id=?',
                                    (req.user_id, identity)).fetchone()
            if old and (old['content'] != message.content or old['role'] != message.role
                        or old['speaker_id'] != (message.speaker_id or message.role)):
                raise ValueError('Source message identity cannot be reused for different content or speaker')
        for i, message in enumerate(req.messages):
            identity, event = source_identity(req, message, i)
            self._write("INSERT INTO source_messages (request_id,message_index,user_id,session_id,role,content,timestamp,"
                        "message_id,source_event_id,speaker_id,source_kind,trust_scope,ingested_at,sensitivity) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (req.request_id, i, req.user_id, req.session_id,
                         message.role, message.content, message.timestamp, identity, event,
                         message.speaker_id or message.role, message.source_kind,
                         json.dumps(message.trust_scope), _now(), message.sensitivity))
            self._write('INSERT INTO source_fts(request_id,message_index,user_id,content) VALUES (?,?,?,?)',
                        (req.request_id, i, req.user_id, message.content))
        self._touch_user(req.user_id)
        self.conn.commit()

    def link_sources(self, amu_id, request_id, indices):
        owner = self.conn.execute("SELECT user_id FROM amu WHERE id=?", (amu_id,)).fetchone()
        if owner is None:
            raise ValueError("Missing source target")
        existing = self.sources_for_amu(amu_id)
        existing_events = {s['source_event_id'] for s in existing}
        existing_links = {(s['request_id'], s['message_index']) for s in existing}
        new_indices = []
        new_event = False
        for i in indices:
            source = self.conn.execute("SELECT user_id FROM source_messages WHERE request_id=? AND message_index=?",
                                       (request_id, i)).fetchone()
            if source is None or source[0] != owner[0]:
                raise ValueError("Missing or cross-user source evidence")
            event = self.conn.execute('SELECT source_event_id FROM source_messages WHERE request_id=? AND message_index=?', (request_id, i)).fetchone()[0]
            if (request_id, i) not in existing_links:
                new_indices.append(i)
                new_event = new_event or event not in existing_events
                existing_links.add((request_id, i))
            existing_events.add(event)
        if existing and new_indices:
            self._archive_claim(amu_id, 'reaffirm')
        for i in new_indices:
            self._write("INSERT OR IGNORE INTO amu_sources VALUES (?,?,?)",
                        (amu_id, request_id, i))
        sources = self.sources_for_amu(amu_id)
        if sources:
            scope = set(json.loads(sources[0]['trust_scope']))
            for source in sources[1:]:
                scope.intersection_update(json.loads(source['trust_scope']))
            current = self.conn.execute('SELECT sensitivity FROM amu WHERE id=?', (amu_id,)).fetchone()[0]
            sensitivity = max([current] + [s['sensitivity'] for s in sources],
                              key=lambda s: {'normal': 0, 'sensitive': 1, 'suppressed': 2}.get(s, 2))
            self._write('UPDATE amu SET sensitivity=?,trust_scope=? WHERE id=?',
                        (sensitivity, json.dumps(sorted(scope)), amu_id))
        row = self.conn.execute("SELECT user_id FROM amu WHERE id=?", (amu_id,)).fetchone()
        if row:
            self._touch_user(row[0])
        self.conn.commit()
        return new_event

    def sources_for_amu(self, amu_id):
        scope, params = self._scope_clause('m.')
        return [dict(r) for r in self.conn.execute(
            "SELECT m.* FROM source_messages m JOIN amu_sources s "
            "ON m.request_id=s.request_id AND m.message_index=s.message_index "
            "JOIN amu a ON a.id=s.amu_id AND a.user_id=m.user_id "
            "WHERE s.amu_id=?" + scope + " ORDER BY m.request_id,m.message_index", (amu_id, *params))]

    def sources_for_amus(self, amu_ids) -> Dict[str, List[Dict]]:
        """One read for many memories; same rows and order as sources_for_amu."""
        ids = list(dict.fromkeys(amu_ids))
        out = {mid: [] for mid in ids}
        scope, params = self._scope_clause('m.')
        # Chunk to stay under SQLite's bound-parameter limit on wide recalls.
        for start in range(0, len(ids), 400):
            chunk = ids[start:start + 400]
            ph = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                "SELECT s.amu_id AS _amu_id, m.* FROM source_messages m JOIN amu_sources s "
                "ON m.request_id=s.request_id AND m.message_index=s.message_index "
                "JOIN amu a ON a.id=s.amu_id AND a.user_id=m.user_id "
                f"WHERE s.amu_id IN ({ph})" + scope + " ORDER BY m.request_id,m.message_index",
                (*chunk, *params)).fetchall()
            for row in rows:
                source = dict(row)
                out[source.pop('_amu_id')].append(source)
        return out

    def replace_fact(self, amu_id, fact, embedding):
        """Replace derived representations together; caller supplies final-text vector."""
        integrity.validate_interval((fact.get("temporal") or {}).get("start"), (fact.get("temporal") or {}).get("end"))
        self._archive_claim(amu_id, 'enrich')
        previous = self.conn.execute('SELECT sensitivity FROM amu WHERE id=?', (amu_id,)).fetchone()[0]
        fact = dict(fact, sensitivity=max((previous, fact.get('sensitivity', 'normal')),
                    key=lambda s: {'normal': 0, 'sensitive': 1, 'suppressed': 2}.get(s, 2)))
        self._write("UPDATE amu SET content=?,retrieval_key=?,type=?,entities=?,"
                    "keywords=?,event_time=?,sensitivity=?,embedding=?,"
                    "embedding_space=? WHERE id=?",
                    (fact["content"], fact.get("retrieval_key", ""),
                     fact.get("type", "fact"), json.dumps(fact.get("entities") or []),
                     json.dumps(fact.get("keywords") or []), fact.get("event_time"),
                     fact.get("sensitivity", "normal"),
                     embedding.astype(np.float32).tobytes(), config.EMBEDDING_SPACE,
                     amu_id))
        self.set_metadata(amu_id, fact)
        self._write("UPDATE amu_fts SET content=?,retrieval_key=? WHERE amu_id=?",
                    (fact["content"], fact.get("retrieval_key", ""), amu_id))
        self._write("DELETE FROM triples WHERE amu_id=?", (amu_id,))
        row = self.conn.execute("SELECT user_id FROM amu WHERE id=?", (amu_id,)).fetchone()
        if row:
            self._touch_user(row[0])
        self.conn.commit()

    # ---------------- idempotency ----------------
    def request_seen(self, request_id: str) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM requests WHERE request_id=?",
                (request_id,)).fetchone()
            return row is not None

    def request_owner(self, request_id: str) -> Optional[Dict]:
        with self._lock:
            row = self.conn.execute(
                "SELECT user_id,session_id FROM requests WHERE request_id=?",
                (request_id,)).fetchone()
        return dict(row) if row else None

    def record_request(self, request_id: str, user_id: str, session_id: str):
        self._check_user(user_id)
        with self._lock:
            self._write(
                "INSERT OR IGNORE INTO requests VALUES (?,?,?,?)",
                (request_id, user_id, session_id, _now()))
            self._touch_user(user_id)
            self.conn.commit()

    def add_lock(self, user_id):
        """Serialize same-user Adds per process while allowing other users."""
        self._check_user(user_id)
        key = (id(asyncio.get_running_loop()), user_id)
        return self._add_locks.setdefault(key, asyncio.Lock())

    def purge_user(self, user_id: str) -> Dict:
        """Hard-delete one user's content and retain a content-free receipt."""
        self._check_user(user_id)
        receipt_id = f"purge_{uuid.uuid4().hex}"
        deleted_at = _now()
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                counts = {
                    table: self.conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE user_id=?", (user_id,)
                    ).fetchone()[0]
                    for table in ("amu", "scenes", "triples", "sessions", "requests",
                                  "source_messages")
                }
                self.conn.execute(
                    "DELETE FROM amu_sources WHERE amu_id IN "
                    "(SELECT id FROM amu WHERE user_id=?)", (user_id,))
                for table in ("amu_fts", "sessions_fts", "source_fts", "triples", "scenes", "amu",
                              "sessions", "requests", "source_messages", "claim_versions", "view_dependencies", "feedback_events"):
                    self.conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
                self.conn.execute(
                    "INSERT INTO deletion_epochs(user_id,epoch) VALUES (?,1) "
                    "ON CONFLICT(user_id) DO UPDATE SET epoch=epoch+1", (user_id,))
                self._touch_user(user_id)
                self.conn.execute(
                    "INSERT INTO purge_receipts VALUES (?,?,?,?)",
                    (receipt_id, user_id, deleted_at, json.dumps(counts, sort_keys=True)))
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise
        vector_index.evict(self._cache_identity, user_id, self.user_state(user_id)['epoch'])
        return {"receipt_id": receipt_id, "user_id": user_id,
                "deleted_at": deleted_at, "row_counts": counts}

    # ---------------- AMU CRUD ----------------
    def insert_amu(self, *, user_id, session_id, content, compressed_content=None,
                   retrieval_key="",
                   type="fact", entities=None, keywords=None, event_time=None,
                   valid_from=None, valid_to=None, supersedes=None,
                   confidence=0.9, sensitivity="normal",
                   embedding: Optional[np.ndarray] = None,
                   temporal=None, state=None, evidence=None,
                   scene_id=None, cell_id=None, polarity=None,
                   helpful=0, harmful=0, verified=False,
                   task_signature=None, epistemic_status='asserted', resolution_status='accepted') -> str:
        self._check_user(user_id)
        integrity.validate_interval(valid_from, valid_to)
        integrity.validate_interval((temporal or {}).get("start"), (temporal or {}).get("end"))
        if epistemic_status not in ('asserted', 'observed', 'inferred', 'planned'):
            raise ValueError('Invalid epistemic status')
        if resolution_status not in ('accepted', 'disputed', 'retracted'):
            raise ValueError('Invalid resolution status')
        amu_id = f"amu_{uuid.uuid4().hex[:16]}"
        blob = (embedding.astype(np.float32).tobytes()
                if embedding is not None else None)
        profile_status = ("rule" if type == "rule" else
                          "static" if type == "profile" else
                          "transient" if type == "preference" else None)
        expires_at = ((_now_dt() + timedelta(
            days=config.PROFILE_TRANSIENT_TTL_DAYS)).isoformat()
            if profile_status == "transient" else None)
        with self._lock:
            self._write(
                """INSERT INTO amu (id,user_id,session_id,content,compressed_content,retrieval_key,
                   type,entities,keywords,event_time,valid_from,valid_to,
                   supersedes,confidence,sensitivity,embedding,embedding_space,created_at,
                   scene_id,cell_id,support_sessions,profile_status,expires_at,
                   polarity,helpful,harmful,verified,task_signature)
                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                     (amu_id, user_id, session_id, content, compressed_content, retrieval_key, type,
                 json.dumps(entities or [], ensure_ascii=False),
                 json.dumps(keywords or [], ensure_ascii=False),
                 event_time, valid_from, valid_to, supersedes, confidence,
                 sensitivity, blob,
                 config.EMBEDDING_SPACE if blob is not None else None, _now(),
                 scene_id, cell_id, json.dumps([session_id]), profile_status, expires_at,
                 polarity, helpful, harmful, int(verified), task_signature))
            self._write(
                "INSERT INTO amu_fts (amu_id,user_id,content,retrieval_key)"
                " VALUES (?,?,?,?)",
                (amu_id, user_id, content, retrieval_key))
            self.set_metadata(amu_id, dict(temporal=temporal, state=state, evidence=evidence))
            self._write('UPDATE amu SET recorded_from=?,knowledge_status=?,epistemic_status=?,resolution_status=?,valid_from_status=?,valid_to_status=? WHERE id=?',
                        (_now(), 'known', epistemic_status, resolution_status,
                         'known' if valid_from else 'unknown', 'known' if valid_to else 'open', amu_id))
            self._touch_user(user_id)
            self.conn.commit()
        return amu_id

    def set_metadata(self, amu_id, fact):
        temporal = fact.get("temporal") or {}
        integrity.validate_interval(temporal.get("start"), temporal.get("end"))
        self._write("UPDATE amu SET temporal=?,state=?,evidence=? WHERE id=?",
                    (json.dumps(fact["temporal"]) if fact.get("temporal") is not None else None,
                     json.dumps(fact["state"]) if fact.get("state") is not None else None,
                     json.dumps(fact.get("evidence") or [], ensure_ascii=False), amu_id))
        if fact.get('epistemic_status') is not None:
            if fact['epistemic_status'] not in ('asserted', 'observed', 'inferred', 'planned'):
                raise ValueError('Invalid epistemic status')
            self._write('UPDATE amu SET epistemic_status=? WHERE id=?', (fact['epistemic_status'], amu_id))

    def update_amu_content(self, amu_id: str, content: str,
                           confidence: Optional[float] = None):
        with self._lock:
            self._archive_claim(amu_id, 'correction')
            self._write(
                "UPDATE amu SET content=?, embedding=NULL, embedding_space=NULL, retrieval_key='', "
                "entities='[]', keywords='[]', temporal=NULL, state=NULL, evidence=NULL, confidence=COALESCE(?,confidence)"
                " WHERE id=?",
                (content, confidence, amu_id))
            self._write(
                "UPDATE amu_fts SET content=?, retrieval_key='' WHERE amu_id=?",
                (content, amu_id))
            self._write("DELETE FROM triples WHERE amu_id=?", (amu_id,))
            row = self.conn.execute("SELECT user_id FROM amu WHERE id=?", (amu_id,)).fetchone()
            if row:
                self._touch_user(row[0])
            self.conn.commit()

    def close_validity(self, amu_id: str, valid_to: str):
        with self._lock:
            current = self.conn.execute("SELECT valid_from,valid_to FROM amu WHERE id=?", (amu_id,)).fetchone()
            if current is None or current["valid_to"] is not None:
                raise ValueError("Cannot close a missing or already closed memory")
            integrity.validate_interval(current["valid_from"], valid_to)
            if not valid_to:
                raise ValueError("Closing validity requires an end time")
            self._archive_claim(amu_id, 'world_change')
            self._write("UPDATE amu SET valid_to=?,valid_to_status='known' WHERE id=?",
                              (valid_to, amu_id))
            row = self.conn.execute("SELECT user_id FROM amu WHERE id=?", (amu_id,)).fetchone()
            if row:
                self._touch_user(row[0])
            self.conn.commit()

    def link_supersession(self, old_id: str, new_id: str):
        with self._lock:
            self._write("UPDATE amu SET superseded_by=? WHERE id=?", (new_id, old_id))
            self.conn.commit()

    def get_amus(self, user_id: str, only_valid: bool = True) -> List[Dict]:
        self._check_user(user_id)
        q = "SELECT * FROM amu WHERE user_id=?"
        if only_valid:
            q += " AND valid_to IS NULL"
        with self._lock:
            rows = self.conn.execute(q, (user_id,)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_by_type(self, user_id: str, types: List[str], include_history=False,
                    include_sensitive=False) -> List[Dict]:
        self._check_user(user_id)
        ph = ",".join("?" * len(types))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM amu WHERE user_id=?"
                + ("" if include_history else " AND valid_to IS NULL")
                + ("" if include_sensitive else " AND sensitivity!='sensitive'")
                + _SUPPRESSED
                + f" AND type IN ({ph})",
                (user_id, *types)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def nearest_many_by_embedding(self, user_id: str, vecs: np.ndarray,
                                  k: int, include_history=False,
                                  scene_ids: Optional[List[str]] = None,
                                  include_cold=False,
                                  include_sensitive=False) -> List[List[Dict]]:
        """Reuse immutable vectors; refilter eligibility in the current snapshot."""
        self._check_user(user_id)
        budget.check()
        queries = np.atleast_2d(vecs)
        if k <= 0 or scene_ids == []:
            return [[] for _ in queries]
        conditions = ("user_id=? AND embedding NOT NULL AND embedding_space=?"
                      + ("" if include_history else " AND valid_to IS NULL")
                      + ("" if include_sensitive else " AND sensitivity!='sensitive'")
                      + _SUPPRESSED
                      + ("" if include_cold else " AND tier!='cold'"))
        params = [user_id, config.EMBEDDING_SPACE]
        if scene_ids is not None:
            conditions += f" AND scene_id IN ({','.join('?' * len(scene_ids))})"
            params.extend(scene_ids)
        with self._lock:
            allowed = [r[0] for r in self.conn.execute('SELECT id FROM amu WHERE ' + conditions, params)]
            if not allowed:
                return [[] for _ in queries]
            generation = self.conn.execute('SELECT generation FROM vector_generations WHERE user_id=?',
                                           (user_id,)).fetchone()
            generation = generation[0] if generation else 0
            epoch = self.user_state(user_id)['epoch']
            key = (self._cache_identity, user_id, config.EMBEDDING_SPACE, epoch, generation)
            def load():
                return self.conn.execute('SELECT id,embedding FROM amu WHERE user_id=? '
                                         'AND embedding_space=? AND embedding NOT NULL ORDER BY id',
                                         (user_id, config.EMBEDDING_SPACE))
            index, cache_hit = vector_index.get(key, load)
            rankings, diagnostics = index.search(queries, allowed, k)
            found_ids = list(dict.fromkeys(mid for ranking in rankings for mid, _ in ranking))
            by_id = {}
            for start in range(0, len(found_ids), 500):
                budget.check()
                ids = found_ids[start:start + 500]
                sql = 'SELECT * FROM amu WHERE ' + conditions + f" AND id IN ({','.join('?' * len(ids))})"
                for row in self.conn.execute(sql, [*params, *ids]):
                    item = self._row_to_dict(row)
                    item.pop('embedding', None)
                    by_id[item['id']] = item
        self.vector_diagnostics = dict(cache_hit=cache_hit, generation=generation, queries=diagnostics)
        return [[dict(by_id[mid], _score=score) for mid, score in ranking if mid in by_id]
                for ranking in rankings]

    def nearest_by_embedding(self, user_id: str, vec: np.ndarray,
                             k: int, include_history=False) -> List[Dict]:
        self._check_user(user_id)
        return self.nearest_many_by_embedding(
            user_id, np.asarray(vec)[None, :], k, include_history)[0]

    def fts_search(self, user_id: str, query: str, k: int, include_history=False,
                   include_sensitive=False, include_cold=False) -> List[Dict]:
        # Share stop words and CJK tokenization with source recall and excerpts.
        self._check_user(user_id)
        from .personal_evidence import terms as query_terms
        import re as _re
        terms = sorted(query_terms(query))[:40]
        if not terms:
            return []
        if _re.search(r"[\u3400-\u9fff]{2}", query):
            clauses = " OR ".join("content LIKE ? OR retrieval_key LIKE ?" for _ in terms)
            params = [value for term in terms for value in (f"%{term}%", f"%{term}%")]
            score = ' + '.join('(content LIKE ? OR COALESCE(retrieval_key,\'\') LIKE ?)' for _ in terms)
            history = "" if include_history else " AND valid_to IS NULL"
            sensitive = "" if include_sensitive else " AND sensitivity!='sensitive'"
            cold = "" if include_cold else " AND tier!='cold'"
            with self._lock:
                rows = self.conn.execute(
                    f"SELECT *, ({score}) AS _score FROM amu WHERE user_id=?"
                    f"{history}{sensitive}{_SUPPRESSED}{cold} AND ({clauses}) ORDER BY _score DESC,id LIMIT ?",
                    (*params, user_id, *params, k)).fetchall()
            out = [self._row_to_dict(r) for r in rows]
            return out
        match = " OR ".join(f'"{t}"' for t in terms)
        with self._lock:
            rows = self.conn.execute(
                """SELECT f.amu_id, bm25(amu_fts) AS rank FROM amu_fts f
                   WHERE amu_fts MATCH ? AND f.user_id=?
                   AND EXISTS (SELECT 1 FROM amu a WHERE a.id=f.amu_id
                     AND (? OR a.valid_to IS NULL) AND (? OR a.sensitivity!='sensitive')
                     AND (? OR a.tier!='cold') AND a.sensitivity!='suppressed'
                     AND a.view_status='ready' AND a.resolution_status!='retracted')
                   ORDER BY rank LIMIT ?""",
                (match, user_id, include_history, include_sensitive, include_cold, k)).fetchall()
        ids = [r["amu_id"] for r in rows]
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        with self._lock:
            amu_rows = self.conn.execute(
                f"SELECT * FROM amu WHERE id IN ({ph})"
                + ("" if include_cold else " AND tier!='cold'")
                + ("" if include_history else " AND valid_to IS NULL")
                + _SUPPRESSED,
                ids).fetchall()
        if not include_sensitive:
            amu_rows = [row for row in amu_rows if row["sensitivity"] != "sensitive"]
        by_id = {r["id"]: self._row_to_dict(r) for r in amu_rows}
        out = []
        for r in rows:
            if r["amu_id"] in by_id:
                d = by_id[r["amu_id"]]
                d["_score"] = -float(r["rank"])  # bm25 lower is better
                out.append(d)
        return out

    def temporal_search(self, user_id, time_scope, k, include_sensitive=False):
        """Recall memories whose event/validity intervals overlap the query scope."""
        self._check_user(user_id)
        start = (time_scope or {}).get("from")
        end = (time_scope or {}).get("to")
        if not start and not end:
            return []
        if end and len(end) == 10:
            end = integrity.resolve_time(end, None).get("end") or end
        start = start or "0000-01-01T00:00:00Z"
        end = end or "9999-12-31T23:59:59Z"
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM amu WHERE user_id=? "
                + ("" if include_sensitive else "AND sensitivity!='sensitive' ") + _SUPPRESSED + " AND "
                "((json_extract(temporal,'$.start') IS NOT NULL AND "
                "julianday(json_extract(temporal,'$.start'))<=julianday(?) AND "
                "julianday(json_extract(temporal,'$.end'))>=julianday(?)) OR "
                "(temporal IS NULL AND event_time IS NOT NULL AND julianday(event_time)>=julianday(?) AND julianday(event_time)<=julianday(?)) OR "
                "(state IS NOT NULL AND state!='null' AND julianday(valid_from)<=julianday(?) AND "
                "(valid_to IS NULL OR julianday(valid_to)>=julianday(?))) OR "
                "(temporal IS NULL AND event_time IS NULL AND COALESCE(valid_from,'0000')<=? AND COALESCE(valid_to,'9999')>=?)) "
                "ORDER BY COALESCE(event_time,valid_from,created_at) DESC LIMIT ?",
                (user_id, end, start, start, end, end, start, end, start, k)).fetchall()
        out = [self._row_to_dict(r) for r in rows]
        for item in out:
            item["_score"] = 1.0
        return out

    def source_search(self, user_id, query, k, include_history=False,
                      include_sensitive=False, include_cold=False):
        """Recall eligible claims by their original, linked source text.

        Joining through amu_sources is essential: historical reads, retractions,
        privacy and user isolation apply even when a source remains indexed.
        """
        self._check_user(user_id)
        from .personal_evidence import terms
        import re
        tokens = sorted(terms(query))[:40]
        if not tokens:
            return []
        rank_args = []
        if re.search(r'[\u3400-\u9fff]{2}', query):
            match_sql = '(' + ' OR '.join('source_fts.content LIKE ?' for _ in tokens) + ')'
            match_args = [f'%{t}%' for t in tokens]
            rank_sql = '-(' + ' + '.join('(source_fts.content LIKE ?)' for _ in tokens) + ')'
            rank_args = match_args
        else:
            match_sql = 'source_fts MATCH ?'
            match_args = [' OR '.join('"' + t.replace('"', '""') + '"' for t in tokens[:40])]
            rank_sql = 'bm25(source_fts)'
        scores = {}
        with self._lock:
            cursor = self.conn.execute(f"""SELECT a.id, {rank_sql} AS rank
                FROM source_fts
                JOIN source_messages s ON s.request_id=source_fts.request_id
                  AND s.message_index=source_fts.message_index
                JOIN amu_sources l ON l.request_id=s.request_id AND l.message_index=s.message_index
                JOIN amu a ON a.id=l.amu_id
                WHERE {match_sql} AND source_fts.user_id=? AND a.user_id=?
                  AND s.user_id=? AND (? OR a.valid_to IS NULL)
                  AND (? OR (a.sensitivity!='sensitive' AND s.sensitivity!='sensitive'))
                  AND a.sensitivity!='suppressed' AND s.sensitivity!='suppressed'
                  AND a.view_status='ready' AND a.resolution_status!='retracted'
                  AND (? OR a.tier!='cold')
                ORDER BY rank, a.id""",
                (*rank_args, *match_args, user_id, user_id, user_id, include_history, include_sensitive,
                 include_cold))
            # Apply the limit to distinct AMUs, not source joins. A memory with
            # hundreds of supporting messages must not crowd out other hits.
            for row in cursor:
                scores.setdefault(row['id'], -float(row['rank']))
                if len(scores) >= k:
                    break
            cursor.close()
        ids = list(scores)[:k]
        found = {m['id']: m for m in self.get_amus_by_ids(ids, include_history, include_sensitive)}
        return [dict(found[mid], _score=scores[mid]) for mid in ids if mid in found]

    def get_by_entities(self, user_id, entities, k):
        self._check_user(user_id)
        wanted = {graph.normalize_entity(e) for e in entities
                  if graph.normalize_entity(e)}
        if not wanted:
            return []
        matches = []
        for item in self.get_amus(user_id):
            actual = {graph.normalize_entity(e) for e in item.get("entities", [])}
            overlap = len(wanted & actual)
            if overlap:
                item["_score"] = float(overlap)
                matches.append(item)
        return sorted(matches, key=lambda item: -item["_score"])[:k]

    # ---------------- triples / graph ----------------
    def insert_triple(self, user_id, subject, relation, object_, amu_id,
                      valid_from=None, valid_to=None):
        self._check_user(user_id)
        integrity.validate_interval(valid_from, valid_to)
        with self._lock:
            self._write(
                "INSERT INTO triples (user_id,subject,relation,object,amu_id,valid_from,valid_to)"
                " VALUES (?,?,?,?,?,?,?)",
                (user_id, subject, relation, object_, amu_id, valid_from, valid_to))
            self._touch_user(user_id)
            self.conn.commit()

    def triples_for_user(self, user_id: str, include_sensitive=False, include_history=True) -> List[Dict]:
        self._check_user(user_id)
        with self._lock:
            rows = self.conn.execute(
                "SELECT t.* FROM triples t JOIN amu a ON a.id=t.amu_id "
                "WHERE t.user_id=? AND a.user_id=t.user_id" +
                ("" if include_history else " AND a.valid_to IS NULL AND t.valid_to IS NULL") +
                ("" if include_sensitive else " AND a.sensitivity!='sensitive'")
                + " AND a.sensitivity!='suppressed' AND a.view_status='ready' AND a.resolution_status!='retracted'",
                (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def graph_rows_for_candidates(self, user_id, ids, include_history=False,
                                  include_sensitive=False, per_candidate=8, limit=512):
        """Read only admitted candidates; each candidate gets a turn at the cap."""
        self._check_user(user_id)
        if not ids or per_candidate <= 0 or limit <= 0:
            return []
        lanes = []
        with self._lock:
            for mid in dict.fromkeys(ids):
                budget.check()
                rows = self.conn.execute(
                    "SELECT t.* FROM triples t INDEXED BY idx_triples_candidate "
                    "JOIN amu a ON a.id=t.amu_id WHERE t.user_id=? AND t.amu_id=? AND a.user_id=?"
                    + ("" if include_history else " AND a.valid_to IS NULL AND t.valid_to IS NULL")
                    + ("" if include_sensitive else " AND a.sensitivity!='sensitive'")
                    + " AND a.sensitivity!='suppressed' AND a.view_status='ready'"
                    " AND a.resolution_status!='retracted' ORDER BY t.id LIMIT ?",
                    (user_id, mid, user_id, min(per_candidate, limit))).fetchall()
                lanes.append([dict(r) for r in rows])
        return [lane[index] for index in range(min(per_candidate, limit))
                for lane in lanes if index < len(lane)][:limit]

    def get_amus_by_ids(self, ids: List[str], include_history=False,
                        include_sensitive=False) -> List[Dict]:
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        scope, params = self._scope_clause()
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM amu WHERE id IN ({ph})"
                + ("" if include_history else " AND valid_to IS NULL")
                + ("" if include_sensitive else " AND sensitivity!='sensitive'")
                + _SUPPRESSED + scope,
                [*ids, *params]).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def core_profile(self, user_id: str, limit: int) -> List[Dict]:
        """MemGPT/MIRIX core memory: the user's active rule/profile/preference
        AMUs, rules first, then stable traits, then recent transient ones."""
        self._check_user(user_id)
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM amu WHERE user_id=? AND valid_to IS NULL"
                " AND sensitivity NOT IN ('sensitive','suppressed')"
                " AND view_status='ready' AND resolution_status='accepted'"
                " AND type IN ('rule','profile','preference')"
                " ORDER BY CASE type WHEN 'rule' THEN 0 WHEN 'profile' THEN 1 ELSE 2 END,"
                " CASE WHEN profile_status IN ('static','stable','rule') THEN 0 ELSE 1 END,"
                " created_at DESC LIMIT ?",
                (user_id, limit * 4)).fetchall()
        items = [self._row_to_dict(r) for r in rows]
        # Cap rules and traits separately so one category cannot starve others.
        # Forget requests are already applied at add time; they carry no trait.
        rules = [a for a in items if a["type"] == "rule"
                 and not _FORGET_RULE_RE.search(str(a.get("content") or ""))][: max(1, limit // 3)]
        traits = [a for a in items if a["type"] != "rule"]
        return (rules + traits)[:limit]

    def suppress(self, amu_id: str, when: str):
        """Zep-style invalidation: close validity and hide from all routes."""
        with self._lock:
            row = self.conn.execute(
                "SELECT user_id,valid_from,valid_to FROM amu WHERE id=?", (amu_id,)).fetchone()
            if row is None:
                raise ValueError("Cannot suppress a missing memory")
            self._archive_claim(amu_id, 'retraction')
            if row["valid_to"] is None:
                integrity.validate_interval(row["valid_from"], when)
                self._write("UPDATE amu SET valid_to=? WHERE id=?", (when, amu_id))
            self._write("UPDATE amu SET sensitivity='suppressed',resolution_status='retracted' WHERE id=?", (amu_id,))
            self._touch_user(row["user_id"])
            self.conn.commit()

    def session_max_timestamp(self, user_id: str, session_id: str) -> Optional[int]:
        self._check_user(user_id)
        with self._lock:
            return self.conn.execute(
                "SELECT MAX(timestamp) FROM source_messages "
                "WHERE user_id=? AND session_id=?", (user_id, session_id)).fetchone()[0]

    # ---------------- sessions ----------------
    def get_summary(self, user_id: str, session_id: str) -> str:
        self._check_user(user_id)
        with self._lock:
            row = self.conn.execute(
                "SELECT summary FROM sessions WHERE user_id=? AND session_id=?",
                (user_id, session_id)).fetchone()
        return row["summary"] if row else ""

    def set_summary(self, user_id: str, session_id: str, summary: str):
        self._check_user(user_id)
        with self._lock:
            self._write(
                "INSERT INTO sessions VALUES (?,?,?)"
                " ON CONFLICT(user_id,session_id) DO UPDATE SET summary=?",
                (user_id, session_id, summary, summary))
            self._write("DELETE FROM sessions_fts WHERE user_id=? AND session_id=?",
                        (user_id, session_id))
            self._write("INSERT INTO sessions_fts VALUES (?,?,?)", (user_id, session_id, summary))
            self._touch_user(user_id)
            self.conn.commit()

    def summary_search(self, user_id, query, k):
        self._check_user(user_id)
        import re
        import hashlib
        cjk = "".join(re.findall(r"[\u3400-\u9fff]", query))
        if len(cjk) >= 2:
            grams = list(dict.fromkeys(cjk[i:i + 2] for i in range(len(cjk) - 1)))[:20]
            clauses = " OR ".join("summary LIKE ?" for _ in grams)
            with self._lock:
                rows = self.conn.execute(
                    f"SELECT user_id,session_id,summary FROM sessions WHERE user_id=? "
                    f"AND ({clauses}) LIMIT ?",
                    (user_id, *(f"%{g}%" for g in grams), k)).fetchall()
            return [self._summary_item(r, sum(g in r["summary"] for g in grams))
                    for r in rows]
        terms = re.findall(r"[0-9A-Za-z]+", query)[:20]
        if not terms:
            return []
        match = " OR ".join('"' + t + '"' for t in terms)
        with self._lock:
            rows = self.conn.execute(
                "SELECT user_id,session_id,summary,bm25(sessions_fts) AS rank "
                "FROM sessions_fts WHERE sessions_fts MATCH ? AND user_id=? "
                "ORDER BY rank LIMIT ?", (match, user_id, k)).fetchall()
        return [self._summary_item(r, -float(r["rank"])) for r in rows]

    @staticmethod
    def _summary_item(row, score):
        import hashlib
        return {"id": "summary_" + hashlib.sha256(
                    json.dumps([row["user_id"], row["session_id"]]).encode()).hexdigest()[:24],
                "user_id": row["user_id"], "session_id": row["session_id"],
                "type": "session_summary", "content": row["summary"],
                "_score": float(score)}

    def sources_for_session(self, user_id, session_id):
        self._check_user(user_id)
        with self._lock:
            return [dict(r) for r in self.conn.execute(
                "SELECT * FROM source_messages WHERE user_id=? AND session_id=? "
                "ORDER BY timestamp,request_id,message_index", (user_id, session_id))]

    # ---------------- util ----------------
    @staticmethod
    def _row_to_dict(r: sqlite3.Row) -> Dict:
        d = dict(r)
        d["entities"] = json.loads(d.get("entities") or "[]")
        d["keywords"] = json.loads(d.get("keywords") or "[]")
        d["support_sessions"] = json.loads(d.get("support_sessions") or "[]")
        for key in ("temporal", "state", "evidence"):
            d[key] = json.loads(d.get(key) or ("[]" if key == "evidence" else "null"))
        if d.get("embedding") is not None:
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
        return d

    # ---------------- ULM lifecycle: time anchor, scenes, heat, forgetting ----
    def latest_time(self, user_id: str) -> Optional[str]:
        """Most recent moment known for a user; anchors relative query times (§5.1)."""
        self._check_user(user_id)
        with self._lock:
            ts = self.conn.execute(
                "SELECT MAX(timestamp) FROM source_messages WHERE user_id=?",
                (user_id,)).fetchone()[0]
            iso = self.conn.execute(
                "SELECT MAX(COALESCE(event_time, valid_from)) FROM amu WHERE user_id=?",
                (user_id,)).fetchone()[0]
        candidates = []
        if ts:
            candidates.append(datetime.fromtimestamp(ts / 1000, tz=timezone.utc))
        if iso:
            try:
                candidates.append(integrity.instant(iso))
            except ValueError:
                pass
        return max(candidates).isoformat() if candidates else None

    def list_scenes(self, user_id: str, include_cold=False) -> List[Dict]:
        self._check_user(user_id)
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM scenes WHERE user_id=?"
                + ("" if include_cold else " AND tier!='cold'"), (user_id,)).fetchall()
        return [self._scene_to_dict(r) for r in rows]

    def get_scenes_by_ids(self, ids: List[str]) -> List[Dict]:
        if not ids:
            return []
        scope, params = self._scope_clause()
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM scenes WHERE id IN ({','.join('?' * len(ids))})" + scope, [*ids, *params]).fetchall()
        return [self._scene_to_dict(r) for r in rows]

    def insert_scene(self, user_id, centroid: np.ndarray, keywords, summary="",
                     surprise=0.0) -> str:
        self._check_user(user_id)
        scene_id = f"scene_{uuid.uuid4().hex[:12]}"
        now = _now()
        with self._lock:
            self._write(
                "INSERT INTO scenes (id,user_id,summary,keywords,centroid,embedding_space,"
                "cell_count,visit_count,interaction_count,surprise,last_access,tier,"
                "created_at,updated_at) VALUES (?,?,?,?,?,?,0,0,0,?,?,'hot',?,?)",
                (scene_id, user_id, summary, json.dumps(sorted(keywords), ensure_ascii=False),
                 centroid.astype(np.float32).tobytes(), config.EMBEDDING_SPACE,
                 surprise, now, now, now))
            self._touch_user(user_id)
            self.conn.commit()
        return scene_id

    def update_scene(self, scene_id, *, centroid=None, keywords=None, summary=None,
                     surprise=None, cells_added=0):
        with self._lock:
            row = self.conn.execute("SELECT user_id FROM scenes WHERE id=?", (scene_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown scene")
            if centroid is not None:
                self._write("UPDATE scenes SET centroid=?,embedding_space=? WHERE id=?",
                            (centroid.astype(np.float32).tobytes(), config.EMBEDDING_SPACE, scene_id))
            if keywords is not None:
                self._write("UPDATE scenes SET keywords=? WHERE id=?",
                            (json.dumps(sorted(keywords), ensure_ascii=False), scene_id))
            if summary is not None:
                self._write("UPDATE scenes SET summary=? WHERE id=?", (summary, scene_id))
            if surprise is not None:
                self._write("UPDATE scenes SET surprise=? WHERE id=?", (surprise, scene_id))
            if cells_added:
                self._write("UPDATE scenes SET cell_count=cell_count+?,"
                            "interaction_count=interaction_count+?,updated_at=? WHERE id=?",
                            (cells_added, cells_added, _now(), scene_id))
            self._touch_user(row[0])
            self.conn.commit()

    def assign_scene(self, amu_ids: List[str], scene_id: str, cell_id: str):
        with self._lock:
            for amu_id in amu_ids:
                self._write("UPDATE amu SET scene_id=?,cell_id=? WHERE id=?",
                            (scene_id, cell_id, amu_id))
            self.conn.commit()

    def scene_cell_ids(self, scene_id: str) -> List[str]:
        scope, params = self._scope_clause()
        with self._lock:
            rows = self.conn.execute(
                "SELECT id FROM amu WHERE scene_id=? AND valid_to IS NULL" + scope, (scene_id, *params)).fetchall()
        return [r[0] for r in rows]

    def reset_scene_interactions(self, scene_id: str, tier: Optional[str] = None):
        with self._lock:
            self._write("UPDATE scenes SET interaction_count=0,promoted_at=? WHERE id=?",
                        (_now(), scene_id))
            if tier:
                self._write("UPDATE scenes SET tier=? WHERE id=?", (tier, scene_id))
                self._write("UPDATE amu SET tier=? WHERE scene_id=? AND type!='rule'",
                            (tier, scene_id))
            self.conn.commit()

    def set_tier(self, amu_ids: List[str], tier: str):
        if not amu_ids:
            return
        with self._lock:
            for amu_id in amu_ids:
                self._write("UPDATE amu SET tier=? WHERE id=?", (tier, amu_id))
            self.conn.commit()

    def add_support_keys(self, amu_id: str, keys: List[str]):
        """Append distinct support keys (session ids, request ids, evidence AMU
        ids); enough support promotes a preference transient -> stable."""
        keys = [k for k in keys if k]
        if not keys:
            return
        with self._lock:
            row = self.conn.execute("SELECT support_sessions FROM amu WHERE id=?",
                                    (amu_id,)).fetchone()
            if row is None:
                return
            sessions = json.loads(row[0] or "[]")
            changed = False
            for key in keys:
                if key not in sessions:
                    sessions.append(key)
                    changed = True
            if changed:
                self._write("UPDATE amu SET support_sessions=? WHERE id=?",
                            (json.dumps(sessions), amu_id))
                if len(sessions) >= config.PROFILE_STABLE_SESSIONS:
                    self._write(
                        "UPDATE amu SET profile_status='stable',expires_at=NULL "
                        "WHERE id=? AND type='preference'", (amu_id,))
                self.conn.commit()

    def add_support_session(self, amu_id: str, session_id: str):
        self.add_support_keys(amu_id, [session_id])

    def record_experience_feedback(self, amu_id: str, success: bool, session_id: str):
        with self._lock:
            column = "helpful" if success else "harmful"
            self._write(f"UPDATE amu SET {column}={column}+1 WHERE id=?", (amu_id,))
            self.add_support_session(amu_id, session_id)
            self.conn.commit()

    def promote_scene_profiles(self, scene_id: str):
        """Materialize stable traits once enough distinct sessions support them."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id,support_sessions FROM amu WHERE scene_id=? "
                "AND type='preference' AND valid_to IS NULL", (scene_id,)).fetchall()
            for row in rows:
                if len(json.loads(row["support_sessions"] or "[]")) >= config.PROFILE_STABLE_SESSIONS:
                    self._write(
                        "UPDATE amu SET profile_status='stable',expires_at=NULL WHERE id=?",
                        (row["id"],))
            self.conn.commit()

    def record_recall(self, amu_ids: List[str], scene_ids: List[str]):
        """Search-side usage counters. Relative updates only and no revision bump,
        so a concurrent staged Add of the same user is never invalidated."""
        if not amu_ids and not scene_ids:
            return
        now = _now()
        with self._lock:
            for amu_id in amu_ids:
                self.conn.execute(
                    "UPDATE amu SET recall_count=recall_count+1,last_recalled=?,"
                    "strength=strength+?,tier=CASE WHEN tier='cold' THEN 'hot' ELSE tier END"
                    " WHERE id=?", (now, config.FORGET_RECALL_BONUS_DAYS /
                                     config.FORGET_STRENGTH_DAYS, amu_id))
            for scene_id in scene_ids:
                self.conn.execute(
                    "UPDATE scenes SET visit_count=visit_count+1,last_access=?,tier='hot' WHERE id=?",
                    (now, scene_id))
            self.conn.commit()

    def forgettable(self, user_id: str) -> List[Dict]:
        """Hot, non-rule memories with the fields the retention formula needs."""
        self._check_user(user_id)
        with self._lock:
            rows = self.conn.execute(
                "SELECT id,type,created_at,last_recalled,strength FROM amu "
                "WHERE user_id=? AND tier='hot' AND type NOT IN ('rule','profile')",
                (user_id,)).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _scene_to_dict(r: sqlite3.Row) -> Dict:
        d = dict(r)
        d["keywords"] = json.loads(d.get("keywords") or "[]")
        if d.get("centroid") is not None:
            d["centroid"] = np.frombuffer(d["centroid"], dtype=np.float32)
        return d


def amus_embeddings(amus: List[Dict]) -> List[np.ndarray]:
    return [a["embedding"] for a in amus]
