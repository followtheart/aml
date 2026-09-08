"""SQLite storage layer.

Tables:
  amu          -- atomic memory units (with embedding blob, validity interval)
  triples      -- (subject, relation, object, amu_id) knowledge graph edges
  sessions     -- rolling session summaries
  requests     -- Add idempotency ledger
  amu_fts      -- FTS5 mirror of amu.content + retrieval_key (BM25)

Swap target for production: Postgres + pgvector (same method surface).
"""
import json
import asyncio
from contextlib import contextmanager
import sqlite3
import threading
import uuid
import weakref
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from . import config, graph

_LOCK = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS amu (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  content TEXT NOT NULL,
  retrieval_key TEXT,
  type TEXT NOT NULL DEFAULT 'fact',
  entities TEXT NOT NULL DEFAULT '[]',
  keywords TEXT NOT NULL DEFAULT '[]',
  event_time TEXT,
  valid_from TEXT,
  valid_to TEXT,
  supersedes TEXT,
  confidence REAL DEFAULT 0.9,
  sensitivity TEXT DEFAULT 'normal',
  embedding BLOB,
  embedding_space TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_amu_user ON amu(user_id);
CREATE INDEX IF NOT EXISTS idx_amu_user_type ON amu(user_id, type);

CREATE TABLE IF NOT EXISTS triples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  subject TEXT NOT NULL,
  relation TEXT NOT NULL,
  object TEXT NOT NULL,
  amu_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_triples_user ON triples(user_id);
CREATE INDEX IF NOT EXISTS idx_triples_ent ON triples(user_id, subject);

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
INSERT INTO sessions_fts(user_id,session_id,summary)
 SELECT s.user_id,s.session_id,s.summary FROM sessions s
 WHERE NOT EXISTS (SELECT 1 FROM sessions_fts f
 WHERE f.user_id=s.user_id AND f.session_id=s.session_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Optional[str] = None):
        self.path = path or config.DB_PATH
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._add_locks = weakref.WeakValueDictionary()
        with _LOCK:
            self.conn.execute("PRAGMA busy_timeout=5000")
            if self.path != ":memory:":
                self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            # Forward-compatible migration for databases created before vector
            # spaces were recorded. NULL means "unknown" and is not searched.
            columns = {r["name"] for r in self.conn.execute("PRAGMA table_info(amu)")}
            if "embedding_space" not in columns:
                self.conn.execute("ALTER TABLE amu ADD COLUMN embedding_space TEXT")
            self.conn.executescript(FTS_SCHEMA)
            self.conn.executescript(SOURCE_SCHEMA)
            self.conn.commit()

    def _write(self, sql, params=()):
        result = self.conn.execute(sql, params)
        if hasattr(self, "_pending"):
            self._pending.append((sql, params))
        return result

    def _touch_user(self, user_id):
        """Advance a live user's revision; staged stores advance on publish."""
        if hasattr(self, "_pending"):
            return
        self.conn.execute(
            "INSERT INTO user_revisions(user_id,revision) VALUES (?,1) "
            "ON CONFLICT(user_id) DO UPDATE SET revision=revision+1", (user_id,))

    def _copy_user_to(self, work, user_id):
        """Copy only one user's state; avoids an O(database) Add snapshot."""
        def copy(table, where, params):
            rows = self.conn.execute(f"SELECT * FROM {table} WHERE {where}", params).fetchall()
            if not rows:
                return
            columns = rows[0].keys()
            placeholders = ",".join("?" for _ in columns)
            names = ",".join(columns)
            work.conn.executemany(
                f"INSERT INTO {table}({names}) VALUES ({placeholders})",
                ([row[name] for name in columns] for row in rows))

        copy("amu", "user_id=?", (user_id,))
        copy("triples", "user_id=?", (user_id,))
        copy("sessions", "user_id=?", (user_id,))
        copy("requests", "user_id=?", (user_id,))
        copy("source_messages", "user_id=?", (user_id,))
        copy("amu_sources", "amu_id IN (SELECT id FROM amu WHERE user_id=?)", (user_id,))
        copy("amu_fts", "user_id=?", (user_id,))
        copy("sessions_fts", "user_id=?", (user_id,))
        work.conn.commit()

    @contextmanager
    def staged(self, user_id):
        """Prepare one user's Add privately and publish it in a short transaction.

        Different users can prepare and commit independently. A same-user change
        is detected through a per-user revision and must be retried.
        """
        work = Store(":memory:")
        try:
            with _LOCK:
                self.conn.execute("BEGIN")
                try:
                    revision_row = self.conn.execute(
                        "SELECT revision FROM user_revisions WHERE user_id=?", (user_id,)
                    ).fetchone()
                    revision = revision_row[0] if revision_row else 0
                    self._copy_user_to(work, user_id)
                finally:
                    self.conn.rollback()
            work._pending = []
            yield work
            with _LOCK:
                self.conn.execute("BEGIN IMMEDIATE")
                try:
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
                except BaseException:
                    self.conn.rollback()
                    raise
        finally:
            work.conn.close()

    def save_messages(self, req):
        for i, message in enumerate(req.messages):
            self._write("INSERT INTO source_messages VALUES (?,?,?,?,?,?,?)",
                        (req.request_id, i, req.user_id, req.session_id,
                         message.role, message.content, message.timestamp))
        self._touch_user(req.user_id)
        self.conn.commit()

    def link_sources(self, amu_id, request_id, indices):
        for i in indices:
            self._write("INSERT OR IGNORE INTO amu_sources VALUES (?,?,?)",
                        (amu_id, request_id, i))
        row = self.conn.execute("SELECT user_id FROM amu WHERE id=?", (amu_id,)).fetchone()
        if row:
            self._touch_user(row[0])
        self.conn.commit()

    def sources_for_amu(self, amu_id):
        return [dict(r) for r in self.conn.execute(
            "SELECT m.* FROM source_messages m JOIN amu_sources s "
            "ON m.request_id=s.request_id AND m.message_index=s.message_index "
            "WHERE s.amu_id=? ORDER BY m.request_id,m.message_index", (amu_id,))]

    def replace_fact(self, amu_id, fact, embedding):
        """Replace derived representations together; caller supplies final-text vector."""
        self._write("UPDATE amu SET content=?,retrieval_key=?,type=?,entities=?,"
                    "keywords=?,event_time=?,sensitivity=?,embedding=?,"
                    "embedding_space=? WHERE id=?",
                    (fact["content"], fact.get("retrieval_key", ""),
                     fact.get("type", "fact"), json.dumps(fact.get("entities") or []),
                     json.dumps(fact.get("keywords") or []), fact.get("event_time"),
                     fact.get("sensitivity", "normal"),
                     embedding.astype(np.float32).tobytes(), config.EMBEDDING_SPACE,
                     amu_id))
        self._write("UPDATE amu_fts SET content=?,retrieval_key=? WHERE amu_id=?",
                    (fact["content"], fact.get("retrieval_key", ""), amu_id))
        self._write("DELETE FROM triples WHERE amu_id=?", (amu_id,))
        row = self.conn.execute("SELECT user_id FROM amu WHERE id=?", (amu_id,)).fetchone()
        if row:
            self._touch_user(row[0])
        self.conn.commit()

    # ---------------- idempotency ----------------
    def request_seen(self, request_id: str) -> bool:
        with _LOCK:
            row = self.conn.execute(
                "SELECT 1 FROM requests WHERE request_id=?",
                (request_id,)).fetchone()
            return row is not None

    def record_request(self, request_id: str, user_id: str, session_id: str):
        with _LOCK:
            self._write(
                "INSERT OR IGNORE INTO requests VALUES (?,?,?,?)",
                (request_id, user_id, session_id, _now()))
            self._touch_user(user_id)
            self.conn.commit()

    def add_lock(self, user_id):
        """Serialize same-user Adds per process while allowing other users."""
        key = (id(asyncio.get_running_loop()), user_id)
        return self._add_locks.setdefault(key, asyncio.Lock())

    # ---------------- AMU CRUD ----------------
    def insert_amu(self, *, user_id, session_id, content, retrieval_key="",
                   type="fact", entities=None, keywords=None, event_time=None,
                   valid_from=None, valid_to=None, supersedes=None,
                   confidence=0.9, sensitivity="normal",
                   embedding: Optional[np.ndarray] = None) -> str:
        amu_id = f"amu_{uuid.uuid4().hex[:16]}"
        blob = (embedding.astype(np.float32).tobytes()
                if embedding is not None else None)
        with _LOCK:
            self._write(
                """INSERT INTO amu (id,user_id,session_id,content,retrieval_key,
                   type,entities,keywords,event_time,valid_from,valid_to,
                   supersedes,confidence,sensitivity,embedding,embedding_space,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (amu_id, user_id, session_id, content, retrieval_key, type,
                 json.dumps(entities or [], ensure_ascii=False),
                 json.dumps(keywords or [], ensure_ascii=False),
                 event_time, valid_from, valid_to, supersedes, confidence,
                 sensitivity, blob,
                 config.EMBEDDING_SPACE if blob is not None else None, _now()))
            self._write(
                "INSERT INTO amu_fts (amu_id,user_id,content,retrieval_key)"
                " VALUES (?,?,?,?)",
                (amu_id, user_id, content, retrieval_key))
            self._touch_user(user_id)
            self.conn.commit()
        return amu_id

    def update_amu_content(self, amu_id: str, content: str,
                           confidence: Optional[float] = None):
        with _LOCK:
            self._write(
                "UPDATE amu SET content=?, embedding=NULL, embedding_space=NULL, retrieval_key='', "
                "entities='[]', keywords='[]', confidence=COALESCE(?,confidence)"
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
        with _LOCK:
            self._write("UPDATE amu SET valid_to=? WHERE id=?",
                              (valid_to, amu_id))
            row = self.conn.execute("SELECT user_id FROM amu WHERE id=?", (amu_id,)).fetchone()
            if row:
                self._touch_user(row[0])
            self.conn.commit()

    def get_amus(self, user_id: str, only_valid: bool = True) -> List[Dict]:
        q = "SELECT * FROM amu WHERE user_id=?"
        if only_valid:
            q += " AND valid_to IS NULL"
        with _LOCK:
            rows = self.conn.execute(q, (user_id,)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_by_type(self, user_id: str, types: List[str], include_history=False) -> List[Dict]:
        ph = ",".join("?" * len(types))
        with _LOCK:
            rows = self.conn.execute(
                f"SELECT * FROM amu WHERE user_id=?"
                + ("" if include_history else " AND valid_to IS NULL")
                + f" AND type IN ({ph})",
                (user_id, *types)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def nearest_many_by_embedding(self, user_id: str, vecs: np.ndarray,
                                  k: int, include_history=False) -> List[List[Dict]]:
        """Load a user's matrix once and rank any number of query vectors."""
        with _LOCK:
            rows = self.conn.execute(
                "SELECT id,user_id,session_id,content,retrieval_key,type,"
                "event_time,valid_from,valid_to,created_at,embedding FROM amu"
                " WHERE user_id=? AND embedding NOT NULL AND embedding_space=?"
                + ("" if include_history else " AND valid_to IS NULL"),
                (user_id, config.EMBEDDING_SPACE)).fetchall()
        if not rows:
            return [[] for _ in vecs]
        mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32)
                        for r in rows])
        queries = np.asarray(vecs, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        if mat.shape[1] != queries.shape[1]:
            raise ValueError("Stored and query embedding dimensions differ")
        scores = queries @ mat.T
        results = []
        for query_scores in scores:
            out = []
            for i in np.argsort(-query_scores)[:k]:
                d = {c: rows[int(i)][c] for c in rows[int(i)].keys()
                     if c != "embedding"}
                d["_score"] = float(query_scores[int(i)])
                out.append(d)
            results.append(out)
        return results

    def nearest_by_embedding(self, user_id: str, vec: np.ndarray,
                             k: int, include_history=False) -> List[Dict]:
        return self.nearest_many_by_embedding(
            user_id, np.asarray(vec)[None, :], k, include_history)[0]

    def fts_search(self, user_id: str, query: str, k: int, include_history=False) -> List[Dict]:
        # OR semantics keeps recall high for keyword-ish queries.
        # FTS5 MATCH is syntax-sensitive: keep only alnum tokens, quote each.
        import re as _re
        cjk = "".join(_re.findall(r"[\u3400-\u9fff]", query))
        if len(cjk) >= 2:
            grams = list(dict.fromkeys(cjk[i:i + 2] for i in range(len(cjk) - 1)))[:20]
            clauses = " OR ".join("content LIKE ? OR retrieval_key LIKE ?" for _ in grams)
            params = [value for gram in grams for value in (f"%{gram}%", f"%{gram}%")]
            history = "" if include_history else " AND valid_to IS NULL"
            with _LOCK:
                rows = self.conn.execute(
                    f"SELECT * FROM amu WHERE user_id=?{history} AND ({clauses}) LIMIT ?",
                    (user_id, *params, k)).fetchall()
            out = [self._row_to_dict(r) for r in rows]
            for item in out:
                item["_score"] = float(sum(
                    gram in (item["content"] + " " + (item.get("retrieval_key") or ""))
                    for gram in grams))
            return sorted(out, key=lambda item: -item["_score"])
        terms = [t for t in (_re.sub(r"[^0-9A-Za-z]", "", t)
                             for t in query.split()) if len(t) > 1]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms[:20])
        with _LOCK:
            rows = self.conn.execute(
                """SELECT f.amu_id, bm25(amu_fts) AS rank FROM amu_fts f
                   WHERE amu_fts MATCH ? AND f.user_id=?
                   AND (? OR EXISTS (SELECT 1 FROM amu a WHERE a.id=f.amu_id AND a.valid_to IS NULL))
                   ORDER BY rank LIMIT ?""",
                (match, user_id, include_history, k)).fetchall()
        ids = [r["amu_id"] for r in rows]
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        with _LOCK:
            amu_rows = self.conn.execute(
                f"SELECT * FROM amu WHERE id IN ({ph})"
                + ("" if include_history else " AND valid_to IS NULL"),
                ids).fetchall()
        by_id = {r["id"]: self._row_to_dict(r) for r in amu_rows}
        out = []
        for r in rows:
            if r["amu_id"] in by_id:
                d = by_id[r["amu_id"]]
                d["_score"] = -float(r["rank"])  # bm25 lower is better
                out.append(d)
        return out

    def temporal_search(self, user_id, time_scope, k):
        """Recall memories whose event/validity intervals overlap the query scope."""
        start = (time_scope or {}).get("from")
        end = (time_scope or {}).get("to")
        if not start and not end:
            return []
        start = start or "0000-01-01T00:00:00Z"
        end = end or "9999-12-31T23:59:59Z"
        with _LOCK:
            rows = self.conn.execute(
                "SELECT * FROM amu WHERE user_id=? AND "
                "((event_time IS NOT NULL AND event_time>=? AND event_time<=?) OR "
                "(COALESCE(valid_from,'0000')<=? AND COALESCE(valid_to,'9999')>=?)) "
                "ORDER BY COALESCE(event_time,valid_from,created_at) DESC LIMIT ?",
                (user_id, start, end, end, start, k)).fetchall()
        out = [self._row_to_dict(r) for r in rows]
        for item in out:
            item["_score"] = 1.0
        return out

    def get_by_entities(self, user_id, entities, k):
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
    def insert_triple(self, user_id, subject, relation, object_, amu_id):
        with _LOCK:
            self._write(
                "INSERT INTO triples (user_id,subject,relation,object,amu_id)"
                " VALUES (?,?,?,?,?)",
                (user_id, subject, relation, object_, amu_id))
            self._touch_user(user_id)
            self.conn.commit()

    def triples_for_user(self, user_id: str) -> List[Dict]:
        with _LOCK:
            rows = self.conn.execute(
                "SELECT * FROM triples WHERE user_id=?", (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_amus_by_ids(self, ids: List[str], include_history=False) -> List[Dict]:
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        with _LOCK:
            rows = self.conn.execute(
                f"SELECT * FROM amu WHERE id IN ({ph})"
                + ("" if include_history else " AND valid_to IS NULL"),
                ids).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ---------------- sessions ----------------
    def get_summary(self, user_id: str, session_id: str) -> str:
        with _LOCK:
            row = self.conn.execute(
                "SELECT summary FROM sessions WHERE user_id=? AND session_id=?",
                (user_id, session_id)).fetchone()
        return row["summary"] if row else ""

    def set_summary(self, user_id: str, session_id: str, summary: str):
        with _LOCK:
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
        import re
        import hashlib
        cjk = "".join(re.findall(r"[\u3400-\u9fff]", query))
        if len(cjk) >= 2:
            grams = list(dict.fromkeys(cjk[i:i + 2] for i in range(len(cjk) - 1)))[:20]
            clauses = " OR ".join("summary LIKE ?" for _ in grams)
            with _LOCK:
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
        with _LOCK:
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
        with _LOCK:
            return [dict(r) for r in self.conn.execute(
                "SELECT * FROM source_messages WHERE user_id=? AND session_id=? "
                "ORDER BY timestamp,request_id,message_index", (user_id, session_id))]

    # ---------------- util ----------------
    @staticmethod
    def _row_to_dict(r: sqlite3.Row) -> Dict:
        d = dict(r)
        d["entities"] = json.loads(d.get("entities") or "[]")
        d["keywords"] = json.loads(d.get("keywords") or "[]")
        if d.get("embedding") is not None:
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
        return d


def amus_embeddings(amus: List[Dict]) -> List[np.ndarray]:
    return [a["embedding"] for a in amus]
