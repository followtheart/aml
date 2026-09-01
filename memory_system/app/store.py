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
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from . import config

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
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS amu_fts USING fts5(
  amu_id UNINDEXED, user_id UNINDEXED, content, retrieval_key
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Optional[str] = None):
        self.path = path or config.DB_PATH
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with _LOCK:
            self.conn.executescript(SCHEMA)
            self.conn.executescript(FTS_SCHEMA)
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
            self.conn.execute(
                "INSERT OR IGNORE INTO requests VALUES (?,?,?,?)",
                (request_id, user_id, session_id, _now()))
            self.conn.commit()

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
            self.conn.execute(
                """INSERT INTO amu (id,user_id,session_id,content,retrieval_key,
                   type,entities,keywords,event_time,valid_from,valid_to,
                   supersedes,confidence,sensitivity,embedding,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (amu_id, user_id, session_id, content, retrieval_key, type,
                 json.dumps(entities or [], ensure_ascii=False),
                 json.dumps(keywords or [], ensure_ascii=False),
                 event_time, valid_from, valid_to, supersedes, confidence,
                 sensitivity, blob, _now()))
            self.conn.execute(
                "INSERT INTO amu_fts (amu_id,user_id,content,retrieval_key)"
                " VALUES (?,?,?,?)",
                (amu_id, user_id, content, retrieval_key))
            self.conn.commit()
        return amu_id

    def update_amu_content(self, amu_id: str, content: str,
                           confidence: Optional[float] = None):
        with _LOCK:
            self.conn.execute(
                "UPDATE amu SET content=?, confidence=COALESCE(?,confidence)"
                " WHERE id=?",
                (content, confidence, amu_id))
            self.conn.execute(
                "UPDATE amu_fts SET content=? WHERE amu_id=?",
                (content, amu_id))
            self.conn.commit()

    def close_validity(self, amu_id: str, valid_to: str):
        with _LOCK:
            self.conn.execute("UPDATE amu SET valid_to=? WHERE id=?",
                              (valid_to, amu_id))
            self.conn.commit()

    def get_amus(self, user_id: str, only_valid: bool = True) -> List[Dict]:
        q = "SELECT * FROM amu WHERE user_id=?"
        if only_valid:
            q += " AND valid_to IS NULL"
        with _LOCK:
            rows = self.conn.execute(q, (user_id,)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_by_type(self, user_id: str, types: List[str]) -> List[Dict]:
        ph = ",".join("?" * len(types))
        with _LOCK:
            rows = self.conn.execute(
                f"SELECT * FROM amu WHERE user_id=? AND valid_to IS NULL"
                f" AND type IN ({ph})",
                (user_id, *types)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def nearest_by_embedding(self, user_id: str, vec: np.ndarray,
                             k: int) -> List[Dict]:
        with _LOCK:
            rows = self.conn.execute(
                "SELECT id,user_id,session_id,content,retrieval_key,type,"
                "valid_from,valid_to,created_at,embedding FROM amu"
                " WHERE user_id=? AND valid_to IS NULL AND embedding NOT NULL",
                (user_id,)).fetchall()
        if not rows:
            return []
        ids = [r["id"] for r in rows]
        mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32)
                        for r in rows])
        scores = mat @ vec
        idx = np.argsort(-scores)[:k]
        out = []
        for i in idx:
            d = {c: rows[int(i)][c] for c in rows[int(i)].keys()
                 if c != "embedding"}
            d["_score"] = float(scores[int(i)])
            out.append(d)
        return out

    def fts_search(self, user_id: str, query: str, k: int) -> List[Dict]:
        # OR semantics keeps recall high for keyword-ish queries.
        # FTS5 MATCH is syntax-sensitive: keep only alnum tokens, quote each.
        import re as _re
        terms = [t for t in (_re.sub(r"[^0-9A-Za-z一-鿿]", "", t)
                             for t in query.split()) if len(t) > 1]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms[:20])
        with _LOCK:
            rows = self.conn.execute(
                """SELECT f.amu_id, bm25(amu_fts) AS rank FROM amu_fts f
                   WHERE amu_fts MATCH ? AND f.user_id=?
                   ORDER BY rank LIMIT ?""",
                (match, user_id, k)).fetchall()
        ids = [r["amu_id"] for r in rows]
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        with _LOCK:
            amu_rows = self.conn.execute(
                f"SELECT * FROM amu WHERE id IN ({ph}) AND valid_to IS NULL",
                ids).fetchall()
        by_id = {r["id"]: self._row_to_dict(r) for r in amu_rows}
        out = []
        for r in rows:
            if r["amu_id"] in by_id:
                d = by_id[r["amu_id"]]
                d["_score"] = -float(r["rank"])  # bm25 lower is better
                out.append(d)
        return out

    # ---------------- triples / graph ----------------
    def insert_triple(self, user_id, subject, relation, object_, amu_id):
        with _LOCK:
            self.conn.execute(
                "INSERT INTO triples (user_id,subject,relation,object,amu_id)"
                " VALUES (?,?,?,?,?)",
                (user_id, subject, relation, object_, amu_id))
            self.conn.commit()

    def triples_for_user(self, user_id: str) -> List[Dict]:
        with _LOCK:
            rows = self.conn.execute(
                "SELECT * FROM triples WHERE user_id=?", (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_amus_by_ids(self, ids: List[str]) -> List[Dict]:
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        with _LOCK:
            rows = self.conn.execute(
                f"SELECT * FROM amu WHERE id IN ({ph}) AND valid_to IS NULL",
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
            self.conn.execute(
                "INSERT INTO sessions VALUES (?,?,?)"
                " ON CONFLICT(user_id,session_id) DO UPDATE SET summary=?",
                (user_id, session_id, summary, summary))
            self.conn.commit()

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
