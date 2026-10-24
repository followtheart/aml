"""Transactional SQLite baseline with tenant-scoped snapshots and purge epochs."""
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class Conflict(ValueError):
    pass


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Store:
    def __init__(self, path, space_id):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tenants(user_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, epoch INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS requests(user_id TEXT, request_id TEXT, hash TEXT NOT NULL, response TEXT NOT NULL,
                PRIMARY KEY(user_id, request_id));
            CREATE TABLE IF NOT EXISTS events(user_id TEXT, event_id TEXT, session_id TEXT, logical_id TEXT,
                valid_from TEXT, valid_to TEXT, payload TEXT NOT NULL, PRIMARY KEY(user_id,event_id));
            CREATE INDEX IF NOT EXISTS event_owner ON events(user_id,session_id,logical_id);
        """)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('space',?)", (space_id,))
        if self.db.execute("SELECT value FROM metadata WHERE key='space'").fetchone()[0] != space_id:
            self.db.close()
            raise Conflict("database embedding space mismatch; rebuild into a separate database")

    def close(self):
        with self.lock:
            self.db.close()

    def state(self, user):
        with self.lock:
            row = self.db.execute("SELECT revision,epoch FROM tenants WHERE user_id=?", (user,)).fetchone()
            return tuple(row) if row else (0, 0)

    def replay(self, req):
        with self.lock:
            row = self.db.execute("SELECT hash,response FROM requests WHERE user_id=? AND request_id=?",
                                  (req.user_id, req.request_id)).fetchone()
            if not row:
                return None
            if row["hash"] != digest(req.model_dump()):
                raise Conflict("request_id reused with different payload")
            return json.loads(row["response"])

    def commit(self, req, events, expected_epoch, degraded):
        with self.lock, self.db:
            # Acquire the SQLite write lock before replay/state checks (also across processes).
            self.db.execute("BEGIN IMMEDIATE")
            replay = self.replay(req)
            if replay:
                return replay
            revision, epoch = self.state(req.user_id)
            if epoch != expected_epoch:
                raise Conflict("user purged while Add was preparing; retry explicitly")
            revision += 1
            for event in events:
                previous = self.db.execute(
                    "SELECT event_id,payload,valid_from FROM events WHERE user_id=? AND session_id=? AND logical_id=? AND valid_to IS NULL",
                    (req.user_id, req.session_id, event["logical_id"])).fetchone()
                if previous:
                    old = json.loads(previous["payload"])
                    if event["valid_from"] <= previous["valid_from"]:
                        raise Conflict("replacement must have a later valid_from")
                    event["version"] = old["version"] + 1
                    old["valid_to"] = event["valid_from"]
                    self.db.execute("UPDATE events SET valid_to=?,payload=? WHERE user_id=? AND event_id=?",
                                    (old["valid_to"], json.dumps(old), req.user_id, old["event_id"]))
                event["revision"] = revision
                self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                                (req.user_id, event["event_id"], req.session_id, event["logical_id"],
                                 event["valid_from"], None, json.dumps(event, ensure_ascii=False)))
            self.db.execute("INSERT INTO tenants VALUES (?,?,?) ON CONFLICT(user_id) DO UPDATE SET revision=excluded.revision",
                            (req.user_id, revision, epoch))
            response = {"success": True, "request_id": req.request_id, "user_id": req.user_id,
                        "session_id": req.session_id, "write_revision": revision,
                        "scope_epoch": epoch, "degradation": degraded}
            self.db.execute("INSERT INTO requests VALUES (?,?,?,?)",
                            (req.user_id, req.request_id, digest(req.model_dump()), json.dumps(response)))
            return response

    def snapshot(self, req):
        with self.lock:
            # One read transaction ties revision/epoch to exactly these rows.
            self.db.execute("BEGIN")
            try:
                revision, epoch = self.state(req.user_id)
                if revision < req.min_revision:
                    raise Conflict("requested revision is not available")
                rows = self.db.execute("SELECT payload FROM events WHERE user_id=?", (req.user_id,)).fetchall()
                events = []
                for row in rows:
                    event = json.loads(row[0])
                    cutoff = req.as_of or datetime.now(timezone.utc).isoformat()
                    if req.as_of or not req.include_history:
                        if event["valid_from"] > cutoff or (event["valid_to"] and cutoff >= event["valid_to"]):
                            continue
                    sensitivities = {s["sensitivity"] for s in event["sources"]}
                    if "suppressed" in sensitivities or (not req.include_sensitive and "sensitive" in sensitivities):
                        continue
                    events.append(event)
                return revision, epoch, events
            finally:
                self.db.rollback()

    def purge(self, user):
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            revision, epoch = self.state(user)
            self.db.execute("DELETE FROM events WHERE user_id=?", (user,))
            self.db.execute("DELETE FROM requests WHERE user_id=?", (user,))
            self.db.execute("INSERT INTO tenants VALUES (?,?,?) ON CONFLICT(user_id) DO UPDATE SET revision=excluded.revision,epoch=excluded.epoch",
                            (user, revision + 1, epoch + 1))
            return {"success": True, "user_id": user, "write_revision": revision + 1, "scope_epoch": epoch + 1}
