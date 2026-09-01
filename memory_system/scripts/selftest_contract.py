"""Contract self-test (design doc §10.2): idempotency, ID echo, schema,
auth, empty search, chunk replay, error injection.

Self-contained: forces offline Fake mode and its own temp DB, so you can
just run `python scripts/selftest_contract.py` with no env vars and no keys.
"""
import os
import sys
import tempfile

# Force hermetic offline settings BEFORE importing the app
os.environ.setdefault("AML_FAKE", "1")
os.environ["AML_FAKE"] = "1"
os.environ.setdefault("AML_API_KEY", "testkey")
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.unlink(_tmp.name)  # fresh db every run
os.environ["AML_DB_PATH"] = _tmp.name

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
H = {"Authorization": "Bearer " + os.environ.get("AML_API_KEY", "testkey")}

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"[{PASS if cond else FAIL}] {name} {detail}")


add_body = {
    "request_id": "eval:run_t:locomo:conv-0:chunk-0",
    "messages": [
        {"role": "user", "content": "I went to a LGBTQ support group yesterday and it was so powerful.", "timestamp": 1683310800000},
        {"role": "assistant", "content": "That sounds meaningful. What stood out to you?"},
        {"role": "user", "content": "The sense of community. I also started photography last month."},
    ],
    "user_id": "eval:run_t:locomo:conv-0",
    "session_id": "eval:run_t:sample:0",
}

# 1. auth rejection
r = client.post("/add", json=add_body)
check("auth 401 without token", r.status_code == 401)

# 2. happy path add
r = client.post("/add", json=add_body, headers=H)
ok = (r.status_code == 200 and r.json().get("success") is True
      and r.json()["request_id"] == add_body["request_id"]
      and r.json()["user_id"] == add_body["user_id"]
      and r.json()["session_id"] == add_body["session_id"])
check("add 200 + byte-exact id echo", ok, r.text[:120])

# 3. idempotent replay
r2 = client.post("/add", json=add_body, headers=H)
check("add idempotent replay", r2.status_code == 200)

# 4. search happy path
sbody = {"query": "When did the user start photography?",
         "user_id": add_body["user_id"], "top_k": 100}
r = client.post("/search", json=sbody, headers=H)
items = r.json().get("data", []) if r.status_code == 200 else []
check("search 200 + data array", r.status_code == 200 and isinstance(items, list))
check("search returns relevant memory",
      any("photography" in i["content"] for i in items),
      (items[0]["content"][:80] if items else "EMPTY"))
check("item schema id/content", all(i.get("id") and i.get("content") for i in items))

# 5. isolation: other user sees nothing
r = client.post("/search", json={**sbody, "user_id": "eval:other"}, headers=H)
check("user_id isolation -> empty data", r.json()["data"] == [])

# 6. unanswerable query -> empty or low noise (abstention tolerated)
r = client.post("/search", json={**sbody, "query": "quantum chromodynamics lattice gauge"}, headers=H)
check("search never errors on OOD query", r.status_code == 200)

# 7. chunk replay sequence (platform splits >20 msgs / >2000 words)
msgs = [{"role": "user" if i % 2 == 0 else "assistant",
         "content": f"message {i} about topic {i % 3}"} for i in range(24)]
ok = True
for c in range(2):
    chunk = {"request_id": f"eval:run_t:locomo:conv-1:chunk-{c}",
             "messages": msgs[c * 12:(c + 1) * 12],
             "user_id": "eval:run_t:locomo:conv-1",
             "session_id": "eval:run_t:sample:1"}
    rr = client.post("/add", json=chunk, headers=H)
    ok = ok and rr.status_code == 200
check("chunked session adds", ok)

# 8. validation error 422 (not retried by platform — must not 500)
r = client.post("/add", json={"request_id": "x"}, headers=H)
check("bad request -> 422", r.status_code == 422)

# 9. health unauthenticated
r = client.get("/health")
check("health 200 no auth", r.status_code == 200)

failed = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
