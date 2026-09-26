"""Contract self-test (design doc §10.2): idempotency, ID echo, schema,
auth, empty search, chunk replay, error injection.

Self-contained: forces offline Fake mode and its own temp DB, so you can
just run `python scripts/selftest_contract.py` with no env vars and no keys.
"""
import os
import sys
import tempfile
from unittest.mock import AsyncMock, patch

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
from app import add_pipeline, evidence_packet

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
# ULM §2.5: Add returns the committed write revision; a replay reports the
# same revision without advancing it.
check("add returns write_revision",
      r.json().get("write_revision", 0) >= 1 and "scope_epoch" in r.json())
check("idempotent replay keeps write_revision",
      r2.json().get("write_revision") == r.json().get("write_revision"))

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

# Strict external schemas, independent of internal convenience defaults.
r = client.post('/search', json={k: v for k, v in sbody.items() if k != 'top_k'}, headers=H)
check('missing top_k -> 422', r.status_code == 422)
r = client.post('/add', json={**add_body, 'request_id': 'invalid-role',
    'messages': [{'role': 'system', 'content': 'invalid role'}]}, headers=H)
check('invalid message role -> 422', r.status_code == 422)
for field in ('user_id', 'session_id'):
    r = client.post('/add', json={**add_body, field: 'different'}, headers=H)
    check(f'conflicting request {field} -> 422', r.status_code == 422)
source = {**add_body, 'request_id': 'source-identity',
          'messages': [{'role': 'user', 'content': 'I like tea.', 'message_id': 'stable-message'}]}
check('explicit source identity accepted', client.post('/add', json=source, headers=H).status_code == 200)
r = client.post('/add', json={**source, 'request_id': 'source-conflict',
    'messages': [{**source['messages'][0], 'content': 'I like coffee.'}]}, headers=H)
check('conflicting source identity -> 422', r.status_code == 422)
with patch.object(add_pipeline, 'run_add', AsyncMock(side_effect=ValueError('provider output invalid'))):
    r = client.post('/add', json=add_body, headers=H)
check('provider ValueError remains retryable 500', r.status_code == 500)
spec = client.get('/openapi.json').json()['components']['schemas']
check('OpenAPI requires top_k', 'top_k' in spec['HTTPSearchRequest']['required'])

# Production packers must count constraints toward the total, preserve hashes,
# and reject an entire evidence group when it no longer fits.
for pack in (evidence_packet.pack, evidence_packet.pack_ranked):
    for k in (1, 100):
        for constraints in (1, k + 1):
            candidates = [dict(id=f'rule-{i}', content=f'Privacy rule {i}', is_constraint=True)
                          for i in range(constraints)]
            candidates += [dict(id=f'fact-{i}', content=f'Fact {i}') for i in range(k)]
            packed, digest, manifest = pack(candidates, k, 32000)
            check(f'{pack.__name__} total cap k={k} rules={constraints}',
                  len(packed) == k and evidence_packet.digest(packed) == digest
                  and manifest['constraint_count'] + manifest['evidence_count'] == k)
packed, digest, manifest = evidence_packet.pack_ranked([
    dict(id='rule', content='Privacy rule', is_constraint=True),
    dict(id='a', content='First link'), dict(id='b', content='Second link')],
    2, 32000, groups=[['a', 'b']])
check('constraint consumes slot without splitting evidence group',
      [item['id'] for item in packed] == ['rule']
      and all(item['reason'] == 'atomic_group_top_k' for item in manifest['omitted'])
      and evidence_packet.digest(packed) == digest)

# 10. explicit task feedback creates procedural memory
r = client.post("/feedback", headers=H, json={
    "user_id": add_body["user_id"], "session_id": add_body["session_id"],
    "task": "Summarize the conversation", "outcome": "success",
    "trace": "Retrieved memories and produced a supported summary.",
    "environment_verified": False})
check("feedback creates experience memory",
    r.status_code == 200 and len(r.json().get("memory_ids", [])) == 1)

# 11. authenticated hard purge removes all user content
r = client.delete(f"/memory/{add_body['user_id']}", headers=H)
check("purge returns content-free receipt",
    r.status_code == 200 and r.json().get("receipt_id", "").startswith("purge_"))
r = client.post("/search", json=sbody, headers=H)
check("purged user has no memories", r.status_code == 200 and r.json()["data"] == [])

failed = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
