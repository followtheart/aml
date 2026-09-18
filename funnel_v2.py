# -*- coding: utf-8 -*-
"""Funnel v2: citation-first gold evidence resolution."""
import json, re, math
from collections import Counter, defaultdict

WS = r"C:\Users\aapoo\Desktop\aml"

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

results = load(WS + r"\memory_system\data\results\personamem-v2-32k.jsonl")
searches = {s["search_id"]: s for s in load(WS + r"\memory_system\logs\search-debug.jsonl")}
mem_events = load(WS + r"\memory_system\logs\memory-debug.jsonl")

store = defaultdict(dict)
for ev in mem_events:
    for m in ev.get("memories", []):
        store[ev["user_id"]][m.get("id")] = m

ds = {}
with open(WS + r"\memory_system\data\prepared\personamem-v2-32k.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        ds[rec["conversation_id"]] = {q["id"]: q for q in rec["qa"]}

STOP = set("""a an the and or of to in for on with at by from as is are was were be been being
you your yours yourself we our i me my he she it its they them their this that these those
might may can could should would will shall do does did have has had not no yes if then than
so such very just also about into over after before between during each other more most some
any all both few own same too s t d ll re ve since already enjoy""".split())

def toks(text):
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 2]

df = Counter()
mem_toks = {}
for uid, mems in store.items():
    for mid, m in mems.items():
        t = toks((m.get("content") or "") + " " + (m.get("retrieval_key") or ""))
        mem_toks[mid] = t
        for w in set(t):
            df[w] += 1
N = len(mem_toks)
def idf(w): return math.log((N + 1) / (df.get(w, 0) + 0.5)) + 1
def vec(words):
    c = Counter(words)
    return {w: (1 + math.log(n)) * idf(w) for w, n in c.items()}
def cos(a, b):
    dot = sum(v * b.get(w, 0) for w, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values())); nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0

def strip_letter(opt): return re.sub(r"^[A-D]\.\s*", "", opt)

mem_by_id = {}
for uid, mems in store.items():
    mem_by_id.update(mems)

rows = []
for r in results:
    cid, qid = r["conversation_id"], r["qa_id"]
    q = ds[cid][qid]
    gold = q["gold_labels"][0]
    opts = {re.match(r"^([A-D])\.", o).group(1): strip_letter(o) for o in q["options"]}
    uid = f"local:personamem-v2:{cid}"
    mems = store.get(uid, {})
    s = searches.get(r["search_id"], {})
    ranked_ids = [m["id"] for m in s.get("ranked", [])]
    included = set(r["coverage_manifest"]["included_ids"])

    ca = {c["letter"]: c for c in r.get("choice_alignment", [])}
    ca_gold = ca.get(gold, {})
    cited = ca_gold.get("evidence_id") or ""
    cited_ok = bool(cited and ca_gold.get("citation_valid"))

    # similarity fallback ranking
    ov = {L: vec(toks(t)) for L, t in opts.items()}
    scored = []
    for mid in mems:
        mv = vec(mem_toks[mid])
        sims = {L: cos(mv, v) for L, v in ov.items()}
        g = sims[gold]; oth = max(v for L, v in sims.items() if L != gold)
        scored.append((g - oth, g, mid))
    scored.sort(reverse=True)
    margin, gsim, sim_mid = scored[0]

    if cited_ok:
        gold_mid, src = cited, "cite"
    else:
        gold_mid, src = sim_mid, f"sim(m={margin:.2f})"

    in_store = gold_mid in mems
    in_ranked = gold_mid in ranked_ids
    in_packet = gold_mid in included
    correct = r["score"] == 1.0
    rank = ranked_ids.index(gold_mid) + 1 if in_ranked else None

    if not in_store: stage = "1-记住"
    elif not in_ranked: stage = "2-召回"
    elif not in_packet: stage = "3-入包"
    elif not correct: stage = "4-答对"
    else: stage = "OK"

    rows.append(dict(qid=qid, cat=r["category"], gold=gold, pred=r["prediction"],
        score=r["score"], stage=stage, src=src, rank=rank,
        unsupp=ca_gold.get("unsupported_claims"),
        mem=(mem_by_id.get(gold_mid, {}).get("content") or "")[:130],
        gold_claim=opts[gold][:90],
        pred_claim=opts[r["prediction"]][:90]))

print(f"{'qa':>3} {'cat':<13} g p sc {'stage':<7} {'src':<14} {'rank':>4}")
for x in rows:
    print(f"{x['qid']:>3} {x['cat']:<13} {x['gold']} {x['pred']} {int(x['score'])}  {x['stage']:<7} {x['src']:<14} {str(x['rank']):>4}")

print("\n=== stage counts ===")
for k, v in sorted(Counter(x["stage"] for x in rows).items()):
    print(f"  {k}: {v}")

print("\n=== 失败题详情 ===")
for x in rows:
    if x["stage"] == "OK":
        continue
    print(f"\n--- qa{x['qid']} [{x['cat']}] stage={x['stage']} gold={x['gold']} pred={x['pred']} rank={x['rank']}")
    print(f"  GOLD : {x['gold_claim']}")
    print(f"  PRED : {x['pred_claim']}")
    print(f"  MEM  : {x['mem']}")
    if x["unsupp"]:
        print(f"  UNSUPPORTED: {x['unsupp']}")

with open(WS + r"\funnel_rows_v2.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
