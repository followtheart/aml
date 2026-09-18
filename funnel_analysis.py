# -*- coding: utf-8 -*-
"""Funnel attribution: 记住 -> 召回 -> 入包 -> 答对 for personamem-v2-32k run."""
import json, re, math
from collections import Counter, defaultdict

WS = r"C:\Users\aapoo\Desktop\aml"

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

results = load(WS + r"\memory_system\data\results\personamem-v2-32k.jsonl")
searches = {s["search_id"]: s for s in load(WS + r"\memory_system\logs\search-debug.jsonl")}
mem_events = load(WS + r"\memory_system\logs\memory-debug.jsonl")

# full memory store per user
store = defaultdict(dict)  # user_id -> mem_id -> memory
for ev in mem_events:
    for m in ev.get("memories", []):
        store[ev["user_id"]][m.get("id")] = m

# dataset gold labels
ds = {}
with open(WS + r"\memory_system\data\prepared\personamem-v2-32k.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        ds[rec["conversation_id"]] = {q["id"]: q for q in rec["qa"]}

STOP = set("""a an the and or of to in for on with at by from as is are was were be been being
you your yours yourself we our i me my he she it its they them their this that these those
might may can could should would will shall do does did have has had not no yes if then than
so such very just also about into over after before between during each other more most some
any all both few own same too s t d ll re ve""".split())

def toks(text):
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 2]

# build idf over all memory contents
all_docs = []
for uid, mems in store.items():
    for mid, m in mems.items():
        all_docs.append(toks((m.get("content") or "") + " " + (m.get("retrieval_key") or "")))
df = Counter()
for d in all_docs:
    for w in set(d):
        df[w] += 1
N = len(all_docs)
def idf(w):
    return math.log((N + 1) / (df.get(w, 0) + 0.5)) + 1

def tfidf_vec(words):
    c = Counter(words)
    return {w: (1 + math.log(n)) * idf(w) for w, n in c.items()}

def cos(v1, v2):
    dot = sum(v * v2.get(w, 0) for w, v in v1.items())
    n1 = math.sqrt(sum(v * v for v in v1.values()))
    n2 = math.sqrt(sum(v * v for v in v2.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0

def strip_letter(opt):
    return re.sub(r"^[A-D]\.\s*", "", opt)

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
    omitted = {o.get("id") for o in r["coverage_manifest"].get("omitted", [])}
    view_excluded = {m.get("id") for m in s.get("view_excluded", [])}
    topk_excluded = {m.get("id") for m in s.get("top_k_excluded", [])}

    # score each memory against each option (tf-idf cosine)
    opt_vecs = {L: tfidf_vec(toks(t)) for L, t in opts.items()}
    best = None  # (margin, gold_sim, mem_id)
    for mid, m in mems.items():
        mv = tfidf_vec(toks((m.get("content") or "") + " " + (m.get("retrieval_key") or "")))
        sims = {L: cos(mv, ov) for L, ov in opt_vecs.items()}
        g = sims[gold]
        others = max(v for L, v in sims.items() if L != gold)
        margin = g - others
        if best is None or margin > best[0]:
            best = (margin, g, mid, sims)
    margin, gsim, gold_mid, sims = best
    remembered = gsim >= 0.12 and margin > 0.01  # supports gold specifically

    # corroborate with choice_alignment citation on gold letter
    ca_gold = [c for c in r.get("choice_alignment", []) if c.get("letter") == gold]
    cited = ca_gold[0].get("evidence_id") if ca_gold else ""
    cited_valid = bool(ca_gold and ca_gold[0].get("citation_valid") and cited)

    recalled = remembered and gold_mid in ranked_ids
    in_packet = remembered and gold_mid in included
    correct = r["score"] == 1.0

    if not remembered:
        stage = "1-记住"
    elif not recalled:
        stage = "2-召回"
    elif not in_packet:
        stage = "3-入包"
    elif not correct:
        stage = "4-答对"
    else:
        stage = "OK-答对"

    # sub-reason for 入包 failures
    sub = ""
    if stage == "3-入包":
        if gold_mid in view_excluded: sub = "view/rerank排除"
        elif gold_mid in topk_excluded: sub = "top_k截断"
        elif gold_mid in omitted: sub = "预算omitted"
        else: sub = "其他"

    rows.append(dict(
        qid=qid, cat=r["category"], gold=gold, pred=r["prediction"], score=r["score"],
        stage=stage, sub=sub, gold_mid=gold_mid, gsim=round(gsim, 3), margin=round(margin, 3),
        rank=(ranked_ids.index(gold_mid) + 1) if recalled else None,
        cited=cited if cited_valid else "", cited_same=(cited == gold_mid) if cited_valid else None,
        mem=(mems.get(gold_mid, {}).get("content") or "")[:110] if remembered else "",
        gold_claim=opts[gold][:110],
    ))

print(f"total memories in store: {sum(len(v) for v in store.values())}")
print(f"{'qa':>3} {'cat':<13} {'gold':>4} {'pred':>4} {'score':>5} {'stage':<8} {'sub':<14} {'gsim':>5} {'mgn':>5} {'rank':>4}  mem/gold")
for x in rows:
    print(f"{x['qid']:>3} {x['cat']:<13} {x['gold']:>4} {x['pred']:>4} {x['score']:>5} {x['stage']:<8} {x['sub']:<14} {x['gsim']:>5} {x['margin']:>5} {str(x['rank']):>4}  {x['mem'][:70]} || {x['gold_claim'][:60]}")

print("\n=== stage counts (failures only) ===")
cnt = Counter(x["stage"] for x in rows)
for k in sorted(cnt):
    print(f"  {k}: {cnt[k]}")

print("\n=== choice_alignment corroboration (gold letter cited same memory?) ===")
for x in rows:
    if x["cited_same"] is not None:
        print(f"  qa{x['qid']}: cited_valid={x['cited'][:24]} same_as_matched={x['cited_same']} stage={x['stage']}")

print("\n=== detail: wrong QAs where evidence WAS in packet (答对失败) ===")
for x in rows:
    if x["stage"] == "4-答对":
        print(f"  qa{x['qid']} {x['cat']} gold={x['gold']} pred={x['pred']} rank={x['rank']} mem='{x['mem']}'")

with open(WS + r"\funnel_rows.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
print("\nsaved funnel_rows.json")
