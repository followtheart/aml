"""Exact tenant-local vector + lexical baseline; quotas count events, not views."""
import math
import time
from collections import Counter
from .providers import terms


def recall(events, query_views, vectors, settings, deadline):
    paths = {}
    facets = {v.facet for v in query_views}
    scanned = 0
    for j, query in enumerate(query_views):
        qterms = set(terms(query.text))
        for direct in ([True, False] if j == 0 else [False]):
            docs = [(event, i, view) for event in events for i, view in enumerate(event["views"])
                    if (not direct or i == 0) and (settings.memory_expansion or i == 0)]
            # BM25 over this visible view corpus. Tenant filtering happened before this stage.
            counts = []
            for _, _, view in docs:
                if time.monotonic() >= deadline:
                    raise TimeoutError("recall deadline")
                counts.append(Counter(terms(view["text"])))
            avgdl = sum(sum(c.values()) for c in counts) / max(1, len(counts))
            df = {term: sum(term in c for c in counts) for term in qterms}
            best = {"vector": {}, "lexical": {}}
            for (event, i, view), count in zip(docs, counts):
                if time.monotonic() >= deadline:
                    raise TimeoutError("recall deadline")
                scanned += 1
                cosine = sum(a*b for a, b in zip(vectors[j], view["vector"])) if vectors else 0
                lexical = 0.0
                for term in qterms:
                    tf = count[term]
                    if tf:
                        idf = math.log(1 + (len(docs) - df[term] + .5) / (df[term] + .5))
                        lexical += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * sum(count.values()) / max(avgdl, 1)))
                for channel, score in (("vector", cosine), ("lexical", lexical)):
                    if score <= 0:
                        continue
                    eid = event["event_id"]
                    if eid not in best[channel] or score > best[channel][eid][0]:
                        best[channel][eid] = (score, i, view["kind"])
            for channel, hits in best.items():
                ordered = sorted(hits.items(), key=lambda item: (-item[1][0], item[0]))[:settings.route_limit]
                for rank, (eid, (score, i, kind)) in enumerate(ordered, 1):
                    paths.setdefault(eid, []).append({"a_view": i, "b_view": j, "channel": channel,
                        "raw_score": score, "rank": rank, "kind": kind, "facet": query.facet,
                        "direct": direct, "u": 61/(60+rank)})
    candidates = []
    by_id = {e["event_id"]: e for e in events}
    for eid, connections in paths.items():
        c = max(p["u"] for p in connections)
        d = max((p["u"] for p in connections if p["direct"]), default=0)
        f = len({p["facet"] for p in connections}) / len(facets)
        candidates.append({"event": by_id[eid], "score": .7*c+.2*d+.1*f, "paths": connections})
    candidates.sort(key=lambda c: (-c["score"], c["event"]["event_id"]))
    # Protect a direct hit and each need's best hit inside (not in addition to) the cap.
    reservations = []
    for label in ["__direct__", *sorted(facets)]:
        eligible = [c for c in candidates if any(p["direct"] if label == "__direct__" else p["facet"] == label for p in c["paths"])]
        if eligible and eligible[0] not in reservations:
            reservations.append(eligible[0])
    ordered = reservations + [c for c in candidates if c not in reservations]
    selected = ordered[:settings.candidate_limit]
    return selected, {"view_comparisons": scanned, "matched_events": len(candidates),
                      "candidate_truncated": max(0, len(candidates) - len(selected))}
