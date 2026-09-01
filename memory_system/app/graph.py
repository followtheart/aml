"""Lightweight knowledge-graph retrieval: entity-AMU bipartite graph + PPR.

HippoRAG-style single-step multi-hop: seed Personalized PageRank with query
entities, then rank AMUs by visit probability. Pure numpy, no graph DB.
"""
from typing import Dict, List

import numpy as np


def ppr_recall(triples: List[Dict], seed_entities: List[str],
               damping: float = 0.5, iters: int = 20,
               top_n: int = 30) -> List[str]:
    """Return ranked amu_ids reachable from seed entities."""
    if not triples or not seed_entities:
        return []
    seeds = {s.lower() for s in seed_entities}
    ents, amus = set(), set()
    edges = []
    for t in triples:
        s, o = t["subject"].lower(), t["object"].lower()
        ents.update([s, o])
        amus.add(t["amu_id"])
        edges.append((s, t["amu_id"]))
        edges.append((o, t["amu_id"]))
    nodes = sorted(ents) + sorted(amus)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        return []
    A = np.zeros((n, n), dtype=np.float32)
    for e, a in edges:
        i, j = idx[e], idx[a]
        A[i, j] = 1.0
        A[j, i] = 1.0
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    P = A / deg  # row-stochastic

    p0 = np.zeros(n, dtype=np.float32)
    hit = [e for e in seeds if e in idx]
    if not hit:
        return []
    for e in hit:
        p0[idx[e]] = 1.0 / len(hit)

    p = p0.copy()
    for _ in range(iters):
        p = damping * (P.T @ p) + (1 - damping) * p0
    amu_scores = [(nodes[i], p[i]) for i in range(n) if nodes[i] in amus]
    amu_scores.sort(key=lambda x: -x[1])
    return [a for a, s in amu_scores[:top_n] if s > 0]
