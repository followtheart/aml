"""Lightweight knowledge-graph retrieval: entity-AMU bipartite graph + PPR.

HippoRAG-style single-step multi-hop: seed Personalized PageRank with query
entities, then rank AMUs by visit probability. Pure numpy, no graph DB.
"""
from typing import Dict, List
import re
import unicodedata


def normalize_entity(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = re.sub(r"[^\w\u3400-\u9fff]+", " ", value)
    return " ".join(value.split())

def ppr_recall(triples: List[Dict], seed_entities: List[str],
               damping: float = 0.5, iters: int = 20,
               top_n: int = 30) -> List[str]:
    """Return ranked amu_ids reachable from seed entities."""
    if not triples or not seed_entities:
        return []
    seeds = {normalize_entity(s) for s in seed_entities}
    amus = set()
    adjacency = {}
    for t in triples:
        s, o = normalize_entity(t["subject"]), normalize_entity(t["object"])
        amus.add(t["amu_id"])
        for entity in (s, o):
            adjacency.setdefault(entity, set()).add(t["amu_id"])
            adjacency.setdefault(t["amu_id"], set()).add(entity)
    hit = [e for e in seeds if e in adjacency]
    if not hit:
        return []
    p0 = {entity: 1.0 / len(hit) for entity in hit}
    p = dict(p0)
    for _ in range(iters):
        nxt = {node: (1 - damping) * value for node, value in p0.items()}
        for node, value in p.items():
            neighbors = adjacency.get(node, ())
            if not neighbors:
                continue
            share = damping * value / len(neighbors)
            for neighbor in neighbors:
                nxt[neighbor] = nxt.get(neighbor, 0.0) + share
        p = nxt
    amu_scores = [(amu, p.get(amu, 0.0)) for amu in amus]
    amu_scores.sort(key=lambda x: -x[1])
    return [a for a, s in amu_scores[:top_n] if s > 0]
