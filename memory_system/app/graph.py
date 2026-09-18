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


def filter_triples(triples, query, entities, limit=120, seed_amu_ids=()):
    """Bound query-to-triple candidates before PageRank seeds are introduced."""
    terms = set(normalize_entity(query).split()) | {normalize_entity(e) for e in entities}
    seed_ids = set(seed_amu_ids)
    neighbors = {normalize_entity(t[k]) for t in triples if t['amu_id'] in seed_ids
                 for k in ('subject', 'object')}
    scored = []
    for triple in triples:
        text = normalize_entity(' '.join(str(triple.get(k, '')) for k in ('subject', 'relation', 'object')))
        score = sum(term in text for term in terms if len(term) > 1)
        # Keep the actual seeds and their neighbors even when abstract planner
        # entities have no literal match, while preserving the graph size bound.
        score += 100 if triple['amu_id'] in seed_ids else 0
        score += 10 if any(normalize_entity(triple[k]) in neighbors for k in ('subject', 'object')) else 0
        if score:
            scored.append((score, triple))
    return [triple for _, triple in sorted(scored, key=lambda pair: -pair[0])[:limit]]

def ppr_recall(triples: List[Dict], seed_entities: List[str],
               damping: float = 0.5, iters: int = 20,
               top_n: int = 30, seed_amu_ids=None) -> List[str]:
    """Return ranked amu_ids reachable from seed entities."""
    if not triples or not (seed_entities or seed_amu_ids):
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
    hit += [mid for mid in dict.fromkeys(seed_amu_ids or []) if mid in amus]
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
