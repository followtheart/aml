"""Lightweight knowledge-graph retrieval: entity-AMU bipartite graph + PPR.

HippoRAG-style single-step multi-hop: seed Personalized PageRank with query
entities, then rank AMUs by visit probability. Pure numpy, no graph DB.
"""
from typing import Dict, List
import re
import unicodedata
from . import budget


def normalize_entity(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = re.sub(r"[^\w\u3400-\u9fff]+", " ", value)
    return " ".join(value.split())


def filter_triples(triples, query, entities, limit=120, seed_amu_ids=(), max_hops=3):
    """Bounded connected expansion; preserve intermediate edges before PPR."""
    terms = set(normalize_entity(query).split()) | {normalize_entity(e) for e in entities}
    seed_ids = set(seed_amu_ids)
    indexed, adjacency = [], {}
    for index, triple in enumerate(triples):
        if index % 512 == 0:
            budget.check()
        text = normalize_entity(' '.join(str(triple.get(k, '')) for k in ('subject', 'relation', 'object')))
        score = sum(term in text for term in terms if len(term) > 1)
        score += 100 if triple['amu_id'] in seed_ids else 0
        indexed.append((score, triple))
        for key in ('subject', 'object'):
            adjacency.setdefault(normalize_entity(triple[key]), []).append(index)
    frontier = {i: [] for i, (score, _) in enumerate(indexed) if score}
    chosen, visited = [], set()
    for depth in range(max_hops):
        budget.check()
        following = {}
        # Reserve room for the next hop even when many roots match a common word.
        capacity = max(1, (limit - len(chosen)) // max(1, max_hops - depth))
        ordered = sorted(frontier, key=lambda i: (-indexed[i][0], indexed[i][1]['amu_id'], i))
        for index in ordered[:capacity]:
            if index in visited or len(chosen) >= limit:
                continue
            visited.add(index)
            triple = dict(indexed[index][1])
            path = list(dict.fromkeys(frontier[index] + [triple['amu_id']]))
            triple['_path_ids'], triple['_hop'] = path, depth + 1
            chosen.append(triple)
            for key in ('subject', 'object'):
                for neighbor in adjacency.get(normalize_entity(triple[key]), []):
                    if neighbor not in visited:
                        following.setdefault(neighbor, path)
        # Carry over unexpanded roots, but favor actual continuations.
        for index in ordered[capacity:]:
            following.setdefault(index, frontier[index])
        frontier = {i: path for i, path in following.items() if i not in visited}
        if not frontier:
            break
    return chosen

def ppr_recall(triples: List[Dict], seed_entities: List[str],
               damping: float = 0.5, iters: int = 20,
               top_n: int = 30, seed_amu_ids=None, query='', exclude_ids=()) -> List[str]:
    """Return ranked amu_ids reachable from seed entities.

    `exclude_ids` are removed before the cut so already-recalled seeds do not
    consume the expansion slots their own teleport mass would otherwise win.
    """
    if not triples or not (seed_entities or seed_amu_ids):
        return []
    seeds = {normalize_entity(s) for s in seed_entities}
    excluded = set(exclude_ids or ())
    amus = set()
    adjacency = {}
    relation_terms = set(normalize_entity(query).split())
    def edge(a, b, weight):
        adjacency.setdefault(a, {})[b] = max(adjacency.get(a, {}).get(b, 0), weight)
    for t in triples:
        s, o = ('entity:' + normalize_entity(t[k]) for k in ('subject', 'object'))
        mid = 'memory:' + t['amu_id']
        amus.add(t["amu_id"])
        relevance = 1 + bool(relation_terms & set(normalize_entity(t.get('relation', '')).split()))
        edge(s, mid, relevance)
        edge(mid, o, relevance)
        edge(o, mid, .35 * relevance)
        edge(mid, s, .35 * relevance)
    hit = ['entity:' + e for e in sorted(seeds) if 'entity:' + e in adjacency]
    hit += ['memory:' + mid for mid in dict.fromkeys(seed_amu_ids or []) if mid in amus]
    if not hit:
        return []
    p0 = {entity: 1.0 / len(hit) for entity in hit}
    p = dict(p0)
    for _ in range(iters):
        budget.check()
        nxt = {node: (1 - damping) * value for node, value in p0.items()}
        for node, value in p.items():
            neighbors = adjacency.get(node, {})
            if not neighbors:
                continue
            share = damping * value / sum(neighbors.values())
            for neighbor, weight in neighbors.items():
                nxt[neighbor] = nxt.get(neighbor, 0.0) + share * weight
        p = nxt
    amu_scores = [(amu, p.get('memory:' + amu, 0.0)) for amu in amus if amu not in excluded]
    amu_scores.sort(key=lambda x: (-x[1], x[0]))
    return [a for a, s in amu_scores[:top_n] if s > 0]
