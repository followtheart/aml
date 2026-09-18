"""Query-personalized propagation over an ephemeral, typed candidate graph.

Graph links are retrieval associations, not inferred facts. Source duplication
never creates an independent vote. No reciprocal-rank fusion is used here.
"""
from collections import Counter, defaultdict, deque
import json
import math
from . import budget, config, graph, personal_evidence as pe


def fuse(routes, plan, *, triples=(), sources=None, dependencies=None):
    sources, dependencies = sources or {}, dependencies or {}
    by_id, semantics, channels = {}, {}, defaultdict(set)
    descriptors = plan.get('_routes', [])
    for i, route in enumerate(routes):
        name = descriptors[i]['channel'] if i < len(descriptors) else 'unknown'
        for original in route:
            budget.check()
            mid = original['id']
            item = by_id.setdefault(mid, dict(original))
            for field in ('_coverage_ids', '_bridge_ids'):
                item[field] = sorted(set(item.get(field, [])) | set(original.get(field, [])))
            proofs = item.get('_coverage_proofs', []) + original.get('_coverage_proofs', [])
            item['_coverage_proofs'] = list({json.dumps(p, sort_keys=True): p for p in proofs}.values())
            channels[mid].add(name)
            if name == 'vector':
                score = original.get('_score', 0)
                if isinstance(score, (float, int)) and math.isfinite(score):
                    semantics[mid] = max(semantics.get(mid, 0), max(0.0, min(1.0, score)))
    if not by_id:
        plan['_graph_fusion'] = dict(method='typed_graph_propagation', nodes=0, edges=0, graph_weight=0, truncated_ids=[])
        plan['_fusion_edges'], plan['_fusion_triples'] = [], []
        return []
    queries = plan.get('_used_queries') or [q['text'] for q in plan.get('_query_specs', [])]
    queries = queries or [plan.get('query', '')]
    needles = [pe.terms(q) for q in queries]
    source_sets = {mid: {s.get('source_event_id') or f"{s.get('request_id')}:{s.get('message_index')}"
                        for s in sources.get(mid, [])} for mid in by_id}
    source_frequency = Counter(key for keys in source_sets.values() for key in keys)
    priors, features = {}, {}
    for mid, item in by_id.items():
        terms = pe.terms(item.get('_rank_text', item['content']))
        lexical = max((len(terms & q) / len(q) for q in needles if q), default=0)
        semantic = semantics.get(mid, 0)
        overlap = max((source_frequency[key] for key in source_sets[mid]), default=1)
        prior = max(.01, .55 * semantic + .45 * lexical) / math.sqrt(overlap)
        priors[mid] = prior
        features[mid] = dict(semantic=semantic, lexical=lexical, source_multiplicity=overlap)
        item['_fusion_source_ids'] = sorted(source_sets[mid])

    # The caller also bounds metadata reads. Keep query/option representatives
    # when the input route union is larger than the candidate graph budget.
    ordered = sorted(by_id, key=lambda mid: (-priors[mid], mid))
    protected = [mid for mid in ordered if by_id[mid].get('type') == 'rule']
    reserved = []
    for q in needles:
        hit = max(ordered, key=lambda mid: len(pe.terms(by_id[mid]['content']) & q)) if q else None
        if hit and pe.terms(by_id[hit]['content']) & q and hit not in reserved:
            reserved.append(hit)
    chosen = list(dict.fromkeys(protected + reserved + ordered))[:config.GRAPH_FUSION_MAX_CANDIDATES]
    # Rules use a separate mandatory path even when the ordinary graph is full.
    chosen = list(dict.fromkeys(protected + chosen))
    dropped = [mid for mid in ordered if mid not in set(chosen)]
    by_id = {mid: by_id[mid] for mid in chosen}
    edges, adjacency, forward = [], defaultdict(dict), defaultdict(list)
    edge_keys = set()

    def connect(left, right, weight, kind, **metadata):
        if left == right or left not in by_id or right not in by_id:
            return
        key = (left, right, kind)
        if key in edge_keys or len(edges) >= 4096:
            return
        edge_keys.add(key)
        adjacency[left][right] = max(adjacency[left].get(right, 0), weight)
        edges.append(dict(source=left, target=right, kind=kind, **metadata))

    valid_triples = []
    seen_triples = set()
    for t in triples:
        budget.check()
        mid = t.get('amu_id')
        if mid not in by_id:
            continue
        key = (mid, t.get('subject'), t.get('relation'), t.get('object'))
        if key in seen_triples:
            continue
        seen_triples.add(key)
        valid_triples.append(dict(t))
        if len(valid_triples) >= 512:
            break
    subjects = defaultdict(list)
    for t in valid_triples:
        subjects[graph.normalize_entity(t.get('subject', ''))].append(t)
    relation_words = set().union(*needles) if needles else set()
    if config.GRAPH_FUSION_ENABLED:
        for t in valid_triples:
            budget.check()
            shared = graph.normalize_entity(t.get('object', ''))
            neighbors = sorted(subjects.get(shared, []), key=lambda x: (-priors[x['amu_id']], x['amu_id']))[:8]
            for other in neighbors if shared else []:
                left, right = t['amu_id'], other['amu_id']
                relevant = bool(relation_words & pe.terms(str(t.get('relation', '')) + ' ' + str(other.get('relation', ''))))
                connect(left, right, (1.5 if relevant else 1) / max(1, len(neighbors)), 'relation_path',
                    shared_entity=shared, triple_ids=[t.get('id'), other.get('id')])
                connect(right, left, .2 / max(1, len(neighbors)), 'reverse_relation_path', shared_entity=shared)
                if left != right:
                    forward[left].append(right)
        for mid, item in by_id.items():
            budget.check()
            for dep in dependencies.get(mid, [])[:8]:
                source = by_id.get(dep.get('source_id'))
                if source and source.get('version') == dep.get('source_version'):
                    connect(mid, source['id'], 1.2, 'requires_source', source_version=dep['source_version'])
                    connect(source['id'], mid, .4, 'derived_from')
            for source in item.get('_bridge_ids', [])[:8]:
                connect(source, mid, .25, 'retrieval_bridge')

    mass = sum(priors[mid] for mid in by_id)
    initial = {mid: priors[mid] / mass for mid in by_id}
    values = dict(initial)
    for _ in range(12):
        budget.check()
        following = {mid: .7 * initial[mid] for mid in by_id}
        for mid, value in values.items():
            neighbors = adjacency.get(mid, {})
            if not neighbors:
                following[mid] += .3 * value
            else:
                total = max(1.0, sum(neighbors.values()))
                for target, weight in neighbors.items():
                    following[target] += .3 * value * weight / total
                following[mid] += .3 * value * (1 - sum(neighbors.values()) / total)
        values = following
    graph_weight = (.4 if plan.get('intent') == 'multi_hop' else
                    .25 if plan.get('intent') in ('document', 'narrative') else .1) if edges else 0
    scale = max(values.values()) or 1
    for mid, item in by_id.items():
        item['_fused'] = (1 - graph_weight) * priors[mid] + graph_weight * values[mid] / scale
        item['_fusion'] = dict(features[mid], graph=values[mid] / scale, graph_weight=graph_weight)
        item['_fusion_channels'] = sorted(channels[mid])
    # Only ordered, recorded relation paths may add packaging bridges. Shared
    # sources and unordered _bridge_ids cannot manufacture a logical path.
    if plan.get('intent') == 'multi_hop' and config.GRAPH_FUSION_ENABLED:
        entities = {graph.normalize_entity(e) for e in plan.get('entities', [])}
        entity_roots = {t['amu_id'] for t in valid_triples if graph.normalize_entity(t.get('subject', '')) in entities}
        roots = sorted(entity_roots or by_id, key=lambda mid: (-priors[mid], mid))[:3 if entity_roots else 1]
        paths = {mid: [mid] for mid in roots}
        queue = deque(roots)
        while queue:
            budget.check()
            mid = queue.popleft()
            if len(paths[mid]) >= 3:
                continue
            for target in forward.get(mid, []):
                if target in paths:
                    continue
                paths[target] = paths[mid] + [target]
                queue.append(target)
                by_id[target]['_bridge_ids'] = sorted(set(by_id[target].get('_bridge_ids', [])) | set(paths[mid]))
                by_id[target]['_relation_path_ids'] = list(paths[target])
    plan['_graph_fusion'] = dict(method='typed_graph_propagation', nodes=len(by_id), edges=len(edges),
        edges_by_type=dict(Counter(e['kind'] for e in edges)), graph_weight=graph_weight,
        truncated_ids=dropped, iterations=12, limitation='recorded retrieval associations; not logical entailment')
    plan['_fusion_edges'], plan['_fusion_triples'] = edges, valid_triples
    return sorted(by_id.values(), key=lambda c: (-c['_fused'], c['id']))
