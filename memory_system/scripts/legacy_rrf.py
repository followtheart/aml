"""Historical v5/v6 trace replay only; never used by live Search."""
from typing import Dict, List
import json

def _rrf(routes: List[List[Dict]], k: int = 60, weights=None, families=None) -> List[Dict]:
    scores: Dict[str, float] = {}
    items: Dict[str, Dict] = {}
    votes = {}
    for index, route in enumerate(routes):
        weight = weights[index] if weights is not None else 1.0
        seen = set()
        for item in route:
            iid = item["id"]
            if iid in seen:
                continue
            seen.add(iid)
            value = weight / (k + len(seen))
            if families is None:
                scores[iid] = scores.get(iid, 0.0) + value
            else:
                family = families[index]
                group = votes.setdefault(iid, {})
                group[family] = max(group.get(family, 0), value)
            items.setdefault(iid, item)
            items[iid]['_coverage_ids'] = sorted(set(items[iid].get('_coverage_ids', [])) |
                                                 set(item.get('_coverage_ids', [])))
            items[iid]['_bridge_ids'] = sorted(set(items[iid].get('_bridge_ids', [])) |
                                               set(item.get('_bridge_ids', [])))
            proofs = items[iid].get('_coverage_proofs', []) + item.get('_coverage_proofs', [])
            items[iid]['_coverage_proofs'] = list({json.dumps(p, sort_keys=True): p for p in proofs}.values())
    if families is not None:
        for iid, values in votes.items():
            ordered_votes = sorted(values.values(), reverse=True)
            scores[iid] = ordered_votes[0] + .15 * sum(ordered_votes[1:3])
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    out = []
    for iid, s in ordered:
        d = dict(items[iid])
        d["_fused"] = s
        if families is not None:
            d['_fusion_votes'] = votes[iid]
        out.append(d)
    return out

