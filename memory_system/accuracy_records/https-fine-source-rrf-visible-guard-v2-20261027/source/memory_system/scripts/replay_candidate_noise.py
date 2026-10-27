"""Replay new budgets/previews over recorded routes, without models or DB writes.

This is NOT a new retrieval/accuracy evaluation: vector/source route rankings
were produced by the old queries. It isolates budget and excerpt effects.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, personal_evidence, run_metadata, schemas, search_pipeline as search
from legacy_rrf import _rrf


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


class SnapshotSources:
    def __init__(self, memories):
        self.memories = memories

    def get_amus_by_ids(self, ids, **kwargs):
        return [dict(self.memories[mid]) for mid in ids if mid in self.memories]

    def sources_for_amu(self, mid):
        return self.memories[mid].get('sources', [])

    def sources_for_session(self, user_id, session_id):
        return list({(s['request_id'], s['message_index']): s
                     for m in self.memories.values()
                     if m['user_id'] == user_id and m['session_id'] == session_id
                     for s in m.get('sources', [])}.values())


def replay(memory_path, search_path):
    events, traces = read(memory_path), read(search_path)
    memories = {m['id']: m for event in events for m in event['memories']}
    snapshots = SnapshotSources(memories)
    rows = []
    for t in traces:
        if t.get('fake') or t.get('status') == 'error':
            continue
        req = schemas.SearchRequest(user_id=t['user_id'], query=t['query'], options=t.get('options'),
                                    top_k=t['top_k'], reference_time=t.get('reference_time'))
        plan = dict(t['plan'])
        quotas = dict(vector=config.RECALL_VECTOR_LIMIT, source_text=config.RECALL_SOURCE_LIMIT,
                      full_text=config.RECALL_FTS_LIMIT)
        routes = []
        for r in t['routes']:
            if r['channel'] in ('graph', 'scene'):
                continue
            if r['channel'] == 'profile_rule':
                items = [c for c in r['candidates'] if c.get('type') != 'rule'][:config.RECALL_PROFILE_LIMIT]
                items += [c for c in r['candidates'] if c.get('type') == 'rule']
            else:
                items = r['candidates'][:quotas.get(r['channel'], config.RECALL_PER_ROUTE)]
            routes.append(dict(r, candidates=items))
        direct = {c['id'] for r in routes if r['channel'] in ('source_text', 'full_text') for c in r['candidates']}
        contextual = plan.get('intent') in ('multi_hop', 'narrative', 'document')
        expand = contextual or len(direct) < config.RECALL_EXPANSION_MIN_DIRECT
        remaining = config.RECALL_EXPANSION_LIMIT if expand else 0
        for channel in ('graph', 'scene'):
            r = next((r for r in t['routes'] if r['channel'] == channel), None)
            cap = remaining // 2 if channel == 'graph' else remaining
            if r and cap:
                routes.append(dict(r, candidates=r['candidates'][:cap]))
                remaining -= len(routes[-1]['candidates'])
        plan['_routes'] = routes
        fused = search._fold_versions(_rrf([r['candidates'] for r in routes],
                    weights=[r['weight'] for r in routes]), plan)
        candidates = search._prepare_candidates(snapshots, req, plan, fused)
        head, _ = search._rerank_head(candidates, plan, max(config.RERANK_MAX_CANDIDATES, req.top_k + 20))
        rows.append(dict(search_id=t['search_id'], query=t['query'],
            old_fused=len(t['fused']), replay_fused=len(fused), replay_pool=len(head),
            expansion_enabled=expand, candidate_ids=[c['id'] for c in candidates],
            pool_ids=[mid for c in head for mid in [c['id']] + c.get('_equivalent_ids', [])],
            previews=[dict(id=c['id'], aliases=c.get('_equivalent_ids', []), text=c['_rank_text']) for c in candidates]))
    return dict(scope='歷史 v5/v6 RRF 配額診斷；只重放舊路徑排序上的配額與來源預覽，不執行 v7 圖融合或級聯，不能視為目前搜尋品質或新準確率。',
                inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (memory_path, search_path)},
                current_policy=run_metadata.SEARCH_POLICY,
                current_pipeline_version=run_metadata.versions()['pipeline_version'],
                searches=len(rows), old_mean_candidates=statistics.mean(r['old_fused'] for r in rows) if rows else 0,
                replay_mean_candidates=statistics.mean(r['replay_fused'] for r in rows) if rows else 0, rows=rows)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-log', type=Path, default=root / 'logs/memory-debug.jsonl')
    parser.add_argument('--search-log', type=Path, default=root / 'logs/search-debug.jsonl')
    parser.add_argument('--output', type=Path, default=root / 'runs/candidate-noise-checks/replay.json')
    args = parser.parse_args()
    output = replay(args.memory_log, args.search_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in output.items() if k not in ('inputs', 'rows')}, ensure_ascii=False))
