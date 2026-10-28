"""Replay selection over fixed recorded scores; never call a model or write a DB."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, run_metadata, schemas, search_pipeline as search
from replay_candidate_noise import SnapshotSources, read


def replay(memory_log, search_log):
    memories = {m['id']: m for event in read(memory_log) for m in event['memories']}
    snapshots = SnapshotSources(memories)
    results = []
    for line, trace in enumerate(read(search_log), 1):
        if trace.get('status') == 'error':
            continue
        req = schemas.SearchRequest(user_id=trace['user_id'], query=trace['query'],
                                    options=trace.get('options'), top_k=trace['top_k'])
        plan = dict(trace['plan'], _used_queries=trace['rounds'][0]['queries'],
                    _rerank=deepcopy(trace['rerank']), _rerank_errors=deepcopy(trace.get('rerank_errors', [])),
                    _rerank_status=trace['coverage_manifest'].get('rerank_status', 'ok'))
        cascade = trace.get('cascade') or trace.get('coverage_manifest', {}).get('cascade')
        if cascade:
            plan['_cascade'] = deepcopy(cascade)
        old_ranked = deepcopy(trace['ranked'])
        if cascade:
            # v7 traces contain the exact immutable units and ordinal decisions.
            # Rebuilding them from a snapshot union would change what the model saw.
            ranked = old_ranked
        else:
            hydrated = {c['id']: c for c in search._prepare_candidates(snapshots, req, plan, old_ranked)}
            assert set(hydrated) == {c['id'] for c in old_ranked}, 'Replay changed the candidate inventory'
            ranked = deepcopy(trace['ranked'])
            for c in ranked:
                c['_selection_evidence'] = hydrated[c['id']]['_selection_evidence']
        selected = search._select_evidence(req, plan, ranked)
        old_omitted = {c['id'] for c in trace['selection']['omitted']}
        old_ids = trace['selection'].get('selected_ids', [c['id'] for c in ranked if c['id'] not in old_omitted])
        new_ids = [c['id'] for c in selected]
        assert len(old_ids) == trace['selection']['selected_count']
        # Batch scheduling belongs to the recorded policy; current selection
        # replay must not pretend to reproduce removed historical model calls.
        batches = cascade.get('cross_encoder', {}).get('batches', []) if cascade else trace.get('rerank_batches', [])
        batch_sizes = [b['candidate_count'] for b in batches]
        results.append(dict(search_log_line=line, search_id=trace['search_id'], query=trace['query'],
            old_mode=trace['selection']['mode'], new_mode=plan['_selection']['mode'],
            old_selected=old_ids, new_selected=new_ids,
            added=[mid for mid in new_ids if mid not in old_ids],
            removed=[mid for mid in old_ids if mid not in new_ids],
            selected_user_source_count=sum(c['_selection_evidence']['user_source'] for c in selected if not search._protected_rule(c)),
            old_batch_sizes=batch_sizes, new_batch_sizes=batch_sizes,
            batch_scope='recorded_only', recorded_policy=trace.get('pipeline'),
            candidates=[dict(id=c['id'], score=c.get('_final'), sources=c['_selection_evidence'],
                             content=c['content'][:500]) for c in selected]))
    return dict(scope='固定記錄中的候選與排序，只重放選中策略；批次沿用原紀錄。未重跑模型、圖融合、裝包或回答，不代表新準確率。',
                input_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (memory_log, search_log)},
                versions=run_metadata.versions(), settings=dict(fallback_limit=config.EVIDENCE_FALLBACK_ITEMS), rows=results)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-log', type=Path, default=root / 'logs/memory-debug.jsonl')
    parser.add_argument('--search-log', type=Path, default=root / 'logs/search-debug.jsonl')
    parser.add_argument('--output', type=Path, default=root / 'runs/rerank-selection-checks/replay.json')
    args = parser.parse_args()
    result = replay(args.memory_log, args.search_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(searches=len(result['rows']), changed=[r['search_log_line'] for r in result['rows'] if r['added'] or r['removed']],
                         batch_changes=[dict(line=r['search_log_line'], before=r['old_batch_sizes'], after=r['new_batch_sizes'])
                                        for r in result['rows'] if r['old_batch_sizes'] != r['new_batch_sizes']]), ensure_ascii=False))
