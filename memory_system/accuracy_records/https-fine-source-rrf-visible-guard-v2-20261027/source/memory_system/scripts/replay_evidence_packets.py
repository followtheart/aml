"""Repack recorded rankings with current code, offline and without a live DB.

Uses Add snapshots plus only dependency edges explicitly visible in recorded
packets. This is a packing comparison, not a new Search/Answer accuracy run.
Newly admitted raw-route facts are reported separately, never assigned invented
reranker scores. No Add writes, provider calls, or changes to original logs.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys

os.environ['AML_FAKE'] = '1'
os.environ['AML_MEMORY_DEBUG_LOG'] = ''
os.environ['AML_SEARCH_DEBUG_LOG'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context, choice_alignment, config, personal_evidence, run_metadata, schemas, search_pipeline


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def recorded_ranking(trace):
    if 'ranked' in trace:
        return copy.deepcopy(trace['ranked'])
    last = max((d['round'] for d in trace['rerank']), default=0)
    decisions = {d['id']: d for d in trace['rerank'] if d['round'] == last}
    fused = copy.deepcopy(trace['fused'])
    for candidate in fused:
        decision = decisions.get(candidate['id'], {})
        if decision.get('reason') != 'unscored_fused':
            candidate['_final'] = decision.get('score', candidate.get('_fused', 0))
    if any(d['reason'] == 'global_rrf_fallback' for d in decisions.values()):
        return fused
    scored = [c for c in fused if decisions.get(c['id'], {}).get('reason') == 'kept']
    tail = [c for c in fused if decisions.get(c['id'], {}).get('reason') == 'unscored_fused']
    return sorted(scored, key=lambda c: -c['_final']) + tail


class RecordedStore:
    """Read-only adapter; its scope is the snapshot evidence in these logs."""
    def __init__(self, logs, traces):
        self.memories = {m['id']: m for log in logs for m in log['memories']}
        self.dependencies = {}
        for trace in traces:
            for item in trace['returned']:
                refs = re.findall(r'\[support: (amu_\w+)\]', item['content'])
                self.dependencies.setdefault(item['id'], set()).update(refs)

    def sources_for_amu(self, mid):
        return copy.deepcopy(self.memories.get(mid, {}).get('sources', []))

    def sources_for_session(self, uid, sid):
        sources = {(s['request_id'], s['message_index']): s
                   for m in self.memories.values() if m['user_id'] == uid and m['session_id'] == sid
                   for s in m.get('sources', [])}
        return copy.deepcopy(list(sources.values()))

    def get_amus_by_ids(self, mids, **kwargs):
        return [copy.deepcopy(self.memories[mid]) for mid in mids if mid in self.memories]

    def dependencies_for(self, mid):
        return [dict(source_id=aid, source_version=self.memories[aid]['version'])
                for aid in sorted(self.dependencies.get(mid, [])) if aid in self.memories]

    def core_profile(self, uid, limit):
        # Supplement only profiles actually observed as eligible in this run.
        rows = [copy.deepcopy(m) for m in self.memories.values() if m['user_id'] == uid
                and m.get('type') in ('rule', 'profile', 'preference')
                and m.get('sensitivity', 'normal') == 'normal' and not m.get('valid_to')
                and m.get('view_status', 'ready') == 'ready']
        from app import profile
        rows = [m for m in rows if not profile.is_forget_rule(m)]
        rows.sort(key=lambda m: m.get('created_at', ''), reverse=True)
        rows.sort(key=lambda m: (m['type'] != 'rule', m['type'] != 'profile',
                                m.get('profile_status') not in ('static', 'stable', 'rule')))
        return rows[:limit]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--memory-log', required=True)
    ap.add_argument('--traces', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--budget', type=int, default=12000)
    args = ap.parse_args()
    traces = [t for t in read(args.traces) if t.get('event') == 'memory.search'
              and not t.get('fake') and t.get('returned')]
    adapter = RecordedStore(read(args.memory_log), traces)
    rows = []
    for trace in traces:
        old = trace['returned']
        answer_context.build(old)  # Validate recorded hash before comparing.
        req = schemas.SearchRequest(user_id=trace['user_id'], query=trace['query'],
                                     options=trace.get('options'), top_k=trace['top_k'],
                                     evidence_token_budget=args.budget)
        plan = copy.deepcopy(trace['plan'])
        cascade = trace.get('cascade') or trace.get('coverage_manifest', {}).get('cascade')
        ranking = recorded_ranking(trace)
        if cascade:
            plan['_cascade'] = copy.deepcopy(cascade)
            plan['_evidence_groups'] = copy.deepcopy(trace.get('evidence_groups',
                trace.get('coverage_manifest', {}).get('requested_evidence_groups', [])))
            ranking = [c for c in ranking if c.get('_cascade_selected')]
        routes = [route['candidates'] for route in trace['routes']]
        admitted = {m['id'] for route in routes for m in route}
        fused_ids = {m['id'] for m in trace['fused']}
        # Hold the previous ranking fixed to isolate packaging changes.
        packet, digest, manifest = search_pipeline._pack_evidence(
            adapter, req, plan, ranking, trace['anchor_time'])
        rendered = answer_context.build(packet)
        schema_packet = [schemas.SearchItem(**m).model_dump() for m in packet]
        assert answer_context.build(schema_packet) == rendered
        assert len(rendered.encode()) <= args.budget
        pool, constraints = choice_alignment.evidence_pool(packet)
        rows.append(dict(search_id=trace['search_id'], query=trace['query'],
                         options=trace.get('options'), user_id=trace['user_id'], fake=True,
                         status='offline_repacked_not_verified', packet_hash=digest,
                         original_count=len(old), count=len(packet),
                         original_bytes=len('\n'.join(m['content'] for m in old).encode()),
                         bytes=len(rendered.encode()),
                         restored_by_view_ids=sorted(admitted-fused_ids),
                         newly_packed_ids=sorted(set(manifest['included_ids'])-{m['id'] for m in old}),
                         alignment_input_ids=list(pool), alignment_constraint_ids=list(constraints),
                         coverage_manifest=manifest, returned=schema_packet))
    output = Path(args.output)
    if output.resolve() in {Path(args.traces).resolve(), Path(args.memory_log).resolve()}:
        ap.error('Output must not overwrite source logs')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = dict(questions=len(rows), budget=args.budget, provider_calls=0,
                   old_average_items=sum(r['original_count'] for r in rows)/max(1,len(rows)),
                   new_average_items=sum(r['count'] for r in rows)/max(1,len(rows)),
                   old_average_bytes=sum(r['original_bytes'] for r in rows)/max(1,len(rows)),
                   new_average_bytes=sum(r['bytes'] for r in rows)/max(1,len(rows)),
                   versions=run_metadata.versions(),
                   limitation='Recorded ranking held fixed; snapshot union is not a full historical database; no new answer scores.')
    output.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='versions'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
