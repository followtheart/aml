"""Inventory optional fine additions grounded only by assistant lexical matches.

Read terminal frozen HTTPS traces one at a time. No selection changes, provider
calls, threshold sweeps or gold-based candidate nomination. Recorded references
are joined only after the candidate inventory is fixed. This inventory does not
establish that removing a candidate improves recall or semantic relevance.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(args):
    run = args.run.resolve()
    root = run.parent
    require(not args.output.exists(), 'Refusing to overwrite inventory')
    manifest = json.loads((run / 'isolated-manifest.json').read_text())
    plan = json.loads((root / 'plan.json').read_text())
    require((run / 'exit-code').exists() and manifest.get('audit_exit_code') == 0
            and manifest.get('server_exit_code') in (0, -15), 'Arm must be terminal and stage-audited')
    process = Path('/proc') / str(manifest['server_pid']) / 'cmdline'
    require(not process.exists() or str(root).encode() not in process.read_bytes(), 'Service still live')
    for name, digest in plan['source_files'].items():
        require(sha(root / 'source' / name) == digest, 'Frozen source changed: ' + name)
    sys.path.insert(0, str(root / 'source/memory_system'))
    from app import config, evidence_selection, fine_evidence
    result_path = run / 'completed/data/results/personamem-v2-32k.jsonl'
    results = {r['search_id']: dict(qa_id=r['qa_id'], conversation_id=r['conversation_id'])
               for r in records(result_path)}
    require(len(results) == plan['questions'], 'Incomplete result cohort')
    trace_path = run / 'server-search-debug.jsonl'
    rows, seen = [], set()
    with patch('socket.socket.connect', side_effect=AssertionError('Offline inventory')):
        for trace in records(trace_path):
            sid = trace['search_id']
            require(sid in results and sid not in seen, 'Unexpected or duplicate trace')
            seen.add(sid)
            require(all(plan['source_files'].get(name) == digest
                        for name, digest in trace['versions']['file_hashes'].items()),
                    'Trace does not identify the frozen implementation')
            settings = trace['versions']['settings']
            with patch.multiple(config, **settings):
                cascade = trace['cascade']
                adaptive = cascade['fine']['selection'].get('adaptive', {})
                pool = {c['id']: c for c in trace['ranked']}
                coarse = [pool[mid] for mid in cascade['coarse']['selected_ids']]
                visible = {c['id']: evidence_selection.fragments(c) for c in coarse}
                requirements = trace['coverage_manifest']['coverage']['requirements'] + adaptive.get('context_facets', [])
                sparse = fine_evidence.sparse_matches(coarse, requirements, visible)
                matches = {}
                for c in coarse:
                    direct = fine_evidence.relevance(c, requirements, visible[c['id']])
                    combined = dict(direct)
                    for rid, value in sparse[c['id']].items():
                        combined[rid] = max(combined.get(rid, (False, False, 0)), value)
                    matches[c['id']] = dict(direct=direct, sparse=sparse[c['id']], combined=combined)
                ordered = sorted([c for c in coarse if c.get('_ce_score') is not None],
                                 key=fine_evidence.key, reverse=True)
                ranks = {c['id']: i + 1 for i, c in enumerate(ordered)}
                additions = []
                for addition in adaptive.get('additions', []):
                    mid = addition['candidate_id']
                    c = pool[mid]
                    combined = matches[mid]['combined']
                    # Match tuples distinguish user/non-user. Only call the
                    # latter assistant when no system/unknown source exists.
                    assistant_only_match = (bool(combined) and not any(v[0] for v in combined.values())
                        and all(key[1] in ('user', 'assistant') for key, _ in visible[mid]))
                    rank = ranks.get(mid)
                    # Describe one bounded population using the existing fine
                    # cutoff; do not choose thresholds using reference outcomes.
                    diagnostic_population = (addition['reason'] in ('novel_visible_evidence', 'sparse_topic_evidence')
                        and assistant_only_match and rank is not None and rank > settings['CASCADE_FINE_LIMIT']
                        and len(addition['ids']) == 1 and len(addition['group_ids']) == 1
                        and not fine_evidence.dependencies(c) and mid not in adaptive.get('core_ids', []))
                    additions.append(dict(candidate_id=mid, reason=addition['reason'],
                        ids=addition['ids'], group_ids=addition['group_ids'], ce_score=c.get('_ce_score'),
                        full_pool_ce_rank=rank, existing_fine_cutoff=settings['CASCADE_FINE_LIMIT'],
                        matches=matches[mid], assistant_only_match=assistant_only_match,
                        diagnostic_population=diagnostic_population,
                        in_main=mid in cascade['listwise']['candidate_ids'],
                        in_packet=mid in trace['coverage_manifest']['included_ids'],
                        source_roles=sorted({key[1] for key, _ in visible[mid] if key[1] is not None}),
                        rank_text=c.get('_rank_text', c.get('content', '')),
                        sources=[dict(identity=key[0], role=key[1], text=text) for key, text in visible[mid]]))
                rows.append(dict(**results[sid], search_id=sid, query=trace['query'], options=trace['options'],
                                 ce_status=cascade['cross_encoder']['status'], additions=additions))
    require(seen == set(results), 'Incomplete trace inventory')
    # Gold/source-carrier annotations cannot alter which candidates were counted.
    stage_path = run / 'funnel/stage-transport.json'
    stages = json.loads(stage_path.read_text())
    by_question = {(r['conversation_id'], r['qa_id']): r for r in stages['rows']}
    require(len(by_question) == len(rows), 'Stage inventory count mismatch')
    for row in rows:
        stage = by_question[row['conversation_id'], row['qa_id']]
        carriers = set(stage['carrier_ids']['fine'])
        for addition in row['additions']:
            addition['linked_reference_carrier'] = addition['candidate_id'] in carriers
        row['actual_reference_turns'] = {s: stage['visible'][s]['full_turns'] for s in ('fine', 'listwise_input', 'catalog')}
    additions = [a for row in rows for a in row['additions']]
    diagnostic = [a for a in additions if a['diagnostic_population']]
    summary = dict(questions=len(rows), additions=len(additions),
        reasons=dict(Counter(a['reason'] for a in additions)),
        assistant_only_matches=sum(a['assistant_only_match'] for a in additions),
        diagnostic_instances=len(diagnostic), diagnostic_main=sum(a['in_main'] for a in diagnostic),
        diagnostic_packet=sum(a['in_packet'] for a in diagnostic),
        diagnostic_linked_reference_carriers=sum(a['linked_reference_carrier'] for a in diagnostic))
    paths = [Path(__file__), root / 'plan.json', run / 'isolated-manifest.json', result_path, trace_path, stage_path]
    report = dict(status='inventory_complete', summary=summary, rows=rows, provider_calls=0,
                  selection_changed=False, limitation=__doc__, input_sha256={str(p): sha(p) for p in paths})
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
