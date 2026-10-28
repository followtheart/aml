"""Observe quote-based redundancy decisions against complete visible bodies.

Run the frozen production selection unchanged. Record calls that consider a
candidate represented by other candidates even though its complete visible
source body is not covered under the same metadata/graph compatibility rules.
This is an inventory, not a replacement policy or semantic relevance judgment.
Calls used only to produce diagnostic decisions are also recorded separately
by caller/line; a collision is not proof that it caused an omission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

from audit_retained_porter_https import require, same, sha, selection_code
from probe_assistant_admission_order import replay


def records(path):
    pending = b''
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            batch = (pending + block).split(b'\n')
            pending = batch.pop()
            for record in batch:
                if record.strip():
                    yield json.loads(record)
        if pending.strip():
            yield json.loads(pending)


def with_complete_bodies(row, selection):
    pieces = list(selection.fragments(row))
    body = selection._text(row.get('_rank_text', row.get('content', '')))
    for source in row.get('_packet_item', {}).get('sources', []):
        text = selection._text(source.get('content') or '')
        if not text or text not in body:
            continue
        sid = source.get('source_event_id')
        if not sid and source.get('request_id') is not None and source.get('message_index') is not None:
            sid = f"{source['request_id']}:{source['message_index']}"
        if sid:
            # Keep all original quote/profile obligations; full bodies only add
            # obligations. Source role, row qualifiers and graph checks stay.
            piece = ((f'visible-body:{sid}', source.get('role')), text)
            if piece not in pieces:
                pieces.append(piece)
    return pieces


def main(args):
    run = args.run.resolve()
    root = run.parent
    require(run.name == 'control', 'Use full original control cohort')
    require(not args.output.exists(), 'Refusing to overwrite inventory')
    plan = json.loads((root / 'plan.json').read_text())
    audited = json.loads((root / 'retained-selection-audit.json').read_text())
    same(audited['status'], 'passed', 'Frozen selection audit')
    same(sha(root / 'plan.json'), audited['plan_sha256'], 'Frozen plan')
    for name, digest in plan['source_files'].items():
        same(sha(root / 'source' / name), digest, 'Frozen source ' + name)
    paths = [run / name for name in ('server-search-debug.jsonl', 'funnel/stage-transport.json',
        'isolated-manifest.json', 'completed/data/results/personamem-v2-32k.jsonl')]
    for path in paths:
        same(sha(path), audited['input_sha256'][str(path)], 'Audited input ' + str(path))
    manifest = json.loads(paths[2].read_text())
    require(manifest['client_exit_code'] == manifest['audit_exit_code'] == 0
            and manifest['server_exit_code'] in (0, -15), 'Terminal source cohort')
    same(sha(Path(manifest['data'])), plan['data_sha256'], 'Prepared input')
    sys.path.insert(0, str(root / 'source/memory_system'))
    from app import cascade_rerank, config, fine_evidence, schemas
    modules = cascade_rerank, config, fine_evidence, schemas
    selection = fine_evidence.selection
    code, code_sha = selection_code(cascade_rerank)
    same(code_sha, audited['selection_ast_sha256'], 'Frozen selection block')
    results = {}
    for row in records(paths[3]):
        require(row['search_id'] not in results, 'Duplicate source result')
        results[row['search_id']] = {k: row[k] for k in ('qa_id', 'conversation_id', 'user_id', 'packet_hash')}
    same(len(results), plan['questions'], 'Complete result count')
    rows, seen = [], set()
    with patch('socket.socket.connect', side_effect=AssertionError('Offline inventory')):
        for t in records(paths[0]):
            sid = t['search_id']
            require(sid in results and sid not in seen, 'Unexpected/duplicate trace')
            seen.add(sid)
            result = results[sid]
            for key in ('user_id', 'packet_hash'):
                same(t[key], result[key], 'Trace identity ' + key)
            c = t['cascade']
            pool = {r['id']: r for r in t['ranked']}
            coarse = [pool[mid] for mid in c['coarse']['selected_ids']]
            captured = []
            with patch.multiple(config, **t['versions']['settings']):
                full = {r['id']: with_complete_bodies(r, selection) for r in coarse}
                original = selection.represented_by
                def observe(row, selected, cache=None):
                    owners = original(row, selected, cache)
                    if owners and not original(row, selected, full):
                        caller = sys._getframe(1)
                        captured.append(dict(candidate_id=row['id'], owners=owners,
                            selected_ids=[r['id'] for r in selected],
                            caller=caller.f_code.co_name, caller_line=caller.f_lineno,
                            candidate_ce=row.get('_ce_score')))
                    return owners
                with patch.object(selection, 'represented_by', new=observe):
                    baseline, _, size = replay(t, modules, code)
                same(baseline['trace']['fine'], c['fine'], 'Unchanged complete fine trace')
                same(baseline['trace'].get('adaptive_admission'), c.get('adaptive_admission'), 'Unchanged admission')
                same([r['id'] for r in baseline['submitted']], c['listwise']['candidate_ids'], 'Unchanged main order')
                same(size, c['listwise']['input_bytes'], 'Unchanged main bytes')
                fine_ids = set(c['fine']['selected_ids'])
                main_ids = set(c['listwise']['candidate_ids'])
                decisions = {d['id']: d for d in c['fine']['selection']['decisions']}
                affected = []
                for mid in sorted({r['candidate_id'] for r in captured}):
                    candidate = pool[mid]
                    base = selection.fragments(candidate)
                    affected.append(dict(candidate_id=mid, in_fine=mid in fine_ids,
                        in_main=mid in main_ids, ce_score=candidate.get('_ce_score'),
                        original_decision=decisions.get(mid),
                        original_fragment_count=len(base), expanded_fragment_count=len(full[mid]),
                        source_body_hashes=[hashlib.sha256(text.encode()).hexdigest()
                            for (key, _), text in full[mid] if key.startswith('visible-body:')]))
                rows.append(dict(**result, baseline_exact=True, calls=captured, affected=affected))
            print(json.dumps(dict(qa_id=result['qa_id'], baseline_exact=True,
                differing_calls=len(captured), affected_candidates=len(affected))), flush=True)
    same(seen, set(results), 'Complete trace inventory')
    # Join reference carrier annotations only after every observation is fixed.
    stages = {(r['conversation_id'], r['qa_id']): r for r in json.loads(paths[1].read_text())['rows']}
    for row in rows:
        stage = stages[row['conversation_id'], row['qa_id']]
        linked = set(stage['carrier_ids']['coarse'])
        row['reference_linked_affected_ids'] = [r['candidate_id'] for r in row['affected']
            if r['candidate_id'] in linked]
    summary = dict(questions=len(rows), baseline_exact_questions=len(rows),
        affected_questions=sum(bool(r['affected']) for r in rows),
        differing_calls=sum(len(r['calls']) for r in rows),
        affected_candidates=sum(len(r['affected']) for r in rows),
        affected_missing_fine=sum(not c['in_fine'] for r in rows for c in r['affected']),
        affected_missing_main=sum(not c['in_main'] for r in rows for c in r['affected']),
        reference_linked_affected=sum(len(r['reference_linked_affected_ids']) for r in rows))
    report = dict(status='complete', summary=summary, rows=rows, provider_calls=0,
        production_changed=False, limitation=__doc__ + '\nWhitespace-normalized source bodies only; '
        'not a proof of full rendered metadata preservation or semantic relevance. Diagnostic calls '
        'are not automatically causal selection events.', input_sha256={str(p): sha(p) for p in
        (Path(__file__), Path(__file__).with_name('audit_retained_porter_https.py'),
         Path(__file__).with_name('probe_assistant_admission_order.py'), root/'plan.json', *paths)})
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
