"""Offline diagnostic: compact candidate identities without editing source text.

Preserves the complete recorded listwise input before trying extra fine groups.
Only input coverage is measured; this is not a production policy or model test.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
from unittest.mock import patch

from evaluate_recall_context import CSV, packet_coverage, records, reference_turns
from app import cascade_rerank, config, fine_evidence, schemas


def compact_request(req, plan, rows):
    ids = {c['id'] for c in rows}
    prefix = 'c'
    while any(f'{prefix}{i}' in ids for i in range(len(rows))):
        prefix = '_' + prefix
    # Literal data is never rewritten. Leave identities referenced by that
    # data intact, so a source's explicit reference remains resolvable.
    literal = req.query + '\n' + '\n'.join(req.options or []) + '\n' + json.dumps(
        plan.get('_coverage_requirements', [])) + '\n' + '\n'.join(
        c.get('_rank_text', c['content']) for c in rows)
    aliases = {c['id']: c['id'] if c['id'] in literal else f'{prefix}{i}' for i, c in enumerate(rows)}
    display = [dict(c, id=aliases[c['id']]) for c in rows]
    translated = dict(plan)
    # Filter in the ORIGINAL namespace first: an unselected ID must not
    # accidentally become an edge to an aliased selected candidate.
    translated['_fusion_edges'] = [dict(e, source=aliases[e['source']], target=aliases[e['target']])
        for e in plan.get('_fusion_edges', []) if e['source'] in ids and e['target'] in ids]
    translated['_fusion_triples'] = [dict(t, amu_id=aliases[t['amu_id']])
        for t in plan.get('_fusion_triples', []) if t.get('amu_id') in ids]
    translated['_fine_path_groups'] = [[aliases[mid] for mid in group]
        for group in plan.get('_fine_path_groups', []) if set(group) <= ids]
    return cascade_rerank.listwise_request(req, translated, display)


def main(args):
    traces = {t['search_id']: t for t in records(args.run / 'search-debug.jsonl')}
    conversations = {c['conversation_id']: c for c in records(args.prepared)}
    csv.field_size_limit(32*1024*1024)
    with CSV.open() as source:
        gold = list(csv.DictReader(source))
    raw = {}
    with sqlite3.connect((args.run / 'memory.db').resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(row['user_id'], {})[row['request_id'], row['message_index']] = dict(row)
    rows = []
    for result in records(args.run / 'results.jsonl'):
        t = traces[result['search_id']]
        by_id = {c['id']: c for c in t['ranked']}
        original = [by_id[mid] for mid in t['cascade']['listwise']['candidate_ids']]
        fine = [by_id[mid] for mid in t['cascade']['fine']['selected_ids']]
        positions = {c['id']: i for i, c in enumerate(fine)}
        plan = copy.deepcopy(t['plan'])
        plan['_coverage_requirements'] = [{k: r[k] for k in ('id', 'text', 'origin') if k in r}
            for r in t['coverage_manifest']['coverage']['requirements']]
        plan['_fusion_edges'] = t['fusion_edges']
        plan['_fusion_triples'] = t['fusion_triples']
        plan['_fine_path_groups'] = t['cascade']['fine']['selection'].get('adaptive', {}).get('path_groups', [])
        req = schemas.SearchRequest(user_id=t['user_id'], query=t['query'], options=t['options'])
        with patch.multiple(config, **t['versions']['settings']), \
                patch('socket.socket.connect', side_effect=AssertionError('Offline only')):
            request = cascade_rerank.listwise_request(req, plan, original)
            before_bytes = len(request['prompt'].encode())
            if before_bytes != t['cascade']['listwise']['input_bytes']:
                raise ValueError('Original listwise request did not reproduce')
            selected = list(original)
            fixed_compact_bytes = len(compact_request(req, plan, original)['prompt'].encode())
            headroom = t['cascade']['admission_budget']['token_headroom']
            byte_rejected = {o['id'] for o in t['cascade']['omitted']
                             if o['stage'] == 'listwise' and o['reason'] == 'prompt_budget'}
            fine_by_id = {c['id']: c for c in fine}
            for c in fine:
                if c['id'] not in byte_rejected:
                    continue
                group, missing = fine_evidence.closure(c, fine_by_id)
                selected_ids = {x['id'] for x in selected}
                extra = group - selected_ids
                if missing or not extra:
                    continue
                trial = sorted(selected + [fine_by_id[mid] for mid in extra], key=lambda x: positions[x['id']])
                if len(trial) > config.CASCADE_LLM_LIMIT + config.FINE_EXTRA_LIMIT:
                    continue
                request = compact_request(req, plan, trial)
                if len(request['prompt'].encode()) > config.RERANK_MAX_PROMPT_BYTES:
                    continue
                if headroom is not None and request['reservation'] > headroom:
                    continue
                selected = trial
        assert {c['id'] for c in original} <= {c['id'] for c in selected}
        conv = conversations[result['conversation_id']]
        q = next(q for q in conv['qa'] if q['id'] == result['qa_id'])
        assert t['query'] == q['question'] and t['options'] == q['options']
        turns = reference_turns(q, conv['source_history'], gold[int(result['qa_id'])])
        def coverage(items):
            return packet_coverage(turns, [c['_packet_item'] for c in items], raw[t['user_id']])
        rows.append(dict(conversation_id=result['conversation_id'], qa_id=result['qa_id'],
            control=coverage(original), treatment=coverage(selected),
            control_ids=[c['id'] for c in original], treatment_ids=[c['id'] for c in selected],
            before_bytes=before_bytes, fixed_compact_bytes=fixed_compact_bytes))
    summary = dict(questions=len(rows), reproduced_inputs=len(rows),
        visible={a: sum(r[a]['visible'] for r in rows) for a in ('control', 'treatment')},
        full_turns={a: sum(r[a]['full_turns'] for r in rows) for a in ('control', 'treatment')},
        candidate_counts={a: sum(len(r[a+'_ids']) for r in rows) for a in ('control', 'treatment')},
        gained=[r['qa_id'] for r in rows if r['treatment']['visible'] and not r['control']['visible']],
        lost=[r['qa_id'] for r in rows if r['control']['visible'] and not r['treatment']['visible']])
    with args.output.open('x') as stream:
        json.dump(dict(summary=summary, rows=rows, limitation=__doc__,
            trace_sha256=hashlib.sha256((args.run/'search-debug.jsonl').read_bytes()).hexdigest()), stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--prepared', type=Path, default=Path('memory_system/data/prepared/personamem-v2-32k.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
