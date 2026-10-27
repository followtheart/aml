"""Reversible candidate display IDs; source strings and stored IDs stay intact.

Related to the historical local-ID diagnostic. This version derives literal
protection from the actual graph payload and verifies the rendered payload in
the original namespace before returning a shorter request.
"""
import copy
from collections import Counter
import json
from unittest.mock import patch

import compact_graph_metadata

TEMPLATE = '05c_listwise_rerank.txt'


def stable(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def graph_literals(metadata):
    literal = []
    for row in metadata.get('triples', []):
        literal.append({k:v for k,v in row.items() if k != 'amu_id'})
    for row in metadata.get('associations', []):
        literal.append({k:v for k,v in row.items() if k not in ('source', 'target')})
    literal.extend(metadata.get('incomplete_candidate_context', {}).values())
    literal.append({k:v for k,v in metadata.items() if k not in
                    ('triples', 'associations', 'incomplete_candidate_context', 'recorded_paths_for_joint_review')})
    return stable(literal)


def translate(req, plan, rows, metadata):
    ids = {c['id'] for c in rows}
    if len(ids) != len(rows):
        raise ValueError('Duplicate candidate IDs')
    literal = '\n'.join([req.query, *(req.options or []), stable(plan.get('time_scope') or {}),
        stable(plan.get('_coverage_requirements', [])), graph_literals(metadata),
        *(c.get('_rank_text', c['content']) for c in rows)])
    prefix = 'c'
    while any(f'{prefix}{i}' in ids or f'{prefix}{i}' in literal for i in range(len(rows))):
        prefix = '_' + prefix
    aliases = {c['id']: c['id'] if c['id'] in literal else f'{prefix}{i}' for i,c in enumerate(rows)}
    if len(set(aliases.values())) != len(aliases):
        raise ValueError('Display alias collision')
    display = [dict(c, id=aliases[c['id']]) for c in rows]
    translated = dict(plan)
    # Select in the original namespace before renaming any structural fields.
    translated['_fusion_edges'] = [dict(e, source=aliases[e['source']], target=aliases[e['target']])
        for e in plan.get('_fusion_edges', []) if e['source'] in ids and e['target'] in ids]
    translated['_fusion_triples'] = [dict(t, amu_id=aliases[t['amu_id']])
        for t in plan.get('_fusion_triples', []) if t.get('amu_id') in ids]
    translated['_fine_path_groups'] = [[aliases[mid] for mid in group]
        for group in plan.get('_fine_path_groups', []) if set(group) <= ids]
    if plan.get('_fine_path_groups') and not translated['_fine_path_groups']:
        # Preserve the producer's empty visible paths list and its scope when
        # recorded paths exist but none are in this request.
        marker = '_outside_local_display'
        while marker in aliases.values():
            marker = '_' + marker
        translated['_fine_path_groups'] = [[marker]]
    return display, translated, aliases


def decode_graph(metadata, aliases):
    original = {value:key for key,value in aliases.items()}
    restored = copy.deepcopy(metadata)
    for row in restored.get('triples', []):
        row['amu_id'] = original[row['amu_id']]
    for row in restored.get('associations', []):
        row['source'], row['target'] = original[row['source']], original[row['target']]
    if 'incomplete_candidate_context' in restored:
        restored['incomplete_candidate_context'] = {
            original[key]:value for key,value in restored['incomplete_candidate_context'].items()}
    if 'recorded_paths_for_joint_review' in restored:
        restored['recorded_paths_for_joint_review'] = [
            [original[mid] for mid in group] for group in restored['recorded_paths_for_joint_review']]
    return restored


def missing_graph(before, after):
    missing = []
    for name, value in before.items():
        if name in ('triples', 'associations', 'recorded_paths_for_joint_review'):
            absent = Counter(stable(row) for row in value) - Counter(stable(row) for row in after.get(name, []))
            missing.extend(dict(field=name, value=json.loads(row), count=count) for row,count in absent.items())
            if not absent:
                remainder = iter(stable(row) for row in after.get(name, []))
                if not all(any(item == stable(row) for item in remainder) for row in value):
                    missing.append(dict(field=name, reason='original_order_changed'))
        elif name == 'incomplete_candidate_context':
            missing.extend(dict(field=name, id=key, value=item) for key,item in value.items()
                           if key not in after.get(name, {}) or after[name][key] != item)
        elif name not in after or after[name] != value:
            missing.append(dict(field=name, value=value))
    return missing


def request(req, plan, rows, cascade):
    original_render = cascade.prompts.render
    captured = []
    def capture(name, **kwargs):
        if name == TEMPLATE:
            captured.append(json.loads(kwargs['relations']))
        return original_render(name, **kwargs)
    with patch.object(cascade.prompts, 'render', capture):
        original = cascade.listwise_request(req, plan, rows)
    if not captured or any(stable(value) != stable(captured[0]) for value in captured):
        raise ValueError('Unexpected original graph renderer boundary')
    metadata = captured[0]
    display, translated, aliases = translate(req, plan, rows, metadata)
    captured = []
    compact_render = compact_graph_metadata.renderer(capture)
    with patch.object(cascade.prompts, 'render', compact_render):
        shortened = cascade.listwise_request(req, translated, display)
    if not captured or any(stable(decode_graph(value, aliases)) != stable(metadata) for value in captured):
        raise ValueError('Graph identity roundtrip changed values or order')
    for old, new in zip(rows, display, strict=True):
        if {k:v for k,v in old.items() if k != 'id'} != {k:v for k,v in new.items() if k != 'id'}:
            raise ValueError('Non-identity candidate field changed')
    if {k:v for k,v in original.items() if k not in ('prompt','reservation')} != {
            k:v for k,v in shortened.items() if k not in ('prompt','reservation')}:
        raise ValueError('Model output protocol changed')
    saved = len(original['prompt'].encode()) - len(shortened['prompt'].encode())
    if original['reservation'] - shortened['reservation'] != saved:
        raise ValueError('Reservation change differs from prompt byte change')
    used = saved > 0
    return (shortened if used else original), dict(used=used, bytes_saved=max(0,saved),
        aliases=aliases if used else {mid:mid for mid in aliases}, decoded_graph=metadata,
        graph_roundtrip_exact=True, candidate_fields_exact=True, output_protocol_exact=True)
