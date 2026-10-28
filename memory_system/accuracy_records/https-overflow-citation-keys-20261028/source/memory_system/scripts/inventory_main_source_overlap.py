"""Inventory residual literal source duplication after current shared rendering.

No candidate selection changes or provider calls. Stream one terminal trace at
a time, use its frozen renderer, and verify every rendered candidate exactly.
Containment estimates exclude encoding/decoder overhead and are not predicted
admission gains. Only identical displayed text is compared; no provenance or
semantic equivalence is inferred from repeated strings.
"""
import argparse
from collections import defaultdict
import difflib
import json
from pathlib import Path
import sys
from unittest.mock import patch

from audit_retained_porter_https import records, require, same, sha


def inventory(rows, renderer, context):
    rendered = renderer.shared_candidates(rows)
    data = json.loads(rendered.split('\n', 1)[1])
    sources = data['shared_sources']
    stats = dict(candidates=len(rows), shared_sources=len(sources),
                 fallback_candidates=0, exact_reconstructed=0,
                 same_origin_variant_groups=0, contained_source_pairs=0,
                 contained_content_bytes_upper_bound=0,
                 partial_overlap_sources=0, partial_overlap_bytes_upper_bound=0,
                 fallback_literal_source_matches=0,
                 fallback_literal_bytes_upper_bound=0)
    variants = defaultdict(dict)
    pairs, fallback, partial = [], [], []
    for row, card in zip(rows, data['candidates'], strict=True):
        original = renderer.normalized(row.get('_rank_text', row['content']))
        if 'full_text' in card:
            restored = card['full_text']
            stats['fallback_candidates'] += 1
            # Select nonoverlapping literal spans, longest first. This is an
            # optimistic bytes-only bound; no encoding is proposed here.
            occupied, matches = [], []
            for key, text in sorted(sources.items(), key=lambda x: -len(x[1])):
                start = 0
                while (start := restored.find(text, start)) >= 0:
                    end = start + len(text)
                    if not any(start < b and end > a for a, b in occupied):
                        occupied.append((start, end))
                        matches.append(dict(source_index=key, start=start, end=end,
                                            literal_bytes=len(text.encode())))
                    start = end
            if matches:
                fallback.append(dict(candidate_id=row['id'], matches=matches))
                stats['fallback_literal_source_matches'] += len(matches)
                stats['fallback_literal_bytes_upper_bound'] += sum(m['literal_bytes'] for m in matches)
        else:
            restored = renderer.normalized(card['body'] + context.SOURCE_MARKER +
                ' '.join(sources[str(ref)] for ref in card['source_refs']))
            for source, ref in zip([s for s in row['_packet_item']['sources'] if s.get('content')],
                                   card['source_refs'], strict=True):
                # Restrict possible overlap to the same recorded message,
                # speaker and displayed timestamp; never merge by content alone.
                origin = (source.get('request_id'), source.get('message_index'),
                          source.get('role'), context.readable_time(source.get('timestamp')))
                if origin[0] is None or origin[1] is None:
                    continue
                text = renderer.normalized(source['content'])
                variants[origin][text] = str(ref)
        same(restored, original, 'Lossless current display')
        same(card['id'], row['id'], 'Card identity')
        stats['exact_reconstructed'] += 1
    for group in variants.values():
        if len(group) < 2:
            continue
        stats['same_origin_variant_groups'] += 1
        for text, ref in group.items():
            supersets = [(other, other_ref) for other, other_ref in group.items()
                         if text != other and text in other]
            if supersets:
                parent, parent_ref = max(supersets, key=lambda x: len(x[0]))
                pairs.append(dict(source_index=ref, containing_source_index=parent_ref,
                                  contained_bytes=len(text.encode()),
                                  offset=parent.index(text)))
        # Inventory exact word-aligned substrings of at least 64 characters.
        # Compare each source only with earlier (longer) strings, preventing
        # circular references. Union intervals so multiple matches cannot
        # inflate savings. This does not propose a renderer or admission rule.
        ordered = sorted(group.items(), key=lambda x: (-len(x[0]), x[0]))
        for index, (text, ref) in enumerate(ordered):
            if any(text in earlier for earlier, _ in ordered[:index]):
                continue  # Already counted in whole-content containment.
            words, starts, cursor = text.split(' '), [], 0
            for word in words:
                starts.append(cursor); cursor += len(word) + 1
            intervals = []
            for earlier, other_ref in ordered[:index]:
                for block in difflib.SequenceMatcher(None, words, earlier.split(' '), autojunk=False).get_matching_blocks():
                    if not block.size:
                        continue
                    start = starts[block.a]
                    end = starts[block.a + block.size - 1] + len(words[block.a + block.size - 1])
                    if end - start >= 64:
                        intervals.append((start, end))
            merged = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            if merged:
                partial.append(dict(source_index=ref, intervals=merged,
                                    overlap_bytes=sum(len(text[a:b].encode()) for a, b in merged)))
    # A display entry can represent several equal-text origins; count it once.
    unique = {p['source_index']: p for p in pairs}
    stats['contained_source_pairs'] = len(unique)
    stats['contained_content_bytes_upper_bound'] = sum(p['contained_bytes'] for p in unique.values())
    partial_unique = {p['source_index']: p for p in partial}
    stats['partial_overlap_sources'] = len(partial_unique)
    stats['partial_overlap_bytes_upper_bound'] = sum(p['overlap_bytes'] for p in partial_unique.values())
    return dict(summary=stats, contained_sources=list(unique.values()), fallback_matches=fallback,
                partial_overlaps=list(partial_unique.values()))


def main(args):
    run = args.run.resolve()
    root = run.parent
    require(run.name == 'control' and not args.output.exists(), 'Require control and fresh output')
    plan = json.loads((root / 'plan.json').read_text())
    audit = json.loads((root / 'retained-selection-audit.json').read_text())
    manifest = json.loads((run / 'isolated-manifest.json').read_text())
    same(audit['status'], 'passed', 'Original audit status')
    require(manifest['client_exit_code'] == manifest['audit_exit_code'] == 0
            and manifest['server_exit_code'] in (0, -15), 'Require terminal control')
    same(sha(root / 'plan.json'), audit['plan_sha256'], 'Plan identity')
    for name, digest in plan['source_files'].items():
        same(sha(root / 'source' / name), digest, 'Frozen file ' + name)
    paths = [run / 'server-search-debug.jsonl', run / 'completed/data/results/personamem-v2-32k.jsonl',
             run / 'funnel/stage-transport.json', run / 'isolated-manifest.json']
    for path in paths:
        same(sha(path), audit['input_sha256'][str(path)], 'Audited input ' + str(path))
    sys.path.insert(0, str(root / 'source/memory_system'))
    from app import listwise_sources, answer_context
    results = {r['search_id']: {k: r[k] for k in ('qa_id', 'conversation_id', 'packet_hash')}
               for r in records(paths[1])}
    same(len(results), plan['questions'], 'Complete results')
    rows, seen = [], set()
    with patch('socket.socket.connect', side_effect=AssertionError('Offline inventory')):
        for trace in records(paths[0]):
            sid = trace['search_id']
            require(sid in results and sid not in seen, 'Unexpected/duplicate search')
            seen.add(sid)
            same(trace['packet_hash'], results[sid]['packet_hash'], 'Packet identity')
            pool = {r['id']: r for r in trace['ranked']}
            groups = dict(main=trace['cascade']['listwise']['candidate_ids'],
                          fine=trace['cascade']['fine']['selected_ids'])
            values = {name: inventory([pool[mid] for mid in mids], listwise_sources, answer_context)
                      for name, mids in groups.items()}
            rows.append(dict(search_id=sid, **results[sid], stages=values))
            print(json.dumps(dict(qa_id=results[sid]['qa_id'],
                main=values['main']['summary'])), flush=True)
    same(seen, set(results), 'Full trace coverage')
    # Annotate existing stage gaps only after the complete lexical inventory.
    stages = {r['qa_id']: r for r in json.loads(paths[2].read_text())['rows']}
    for row in rows:
        visible = stages[row['qa_id']]['visible']
        row['missing_fine_to_main_turns'] = [i for i, (a, b) in enumerate(zip(
            visible['fine']['turn_coverage'], visible['listwise_input']['turn_coverage'], strict=True))
            if a['full'] and not b['full']]
    summary = {stage: {key: sum(r['stages'][stage]['summary'][key] for r in rows)
                       for key in rows[0]['stages'][stage]['summary']} for stage in ('main', 'fine')}
    for stage in summary:
        summary[stage]['questions_with_potential_savings'] = sum(
            bool(r['stages'][stage]['summary']['contained_content_bytes_upper_bound'] or
                 r['stages'][stage]['summary']['fallback_literal_bytes_upper_bound'] or
                 r['stages'][stage]['summary']['partial_overlap_bytes_upper_bound']) for r in rows)
    report = dict(status='complete_residual_source_inventory', questions=len(rows), summary=summary,
        rows=rows, provider_calls=0, production_changed=False, limitation=__doc__,
        input_sha256={str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name('audit_retained_porter_https.py'),
            root / 'plan.json', root / 'retained-selection-audit.json', *paths]})
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(questions=len(rows), summary=summary)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
