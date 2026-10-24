"""Replay source recall and diagnose allocation across overlapping source sets.

Offline only. Reference carrier IDs are scored after allocation; source linkage
is not visible text, semantic support, final recall, or answer correctness.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, schemas, search_pipeline as search, store


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def novel_sources(routes, limit, source_keys):
    """Inspect candidates with a new source before those fully represented."""
    ordered = search._merge_query_results(routes, sum(map(len, routes)))
    selected, deferred, seen = [], [], set()
    for item in ordered:
        keys = set(source_keys.get(item['id'], ()))
        if not keys or keys - seen:
            selected.append(item)
            seen.update(keys)
        else:
            deferred.append(item)
    return (selected + deferred)[:limit]


def question_prefix(routes, limit, source_keys, count):
    """Reserve a bounded original-question prefix, then existing allocation."""
    prefix = routes[0][:min(count, limit)] if routes else []
    used = {c['id'] for c in prefix}
    rest = [[c for c in lane if c['id'] not in used] for lane in routes]
    return prefix + search._merge_source_query_results(rest, limit-len(prefix), source_keys)


def bounded_novel_replacements(routes, baseline, source_keys, max_replacements):
    """Offline alternative: preserve baseline order; replace redundant sets last.

    A redundant source set is not necessarily redundant prepared text. This
    checks only original-message coverage; downstream evaluation is required.
    """
    selected=list(baseline)
    ordered=search._merge_query_results(routes,sum(map(len,routes)))
    added=set()
    for candidate in ordered:
        if len(added)>=max_replacements:
            break
        ids={c['id'] for c in selected}
        if candidate['id'] in ids:
            continue
        covered=set().union(*(set(source_keys.get(mid,())) for mid in ids))
        if not set(source_keys.get(candidate['id'],()))-covered:
            continue
        victim=None
        for current in reversed(selected):
            mid=current['id'];keys=set(source_keys.get(mid,()))
            others=set().union(*(set(source_keys.get(other,())) for other in ids-{mid}))
            if mid not in added and keys and keys<=others:
                victim=current
                break
        if victim is None:
            continue
        selected.remove(victim)
        selected.append(candidate)
        added.add(candidate['id'])
        new_coverage=set().union(*(set(source_keys.get(c['id'],())) for c in selected))
        if not covered<new_coverage:
            raise ValueError('Bounded replacement did not strictly increase source coverage')
    return selected


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.run/'exit-code').read_text().strip() != '0':
        raise ValueError('Requires a completed successful arm')
    stage = json.loads((args.run/'funnel/stage-transport.json').read_text())
    if stage['summary']['progress_snapshot']:
        raise ValueError('Requires a final stage audit')
    refs = {(r['conversation_id'], r['qa_id']): r for r in stage['rows']}
    traces = {t['search_id']: t for t in records(args.run/'server-search-debug.jsonl')}
    st = store.Store(':memory:')
    with sqlite3.connect((args.run/'memory.db').resolve().as_uri()+'?mode=ro', uri=True) as src:
        src.backup(st.conn)
    rows = []
    try:
        for result in records(args.run/'completed/data/results/personamem-v2-32k.jsonl'):
            t = traces[result['search_id']]
            ref = refs[result['conversation_id'], result['qa_id']]
            req = schemas.SearchRequest(user_id=t['user_id'], query=t['query'], options=t['options'])
            if 'requested_include_sensitive' in t:
                req.include_sensitive=t['requested_include_sensitive']
            sensitive=t['versions']['settings'].get('SENSITIVE_RECALL_ENABLED')
            if sensitive is None:
                if args.legacy_sensitive_recall is None:
                    raise ValueError('Historical trace lacks sensitive policy; specify --legacy-sensitive-recall on/off')
                sensitive=args.legacy_sensitive_recall=='on'
            elif args.legacy_sensitive_recall is not None:
                raise ValueError('Do not override a recorded sensitive policy')
            plan = copy.deepcopy(t['plan'])
            plan.update(_include_history=t['include_history'],
                        _include_sensitive=bool(sensitive and req.include_sensitive))
            specs = [s for s in t['query_specs'] if s['origin'] != 'grounded_followup']
            observed = [r for r in t['routes'] if r['channel']=='source_text' and r['round']==1]
            if len(observed) != 1 or not specs or specs[0]['origin'] != 'question':
                raise ValueError('Unexpected source route or query ordering')
            capture = {}
            original = search._merge_source_query_results

            def capture_merge(routes, limit, source_keys, **kwargs):
                capture.update(routes=routes, limit=limit, source_keys=source_keys)
                return original(routes, limit, source_keys, **kwargs)

            with patch.multiple(config, **dict(t['versions']['settings'],SENSITIVE_RECALL_ENABLED=sensitive)), \
                    patch('socket.socket.connect', side_effect=AssertionError('Offline')), \
                    patch.object(search, '_merge_source_query_results', capture_merge):
                if not config.SOURCE_RECALL_DIVERSITY:
                    raise ValueError('Requires baseline source diversity')
                fts, baseline = search._lexical_recall(st, req, plan, specs)
            baseline_ids = [c['id'] for c in baseline]
            if baseline_ids != [c['id'] for c in observed[0]['candidates']]:
                raise ValueError('Source route did not reproduce: '+result['qa_id'])
            observed_fts = [r for r in t['routes'] if r['channel']=='full_text' and r['round']==1]
            if len(observed_fts)!=1:
                raise ValueError('Unexpected FTS route count')
            fts_matches = [c['id'] for c in fts] == [c['id'] for c in observed_fts[0]['candidates']]
            if not fts_matches:
                raise ValueError('FTS route did not reproduce: '+result['qa_id'])
            routes, limit, keys = (capture[k] for k in ('routes', 'limit', 'source_keys'))
            proposals = dict(baseline=baseline, novel_sources=novel_sources(routes, limit, keys))
            if original(routes, limit, keys, novel_first=True) != proposals['novel_sources']:
                raise ValueError('Production novelty allocation differs from offline proposal')
            for count in (8,16):
                proposals[f'question_prefix_{count}'] = question_prefix(routes, limit, keys, count)
            for count in (1,2,4):
                proposals[f'novel_replace_{count}']=bounded_novel_replacements(routes,baseline,keys,count)
            target = set(ref['carrier_ids']['memory'])
            other = {c['id'] for r in t['routes'] if r['channel']!='source_text'
                     for c in r['candidates']}
            all_sources = {c['id'] for r in t['routes'] if r['channel']=='source_text' and r['round']!=1
                           for c in r['candidates']}
            variants = {}
            for name, candidates in proposals.items():
                ids = [c['id'] for c in candidates]
                if len(ids)>limit or len(set(ids))!=len(ids):
                    raise ValueError('Source allocation exceeded count or duplicated IDs')
                variants[name] = dict(ids=ids, carrier_ids=sorted(set(ids)&target),
                    source_hit=bool(set(ids)&target),
                    union_with_historical_other_routes_hit=bool((set(ids)|other|all_sources)&target),
                    distinct_source_keys=len(set().union(*(set(keys.get(mid,())) for mid in ids))))
            rows.append(dict(conversation_id=result['conversation_id'], qa_id=result['qa_id'],
                reference_present=bool(target), query_count=len(specs), fts_replay_matches=fts_matches,
                sensitive_recall_enabled=sensitive, effective_include_sensitive=plan['_include_sensitive'],
                per_query_target_ids=[[c['id'] for c in r if c['id'] in target] for r in routes],
                per_query_target_ranks=[{c['id']:i+1 for i,c in enumerate(r) if c['id'] in target}
                                        for r in routes],
                variants=variants))
    finally:
        st.conn.close()
    summary = dict(questions=len(rows), reproduced_source_routes=len(rows),
        reproduced_fts_routes=sum(r['fts_replay_matches'] for r in rows), variants={})
    for name in rows[0]['variants']:
        summary['variants'][name] = dict(candidates=sum(len(r['variants'][name]['ids']) for r in rows))
        for metric in ('source_hit','union_with_historical_other_routes_hit'):
            summary['variants'][name][metric] = dict(count=sum(r['variants'][name][metric] for r in rows),
                gained=[r['qa_id'] for r in rows if r['variants'][name][metric] and not r['variants']['baseline'][metric]],
                lost=[r['qa_id'] for r in rows if not r['variants'][name][metric] and r['variants']['baseline'][metric]])
    with args.output.open('x') as stream:
        json.dump(dict(summary=summary, rows=rows, limitation=__doc__,
            legacy_sensitive_policy=args.legacy_sensitive_recall,
            source_trace_sha256=hashlib.sha256((args.run/'server-search-debug.jsonl').read_bytes()).hexdigest(),
            union_caveat='Other historical routes are held fixed only for an upper-stage diagnostic; FTS replay is checked and reported separately, not substituted for the historical route. A live source-route change can alter graph seeds, follow-up, fusion, and downstream decisions.'), stream, indent=2)
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy-sensitive-recall',choices=('on','off'),
        help='Explicit deployment policy for old traces that did not record this setting')
    main(parser.parse_args())
