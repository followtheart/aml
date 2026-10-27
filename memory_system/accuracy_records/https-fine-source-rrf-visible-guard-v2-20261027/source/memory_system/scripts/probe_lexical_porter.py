"""Offline, full-cohort lexical replay with SQLite's standard Porter tokenizer.

Only copied FTS indices change. Query terms, caps, joins and source allocation
are production code. Historical nonlexical routes remain fixed, so union
coverage is a diagnostic, not a live fusion, final recall or answer result.
References enter only after both complete candidate allocations are fixed.
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reindex(st):
    for name, columns in (
        ('amu_fts', 'amu_id UNINDEXED, user_id UNINDEXED, content, retrieval_key'),
        ('source_fts', 'request_id UNINDEXED, message_index UNINDEXED, user_id UNINDEXED, content'),
    ):
        # Keep indexed rows, row IDs and text identical, including duplicate rows.
        before = list(map(tuple, st.conn.execute(f'SELECT rowid,* FROM {name} ORDER BY rowid')))
        st.conn.execute(f'DROP TABLE {name}')
        st.conn.execute(f"CREATE VIRTUAL TABLE {name} USING fts5({columns}, tokenize='porter unicode61')")
        st.conn.executemany(f'INSERT INTO {name}(rowid,{",".join(c.split()[0] for c in columns.split(","))}) VALUES ({",".join("?" for _ in before[0])})', before)
        after = list(map(tuple, st.conn.execute(f'SELECT rowid,* FROM {name} ORDER BY rowid')))
        if before != after:
            raise ValueError('Index text or row identities changed')
    st.conn.commit()


def request(t):
    req = schemas.SearchRequest(user_id=t['user_id'], query=t['query'], options=t['options'],
                                include_sensitive=t['requested_include_sensitive'])
    plan = dict(copy.deepcopy(t['plan']), _include_history=t['include_history'],
                _include_sensitive=t['include_sensitive'])
    specs = [s for s in t['query_specs'] if s['origin'] != 'grounded_followup']
    return req, plan, specs


def replay(st, t):
    req, plan, specs = request(t)
    lanes = {'full_text': [], 'source_text': []}
    original_fts, original_source = st.fts_search, st.source_search

    def capture(method, channel):
        def wrapped(*args, **kwargs):
            candidates = method(*args, **kwargs)
            lanes[channel].append([dict(id=c['id'], score=c['_score']) for c in candidates])
            return candidates
        return wrapped

    st.read_user_id = t['user_id']
    with patch.multiple(config, **t['versions']['settings']), \
            patch.object(st, 'fts_search', capture(original_fts, 'full_text')), \
            patch.object(st, 'source_search', capture(original_source, 'source_text')):
        merged = search._lexical_recall(st, req, plan, specs)
    return dict(merged={name: [c['id'] for c in cs]
                       for name, cs in zip(('full_text', 'source_text'), merged)}, lanes=lanes)


def main(args):
    root = args.run.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (root / 'exit-code').read_text().strip() != '0':
        raise ValueError('Requires a terminal successful arm')
    inputs = [root / 'memory.db', root / 'server-search-debug.jsonl',
              root / 'completed/data/results/personamem-v2-32k.jsonl',
              root / 'funnel/stage-transport.json']
    hashes = {str(p): sha(p) for p in inputs}
    results = records(inputs[2])
    traces = {t['search_id']: t for t in records(inputs[1])}
    if len(results) != args.expected_count or len({r['search_id'] for r in results}) != len(results):
        raise ValueError('Incomplete or duplicate cohort')
    jobs = {r['search_id']: dict(search_id=r['search_id'], variants={}) for r in results}
    st = store.Store(':memory:')
    with sqlite3.connect(inputs[0].as_uri() + '?mode=ro', uri=True) as source:
        source.backup(st.conn)
    try:
        with patch('socket.socket.connect', side_effect=AssertionError('Offline diagnostic')):
            for mode in ('baseline', 'porter'):
                if mode == 'porter':
                    reindex(st)
                for index, (sid, job) in enumerate(jobs.items()):
                    t = traces[sid]
                    candidate = replay(st, t)
                    if mode == 'baseline':
                        for channel, ids in candidate['merged'].items():
                            observed = [r for r in t['routes'] if r['channel'] == channel and r['round'] == 1]
                            if len(observed) != 1 or ids != [c['id'] for c in observed[0]['candidates']]:
                                raise ValueError('Historical route mismatch: ' + sid + '/' + channel)
                    job['variants'][mode] = candidate
                    if (index + 1) % 10 == 0:
                        print(f'{mode}: {index + 1}/{len(jobs)}', flush=True)
            # Candidate allocation is complete before loading reference labels.
            stage = json.loads(inputs[3].read_text())
            if stage['summary']['progress_snapshot'] or len(stage['rows']) != len(jobs):
                raise ValueError('Incomplete reference audit')
            refs = {(r['conversation_id'], r['qa_id']): r for r in stage['rows']}
            missing = []
            for result in results:
                job = jobs[result['search_id']]
                t = traces[result['search_id']]
                ref = refs[result['conversation_id'], result['qa_id']]
                target = set(ref['carrier_ids']['memory'])
                job.update(qa_id=result['qa_id'], reference_carriers=sorted(target))
                other = {c['id'] for r in t['routes']
                         if r['channel'] not in ('source_text', 'full_text') or r['round'] != 1
                         for c in r['candidates']}
                job['historical_other_ids'] = sorted(other)
                job['metrics'] = {}
                for mode, variant in job['variants'].items():
                    lexical = set().union(*(set(ids) for ids in variant['merged'].values()))
                    job['metrics'][mode] = dict(lexical=bool(target & lexical),
                        union_other=bool(target & (lexical | other)),
                        union_carriers=sorted(target & (lexical | other)))
                if target and not ref['recall_route_carrier_ids']:
                    missing.append(job)
            # Diagnose misses at unlimited per-query depth with the SAME joins
            # and source diversity. This does not change either proposal.
            for mode in ('porter', 'baseline'):
                if mode == 'baseline':
                    with sqlite3.connect(inputs[0].as_uri() + '?mode=ro', uri=True) as source:
                        source.backup(st.conn)
                for job in missing:
                    t = traces[job['search_id']]
                    req, plan, specs = request(t)
                    st.read_user_id = t['user_id']
                    target = set(job['reference_carriers'])
                    depth = []
                    with patch.multiple(config, **t['versions']['settings']):
                        for spec in specs:
                            row = dict(query=spec['text'], origin=spec['origin'])
                            for channel, method in (('full_text', st.fts_search), ('source_text', st.source_search)):
                                cs = method(req.user_id, spec['text'], 100000,
                                    include_history=plan['_include_history'],
                                    include_sensitive=plan['_include_sensitive'], include_cold=True)
                                if len(cs) == 100000:
                                    raise ValueError('Diagnostic depth was exhausted')
                                row[channel] = dict(count=len(cs), target_ranks={c['id']: i + 1
                                    for i, c in enumerate(cs) if c['id'] in target})
                            depth.append(row)
                    job.setdefault('unlimited_diagnostic', {})[mode] = depth
    finally:
        st.conn.close()
    if hashes != {str(p): sha(p) for p in inputs}:
        raise ValueError('Source artifacts changed during replay')
    summary = dict(questions=len(jobs), reproduced_initial_lexical_routes=2 * len(jobs), variants={})
    for mode in ('baseline', 'porter'):
        summary['variants'][mode] = dict(changed_questions=sum(
            j['variants'][mode]['merged'] != j['variants']['baseline']['merged'] for j in jobs.values()))
        for metric in ('lexical', 'union_other'):
            summary['variants'][mode][metric] = dict(count=sum(j['metrics'][mode][metric] for j in jobs.values()),
                gained=[j['qa_id'] for j in jobs.values() if j['metrics'][mode][metric] and not j['metrics']['baseline'][metric]],
                lost=[j['qa_id'] for j in jobs.values() if not j['metrics'][mode][metric] and j['metrics']['baseline'][metric]])
        summary['variants'][mode]['lost_carriers'] = [dict(qa_id=j['qa_id'], ids=sorted(
            set(j['metrics']['baseline']['union_carriers']) - set(j['metrics'][mode]['union_carriers'])))
            for j in jobs.values() if set(j['metrics']['baseline']['union_carriers']) - set(j['metrics'][mode]['union_carriers'])]
    output = dict(summary=summary, rows=list(jobs.values()), missing_route_questions=[j['qa_id'] for j in missing],
        input_hashes=hashes, script_sha256=sha(Path(__file__)), provider_calls=0,
        limitation=__doc__, tokenizer='porter unicode61')
    with args.output.open('x') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, required=True)
    main(parser.parse_args())
