"""Offline lexical/source admission ablation on a frozen experiment database.

Gold carrier IDs are used only to score returned IDs. This does not measure
semantic entailment, dense recall, final packets or answer quality.
"""
import argparse
import copy
import json
from pathlib import Path
import sqlite3
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, retrieval_queries, schemas, search_pipeline as search, store


def main(args):
    traces = {t['search_id']: t for t in map(json.loads,
        (args.run / 'completed/logs/search-debug.jsonl').open())}
    refs = {(r['conversation_id'], r['qa_id']): r for r in
        json.loads((args.run / 'funnel/stage-transport.json').read_text())['rows']}
    questions = json.loads((args.run / 'funnel/https-rows.json').read_text())
    st = store.Store(':memory:')
    with sqlite3.connect(f'file:{(args.run / "memory.db").resolve()}?mode=ro', uri=True) as src:
        src.backup(st.conn)
    variants = [('baseline', False, False), ('anchors', True, False),
                ('diversity', False, True), ('both', True, True)]
    out = []
    try:
        for q in questions:
            ref = refs[q['conversation_id'], q['qa_id']]
            trace = traces[q['search_id']]
            req = schemas.SearchRequest(user_id=trace['user_id'], query=trace['query'], options=trace['options'])
            limits = {k: trace['versions']['settings'][k] for k in ('RECALL_SOURCE_LIMIT', 'RECALL_FTS_LIMIT')}
            for name, anchor, diversity in variants:
                plan = copy.deepcopy(trace['plan'])
                plan.update(_include_history=trace['include_history'],
                            _include_sensitive=bool(config.SENSITIVE_RECALL_ENABLED and req.include_sensitive))
                with patch.multiple(config, RECALL_TOPIC_ANCHORS=anchor, SOURCE_RECALL_DIVERSITY=diversity, **limits), \
                     patch('socket.socket.connect', side_effect=AssertionError('Offline probe cannot call providers')):
                    specs = retrieval_queries.build(req.query, req.options, plan)
                    started = time.monotonic()
                    fts, sources = search._lexical_recall(st, req, plan, specs)
                    elapsed = time.monotonic() - started
                # Reference IDs never enter retrieval or selection.
                target = set(ref['carrier_ids']['memory'])
                ids = {m['id'] for m in fts + sources}
                out.append(dict(qa_id=q['qa_id'], conversation_id=q['conversation_id'], arm=name,
                    reference_present=bool(target), carrier_hit=bool(ids & target), queries=len(specs),
                    seconds=elapsed, source_count=len(sources)))
    finally:
        st.conn.close()
    baseline = {(r['conversation_id'], r['qa_id']): r for r in out if r['arm'] == 'baseline'}
    summary = {}
    for name, _, _ in variants:
        rows = [r for r in out if r['arm'] == name]
        summary[name] = dict(questions=len(rows), reference_present=sum(r['reference_present'] for r in rows),
            hits=sum(r['carrier_hit'] for r in rows), mean_queries=sum(r['queries'] for r in rows)/len(rows),
            gained=[r['qa_id'] for r in rows if r['carrier_hit'] and not baseline[r['conversation_id'],r['qa_id']]['carrier_hit']],
            lost=[r['qa_id'] for r in rows if not r['carrier_hit'] and baseline[r['conversation_id'],r['qa_id']]['carrier_hit']])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(dict(summary=summary, rows=out, limitation=__doc__), output, indent=2)
        output.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
