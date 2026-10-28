"""Paired whole-message versus exact-window source indexing, same live reader."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from replay_source_vector_route import source_route, validate_probe_granularity
from replay_query_context import records
from app import config, eval_scoring, evidence_packet


async def main(args):
    if config.FAKE:
        raise ValueError('Real reader required')
    probes = {name: json.loads(path.read_text()) for name, path in
              [('whole', args.whole), ('window', args.window)]}
    for probe in probes.values():
        validate_probe_granularity(probe)
        digest = hashlib.sha256(json.dumps([s['content'] for s in probe['sources']],
                                           ensure_ascii=False).encode()).hexdigest()
        if digest != probe['source_text_sha256']:
            raise ValueError('Source content hash mismatch')
    if (probes['whole']['sentence_window'] != 0 or probes['window']['sentence_window'] != 2
            or probes['whole']['sources'] != probes['window']['sources']
            or probes['whole']['questions'] != probes['window']['questions']):
        raise ValueError('Require matching full sources and queries, whole vs two-sentence keys')
    results = {r['search_id']: r for r in records(args.run / 'completed/data/results/personamem-v2-32k.jsonl')}
    traces = {r['search_id']: r for r in records(args.run / 'server-search-debug.jsonl')}
    if {q['search_id'] for q in probes['whole']['questions']} != set(results):
        raise ValueError('Full original cohort required')
    qa = {(d['conversation_id'], q['id']): q for d in records(args.prepared) for q in d['qa']}
    plan = dict(run=str(args.run), questions=len(results), whole=str(args.whole), window=str(args.window),
                probe_hashes={k: hashlib.sha256(p.read_bytes()).hexdigest()
                              for k,p in [('whole',args.whole),('window',args.window)]},
                intervention='Only source-key granularity changes. Top6/query RRF60top8, same original sources and packet budget.',
                caveat='Frozen queries and source index, live full reader. Offline experiment, not HTTPS scores.')
    with args.output.with_suffix('.plan.json').open('x') as stream:
        json.dump(plan, stream, indent=2)
    rows = []
    with args.output.open('x') as stream:
        for index, q in enumerate(probes['whole']['questions']):
            result, trace = results[q['search_id']], traces[q['search_id']]
            if (trace['query_specs'] != q['query_specs'] or trace['user_id'] != probes['whole']['user_id']
                    or evidence_packet.digest(trace['returned']) != result['packet_hash']):
                raise ValueError('Archived query/packet identity mismatch')
            for arm in (['whole', 'window'] if index % 2 == 0 else ['window', 'whole']):
                packet, route = source_route(probes[arm], q, trace['returned'], trace['evidence_token_budget'])
                with patch.object(config, 'CHOICE_QUERY_CONTEXT', False):
                    prediction, score, diagnostics = await eval_scoring.evaluate(
                        qa[result['conversation_id'], result['qa_id']], copy.deepcopy(packet))
                row = dict(qa_id=result['qa_id'], arm=arm, prediction=prediction, score=score,
                           baseline_packet_hash=result['packet_hash'], packet_hash=evidence_packet.digest(packet),
                           route=route, diagnostics=diagnostics)
                stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush()
                rows.append(row)
                print(json.dumps({k:row[k] for k in ('qa_id','arm','prediction','score')}), flush=True)
    pairs = {q['qa_id']: {r['arm']:r for r in rows if r['qa_id']==q['qa_id']}
             for q in probes['whole']['questions']}
    report = dict(partial=False, questions=len(pairs), rows=len(rows),
                  scores={arm:sum(p[arm]['score'] for p in pairs.values()) for arm in ('whole','window')},
                  wins=[q for q,p in pairs.items() if p['window']['score']>p['whole']['score']],
                  losses=[q for q,p in pairs.items() if p['window']['score']<p['whole']['score']],
                  caveat=plan['caveat'])
    args.output.with_suffix('.summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','whole','window','prepared','output'):
        p.add_argument('--'+name,type=Path,required=True)
    asyncio.run(main(p.parse_args()))
