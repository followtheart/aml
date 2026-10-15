"""Read both saved capacity arms without rerunning ranking or filtering questions."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, eval_scoring, evidence_packet, run_metadata


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def main(args):
    if config.FAKE:
        raise ValueError('Real provider required')
    plan = json.loads(args.packets.with_suffix('.plan.json').read_text())
    if plan['repeat'] != 1:
        raise ValueError('Require one complete ranking pair per question')
    source_run = Path(plan['run'])
    audit = json.loads((source_run / 'funnel/https-summary.json').read_text())
    if audit['partial'] or audit['exit_code'] != 0:
        raise ValueError('Require complete successful source experiment')
    if hashlib.sha256(args.prepared.read_bytes()).hexdigest() != audit['prepared_sha256']:
        raise ValueError('Prepared dataset changed')
    qa = {q['id']: q for d in records(args.prepared) for q in d['qa']}
    baseline = {r['qa_id']: r for r in records(source_run / 'completed/data/results/personamem-v2-32k.jsonl')}
    if set(qa) != set(plan['questions']) or set(qa) != set(baseline):
        raise ValueError('Require complete planned question cohort')
    packets = {}
    for row in records(args.packets):
        key = row['qa_id'], row['capacity_limit']
        if key in packets or key[0] not in qa or key[1] not in (None, plan['capacity_limit']):
            raise ValueError('Duplicate or unexpected capacity arm')
        if (row['repetition'] != 1 or not row['frozen_packet_verified']
                or row['baseline_packet_hash'] != baseline[key[0]]['packet_hash']
                or evidence_packet.digest(row['packet']) != row['packet_hash']):
            raise ValueError('Packet identity mismatch')
        packets[key] = row
    if set(packets) != {(q, c) for q in qa for c in (None, plan['capacity_limit'])}:
        raise ValueError('Incomplete ranking experiment')
    metadata = dict(packets=str(args.packets), prepared=str(args.prepared),
                    packets_sha256=hashlib.sha256(args.packets.read_bytes()).hexdigest(),
                    questions=plan['questions'], versions=run_metadata.versions(),
                    caveat='Same reader configuration, different saved ranking packets. All reader calls live; upstream ranking frozen. Not fresh HTTPS.')
    args.output.with_suffix('.plan.json').write_text(json.dumps(metadata, indent=2)+'\n')
    results = []
    with args.output.open('x') as stream:
        for index, qid in enumerate(plan['questions']):
            arms = [None, plan['capacity_limit']]
            for capacity in arms if index % 2 == 0 else reversed(arms):
                packet = packets[qid, capacity]
                prediction, score, diagnostics = await eval_scoring.evaluate(qa[qid], copy.deepcopy(packet['packet']))
                row = dict(qa_id=qid, conversation_id=baseline[qid]['conversation_id'],
                           capacity_limit=capacity, packet_hash=packet['packet_hash'],
                           prediction=prediction, score=score, diagnostics=diagnostics)
                results.append(row)
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                stream.flush()
                print(json.dumps({k: row[k] for k in ['qa_id', 'capacity_limit', 'prediction', 'score']}), flush=True)
    summary = dict(count=len(qa), control=sum(r['score'] for r in results if r['capacity_limit'] is None),
                   treatment=sum(r['score'] for r in results if r['capacity_limit'] is not None),
                   errors=[dict(qa_id=r['qa_id'], capacity_limit=r['capacity_limit']) for r in results
                           if r['diagnostics'].get('answer_status') == 'error'])
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['packets', 'prepared', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
