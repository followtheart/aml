"""Recheck lossless local candidate IDs on complete current HTTPS cohorts.

Reuses the previously tested ID encoding. This is not a new ranking rule.
Preserve every historical main input, then try whole omitted fine groups under
the original count/byte/token bounds. No provider calls or answer claims.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import cascade_rerank, config, fine_evidence, schemas
from probe_full_option_ce import prepare, sha
from probe_listwise_ids import compact_request
from evaluate_recall_context import reference_turns, packet_coverage


def main(args):
    with patch('socket.socket.connect', side_effect=AssertionError('Offline proposal')):
        jobs, manifest, conversations = prepare(args.run)
        fixed = []
        for job in jobs:
            t = job['t']; pool = {c['id']: c for c in t['ranked']}
            original = [pool[mid] for mid in t['cascade']['listwise']['candidate_ids']]
            fine = [pool[mid] for mid in t['cascade']['fine']['selected_ids']]
            positions = {c['id']: i for i, c in enumerate(fine)}
            plan = copy.deepcopy(t['plan'])
            plan['_coverage_requirements'] = [{k:r[k] for k in ('id','text','origin') if k in r}
                for r in t['coverage_manifest']['coverage']['requirements']]
            plan['_fusion_edges'] = t['fusion_edges']; plan['_fusion_triples'] = t['fusion_triples']
            plan['_fine_path_groups'] = t['cascade']['fine']['selection']['adaptive']['path_groups']
            req = schemas.SearchRequest(user_id=t['user_id'], query=t['query'], options=t['options'])
            with patch.multiple(config, **t['versions']['settings']):
                before = cascade_rerank.listwise_request(req, plan, original)
                assert len(before['prompt'].encode()) == t['cascade']['listwise']['input_bytes']
                compact = compact_request(req, plan, original)
                assert compact['schema'] == before['schema'] and compact['system'] == before['system']
                selected = list(original); original_ids = {c['id'] for c in original}
                by_fine = {c['id']:c for c in fine}
                headroom = t['cascade']['admission_budget']['token_headroom']
                limit = config.CASCADE_LLM_LIMIT + config.FINE_EXTRA_LIMIT
                byte_rejected = {o['id'] for o in t['cascade']['omitted']
                                 if o['stage']=='listwise' and o['reason']=='prompt_budget'}
                proposals = []
                for c in fine:
                    if c['id'] not in byte_rejected:
                        continue
                    group, missing = fine_evidence.closure(c, by_fine)
                    selected_ids = {v['id'] for v in selected}
                    extra = group - selected_ids
                    if missing or not extra:
                        continue
                    trial = sorted(selected + [v for v in fine if v['id'] in extra], key=lambda v:positions[v['id']])
                    request = compact_request(req, plan, trial)
                    accepted = (len(trial)<=limit and len(request['prompt'].encode())<=config.RERANK_MAX_PROMPT_BYTES
                                and (headroom is None or request['reservation']<=headroom))
                    proposals.append(dict(candidate_id=c['id'], group_ids=sorted(group),
                        new_ids=sorted(extra), prompt_bytes=len(request['prompt'].encode()),
                        reservation=request['reservation'], accepted=accepted))
                    if accepted:
                        selected = trial
                after = compact_request(req, plan, selected)
                assert original_ids <= {c['id'] for c in selected}
                assert [c['id'] for c in selected if c['id'] in original_ids] == [c['id'] for c in original]
                assert len(after['prompt'].encode()) <= config.RERANK_MAX_PROMPT_BYTES
                assert headroom is None or after['reservation']<=headroom
            fixed.append(dict(job=job, original=original, selected=selected, proposals=proposals,
                before_bytes=len(before['prompt'].encode()), compact_fixed_bytes=len(compact['prompt'].encode()),
                after_bytes=len(after['prompt'].encode()), before_reservation=before['reservation'],
                after_reservation=after['reservation'], prompt_sha256=hashlib.sha256(after['prompt'].encode()).hexdigest()))
    # All proposed inputs are fixed before reference labels are read.
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream: gold = list(csv.DictReader(stream))
    raw = {}
    with sqlite3.connect((args.run/'memory.db').resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory=sqlite3.Row
        for row in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(row['user_id'], {})[row['request_id'],row['message_index']] = dict(row)
    convs={c['conversation_id']:c for c in conversations}; rows=[]
    for f in fixed:
        job=f.pop('job'); original=f.pop('original'); selected=f.pop('selected')
        conv=convs[job['conversation_id']]; qa=next(q for q in conv['qa'] if q['id']==job['qa_id'])
        turns=reference_turns(qa,conv['source_history'],gold[int(job['qa_id'])])
        rows.append(dict(qa_id=job['qa_id'],conversation_id=job['conversation_id'],**f,
            original_ids=[c['id'] for c in original], selected_ids=[c['id'] for c in selected],
            original=packet_coverage(turns,[c['_packet_item'] for c in original],raw[job['t']['user_id']]),
            proposed=packet_coverage(turns,[c['_packet_item'] for c in selected],raw[job['t']['user_id']])))
    summary=dict(questions=len(rows),reproduced_fine_and_main=len(rows),provider_calls=0,
        changed_members=sum(r['original_ids']!=r['selected_ids'] for r in rows),
        added_candidates=sum(len(r['selected_ids'])-len(r['original_ids']) for r in rows),
        total_fixed_prompt_savings=sum(r['before_bytes']-r['compact_fixed_bytes'] for r in rows),
        reference_questions={arm:sum(r[arm]['visible'] for r in rows) for arm in ('original','proposed')},
        user_turns={arm:sum(r[arm]['full_turns'] for r in rows) for arm in ('original','proposed')},
        gained=[r['qa_id'] for r in rows if r['proposed']['visible'] and not r['original']['visible']],
        lost=[r['qa_id'] for r in rows if r['original']['visible'] and not r['proposed']['visible']])
    result=dict(summary=summary, rows=rows, limitation=__doc__,inputs={str(p.resolve()):sha(p) for p in (
        Path(__file__),Path(__file__).with_name('probe_listwise_ids.py'),args.run/'server-search-debug.jsonl',
        args.run/'completed/data/results/personamem-v2-32k.jsonl',Path(manifest['data']),args.gold_csv)})
    with args.output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(parser.parse_args())
