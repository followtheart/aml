"""Use residual original context capacity for newly recalled fine candidates.

Keep every original packet item and original lexical seat. Nominate only new
memory IDs from measured source-route variants that survived real CE and fine
selection. Preserve that fine order, dependency eligibility, full prepared
sources and the original 5120-byte reserve; append at most one whole carrier.
No new provider calls. Frozen upstream costs/headroom are diagnostic only.
Exchange nominations must retain every indexed member in the visible union.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import main as score
from app import config, evidence_packet, fine_evidence, search_pipeline
from app.fine_context_lexical import propose as lexical
from app.fine_context_reserve import propose as reserve

MODES = ('whole', 'window2', 'exchange')


def preparer(measured, exchanges, nominations):
    def prepare(t):
        row = measured[t['search_id']]
        if row['user_id'] != t['user_id']:
            raise ValueError('Cross-user measurement')
        ctx = t['cascade']['conversation_context']
        if ctx['policy'] != 'fine_user_lexical_context_v1':
            raise ValueError('Require original lexical policy')
        base = [dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
        if evidence_packet.digest(base) != ctx['base_packet_hash']:
            raise ValueError('Base hash differs')
        pool = {c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
                if c.get('view_status') != 'stale' and '_packet_item' in c}
        fine = [pool[mid] for mid in t['cascade']['fine']['selected_ids']
                if mid in pool and mid not in t['selection']['selected_ids']]
        kwargs = dict(user_id=t['user_id'],top_k=t['top_k'],byte_budget=t['coverage_manifest']['token_budget'])
        prefix, original = lexical(base,fine,**kwargs,reserve_bytes=5120,query=t['query'],options=t['options'])
        if [dict(p,packet_hash=evidence_packet.digest(prefix)) for p in prefix] != t['returned']:
            raise ValueError('Original final packet differs')
        remaining = 5120-original['added_bytes']
        kept = {p['id'] for p in prefix}
        old_ids = {c['id'] for c in t['ranked']}
        def full_key(s):
            if not isinstance(s.get('content'), str) or not s['content'].strip():
                return None
            span = s.get('content_span')
            if span and not (span.get('start') == 0 and span.get('end') == span.get('original_length') == len(s['content'])):
                return None
            return s['request_id'], s['message_index'], s['role'], s['content']
        known = {key for p in prefix for s in p['sources'] if (key := full_key(s))}
        packets, proposals, inventory = {'baseline':prefix}, {}, []
        for mode in MODES:
            candidates = row['variants'][mode]['fine']
            available = dict(pool,**{c['id']:c for c in candidates})
            eligible = []
            for c in candidates:
                if c['id'] in old_ids:
                    continue
                group, missing = fine_evidence.closure(c,available)
                admitted = not missing and group <= kept | {c['id']}
                context_visible = True
                if mode == 'exchange':
                    visible = known | {key for s in c['_packet_item']['sources'] if (key := full_key(s))}
                    matched = [members for si,members in exchanges.items()
                        if si in nominations[t['search_id']] and c['id'] in members[0]['amu_ids']]
                    context_visible = any({full_key(s) for s in members} <= visible for members in matched)
                    admitted = admitted and context_visible
                inventory.append(dict(mode=mode,candidate_id=c['id'],ce_score=c['_ce_score'],
                    eligible=admitted,indexed_context_visible=context_visible,
                    group_ids=sorted(group),missing=sorted(missing)))
                if admitted:
                    eligible.append(c)
            packets[mode], proposals[mode] = reserve(prefix,eligible,**kwargs,reserve_bytes=remaining,limit=1)
            if packets[mode][:len(prefix)] != prefix:
                raise ValueError('Original prefix changed')
        return dict(packets=packets,proposals=proposals,inventory=inventory,remaining_bytes=remaining)
    return prepare


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'))
    args = p.parse_args()
    root = args.run
    ce = root/'source-upstream-ce-diagnostic'
    summary = json.loads((ce/'summary.json').read_text())
    selections = ce/'selections.jsonl'
    with selections.open('rb') as f:
        measured_sha = hashlib.file_digest(f,'sha256').hexdigest()
    if measured_sha != summary['selections_sha256'] or (ce/'exit-code').read_text().strip() != '0':
        raise ValueError('Require completed unchanged real CE measurement')
    ids = {j['qa']['id']:j['payload']['search_id'] for j in map(json.loads,(root/'http-reader-inputs.jsonl').open())}
    measured = {ids[r['qa_id']]:r for r in map(json.loads,selections.open())}
    exchange_path = root/'exchange-vector-diagnostic/plan.json'
    exchange_plan = json.loads(exchange_path.read_text())
    exchanges = {int(si):members for si,members in exchange_plan['exchange_sources'].items()}
    nomination_path = root/'source-upstream-diagnostic/selections.jsonl'
    nominations = {r['search_id']:r['nominations']['exchange']['source_indices']
        for r in map(json.loads,nomination_path.open())}
    if len(measured) != 56:
        raise ValueError('Require full cohort')
    args.runs = [root/'control']
    with patch('socket.socket.connect',side_effect=AssertionError('Offline residual proposal')):
        score(args,preparer=preparer(measured,exchanges,nominations),modes=MODES,limitation=__doc__,script_path=Path(__file__))
    result = json.loads(args.output.read_text())
    for s in result['summary']:
        mode=s['mode']
        s['complete_sets']={m:sum(bool(r['coverage'][m]['user']) and all(r['coverage'][m]['user'])
            for r in result['rows']) for m in ('baseline',mode)}
    result['upstream_ce_selections_sha256']=measured_sha
    result['exchange_plan_sha256']=hashlib.sha256(exchange_path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
