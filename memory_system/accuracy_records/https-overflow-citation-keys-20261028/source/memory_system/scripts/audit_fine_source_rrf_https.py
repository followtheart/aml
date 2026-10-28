"""Reconstruct source-score wire identity, fixed RRF, budgets and admission.

Offline only. Source transport and fixed ranking do not establish relevance or
answer quality. The stage auditor separately checks raw source/packet binding.
"""
import argparse
import ast
from collections import Counter
import copy
from fractions import Fraction
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

from batched_records import file_digest, json_records, physical_records
from audit_retained_porter_https import same, require
from probe_assistant_admission_order import replay
from source_nomination import wire_body
from rank_visible_catalog_audit import catalog as rank_catalog


def read(path):
    with path.open('rb') as stream:
        return json.loads(b''.join(iter(lambda: stream.read(4194304), b'')))


def selection_code(cascade):
    body = ast.parse(inspect.getsource(cascade.rank)).body[0].body
    start = next(i for i,n in enumerate(body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'failed' for t in n.targets))
    stop = next(i for i,n in enumerate(body) if isinstance(n, ast.If) and ast.unparse(n.test) == 'config.FINE_SOURCE_RRF')
    fragment = ast.Module(body=body[start:stop], type_ignores=[])
    same(hashlib.sha256(ast.dump(fragment).encode()).hexdigest(),
         '81d90fb573660b9596fd1f00b830b17109b0f1d85b3adf57e3df142a08e8e260', 'Frozen baseline selection AST')
    return compile(ast.fix_missing_locations(fragment), '<independent-baseline-replay>', 'exec')


def independent_order(coarse, source, scores, protected):
    ordered = sorted(coarse, key=lambda c:(c['_ce_score'] if c.get('_ce_score') is not None else -1e30,
                                           c.get('_fused', 0)), reverse=True)
    ids = [c['id'] for c in ordered]
    eligible = {mid:max(scores[k] for k in keys) for mid,keys in source['members'].items()
                if keys and mid not in protected}
    movable = [mid for mid in ids if mid in eligible]
    source_ids = sorted(movable, key=lambda mid:eligible[mid], reverse=True)
    whole_rank = {mid:i+1 for i,mid in enumerate(movable)}
    source_rank = {mid:i+1 for i,mid in enumerate(source_ids)}
    values = {mid:Fraction(1,60+whole_rank[mid])+Fraction(1,60+source_rank[mid]) for mid in movable}
    fused = iter(sorted(movable, key=lambda mid:values[mid], reverse=True))
    result = [next(fused) if mid in eligible else mid for mid in ids]
    return dict(status='ordered', reason=None, ordered_ids=result, source_scores=eligible,
                movable_ids=movable, pinned_ids=[mid for mid in ids if mid not in eligible],
                policy='equal_weight_rrf_k60', whole_candidate_ranks=whole_rank,
                user_source_ranks=source_rank, fused_ranks={mid:str(v) for mid,v in values.items()})


def guard_catalogs(baseline, proposed, answer_choice):
    guard = {}
    def identity(card):
        return tuple(card.get(k) for k in ('request_id', 'message_index', 'source_event_id', 'role'))
    for stage in ('fine', 'submitted'):
        before, after = [rank_catalog(value[stage]) for value in (baseline, proposed)]
        missing = [card for card in before if not any(identity(card) == identity(other)
                   and card['text'] in other['text'] for other in after)]
        guard[stage] = dict(original=before, proposed=after, missing=missing)
    return guard


def audit_trace(trace, modules):
    cascade, config, fine_evidence, schemas, source, ce, answer_choice = modules
    settings = dict(trace['versions']['settings'], LLM_MODEL=trace['versions']['model'],
                    CE_MODEL=trace['versions']['cross_encoder_model'], EMBED_MODEL=trace['versions']['embedding_model'])
    same((settings['LLM_MODEL'], settings['CE_MODEL'], settings['EMBED_MODEL']),
         ('gpt-4o-mini','qwen3-rerank','openai/text-embedding-v4'), 'Fixed models')
    c = trace['cascade']; details = c.get('fine_source_rrf')
    code = selection_code(cascade)
    totals = Counter(questions=1)
    with patch.multiple(config, **settings):
        base_trace = copy.deepcopy(trace)
        if details and 'original_headroom' in details:
            base_trace['cascade']['admission_budget'].update(details['original_headroom'])
        baseline, _, original_bytes = replay(base_trace, (cascade,config,fine_evidence,schemas), code)
        baseline['plan']['_cascade'] = baseline['trace']
        baseline['trace']['cross_encoder'] = copy.deepcopy(c['cross_encoder'])
        chosen = baseline
        if not config.FINE_SOURCE_RRF:
            require(details is None, 'Disabled stage absent')
            status = 'disabled'
        else:
            require(isinstance(details, dict) and 'original_headroom' in details, 'Source stage and preflight budget recorded')
            status = details['status']
            same(details.get('guard_policy'), 'actual_rank_visible_sources_v1', 'Guard visibility contract')
            require(status in ('skipped','fallback','selected'), 'Terminal source stage')
            head = details['original_headroom']
            job = source.prepare(baseline['req'], baseline['plan'], baseline['coarse'], baseline['submitted'], cascade, head)
            for key, expected in [('source_view_ids',list(job['source']['views'])), ('source_members',job['source']['members']),
                    ('protected_ids',job['protected_ids']), ('main_reservation',job['main_reservation']),
                    ('conservative_bound',job['conservative_bound']), ('phase_seconds',job['seconds']),
                    ('original_fine_ids',[v['id'] for v in baseline['fine']]),
                    ('original_main_ids',[v['id'] for v in baseline['submitted']])]:
                same(details[key], expected, 'Reconstructed '+key)
            if job['reason']:
                same((status,details['reason']), ('skipped',job['reason']), 'Ineligible stage skipped')
                require(not details['batches'] and not details['responses'], 'No dispatch when preflight fails')
            else:
                require(job['conservative_bound']+job['main_reservation'] <= head['token_headroom'], 'Full source budget preflight')
                responses = {event['batch']:event for event in details['responses']}
                same(len(responses), len(details['responses']), 'One response per batch')
                require(len(responses) <= len(details['batches']) <= len(job['batches']), 'Bounded source dispatches')
                scores = {}; spent = calls = 0; initial_tokens = initial_calls = None
                for i, batch in enumerate(details['batches']):
                    keys = job['batches'][i]
                    same((batch['index'],batch['view_ids']), (i,keys), 'Serial source batch identity')
                    docs = [job['source']['views'][key]['text'] for key in keys]
                    bound = ce.request_size(job['query'], docs)
                    expected_sha = source.digest(wire_body(config.CE_MODEL, config.CE_API_FORMAT, job['query'], docs))
                    same(batch['request_sha256'], expected_sha, 'Prepared wire request')
                    same(batch['input_bound'], bound, 'Conservative input bound')
                    if initial_tokens is None:
                        initial_tokens, initial_calls = batch['tokens_before'], batch['calls_before']
                    same(batch['tokens_before']-initial_tokens, spent, 'Serial token ledger')
                    same(batch['calls_before']-initial_calls, calls, 'Serial call ledger')
                    require(bound+spent+job['main_reservation'] <= head['token_headroom'], 'Main tokens protected for every batch')
                    require(calls+2 <= head['call_headroom'], 'Main call protected for every batch')
                    same(batch['headroom_before']['token_headroom'], head['token_headroom']-spent-job['main_reservation'], 'Reserved token headroom')
                    same(batch['headroom_before']['call_headroom'], head['call_headroom']-calls-1, 'Reserved call headroom')
                    charged = batch['tokens_after']-batch['tokens_before']
                    call_delta = batch['calls_after']-batch['calls_before']
                    require(charged >= 0 and call_delta in (0,1), 'One provider attempt, nonnegative usage')
                    if i in responses:
                        event = responses[i]
                        same(event['kind'], 'rerank', 'Only source rerank responses')
                        same(event['request_sha256'], expected_sha, 'Actual wire request hash')
                        usage = ce._usage(event['response'], bound)['usage']['total_tokens']
                        same(charged, usage, 'Raw response usage charged')
                        totals['archived_source_responses'] += 1
                    if batch['status'] == 'ok':
                        require(i in responses and call_delta == 1, 'Successful batch has one response and call')
                        values = ce._scores(responses[i]['response'], len(keys))
                        same(batch['scores'], values, 'Actual provider score identity')
                        scores.update(zip(keys, values, strict=True))
                    else:
                        require(status == 'fallback' and i == len(details['batches'])-1, 'Failure ends source scoring')
                    spent += charged; calls += call_delta
                    totals['started_source_batches'] += 1
                same((details['charged_tokens'],details['charged_calls']), (spent,calls), 'Total source charge')
                totals.update(accounted_source_tokens=spent, provider_calls=calls)
                complete = len(details['batches']) == len(job['batches']) and all(b['status']=='ok' for b in details['batches'])
                if not complete:
                    require(status=='fallback' and 'ordering' not in details, 'Incomplete score matrix retains baseline')
                elif 'ordering' in details:
                    ordering = independent_order(baseline['coarse'], job['source'], scores, job['protected_ids'])
                    same(details['ordering'], ordering, 'Independent fixed RRF reconstruction')
                    trial_trace = copy.deepcopy(base_trace)
                    trial_trace['cascade']['admission_budget'].update(details['remaining_budget'])
                    positions = {mid:len(ordering['ordered_ids'])-i for i,mid in enumerate(ordering['ordered_ids'])}
                    with patch.object(fine_evidence, 'key', lambda row:(positions[row['id']],0)):
                        proposed, _, _ = replay(trial_trace, (cascade,config,fine_evidence,schemas), code)
                    guard = guard_catalogs(baseline, proposed, answer_choice)
                    same(details['guard'], guard, 'Independent fine/main reader constraint guard')
                    if any(g['missing'] for g in guard.values()):
                        same((status,details['reason']), ('fallback','constraint_catalog_loss'), 'Whole-question constraint fallback')
                    else:
                        same((status,details['reason']), ('selected',None), 'Valid fixed proposal selected')
                        chosen = proposed
                        same(details['proposed_fine_ids'], [v['id'] for v in chosen['fine']], 'Recorded proposed fine')
                        same(details['proposed_main_ids'], [v['id'] for v in chosen['submitted']], 'Recorded proposed main')
                else:
                    require(status=='fallback' and details['reason'] in ('settled_deadline','TimeoutError'), 'Completed scores require selection or deadline fallback')
            same(details['remaining_budget']['token_headroom'], head['token_headroom']-details['charged_tokens'], 'Main token reservation released')
            same(details['remaining_budget']['call_headroom'], head['call_headroom']-details['charged_calls'], 'Main call reservation released')
            if status == 'selected':
                request = cascade.listwise_request(chosen['req'], chosen['plan'], chosen['submitted'])
                require(request['reservation'] <= details['remaining_budget']['token_headroom'], 'Settled main fits budget')
        same(chosen['trace']['fine'], c['fine'], 'Actual fine metadata and order')
        same(chosen['trace'].get('adaptive_admission'), c.get('adaptive_admission'), 'Actual adaptive main admission')
        same([v['id'] for v in chosen['submitted']], c['listwise']['candidate_ids'], 'Actual main order')
        same(chosen['admission_selection'], c['listwise']['selection'], 'Actual main selection metadata')
        same(sorted(chosen['carried_ids']), c['listwise']['carried_ids'], 'Actual carried reservations')
        same(sorted(chosen['baseline_ids']), c['listwise']['baseline_candidate_ids'], 'Actual baseline IDs')
        request = cascade.listwise_request(chosen['req'], chosen['plan'], chosen['submitted'])
        size = len(request['prompt'].encode()) if chosen['submitted'] else 0
        same(size, c['listwise']['input_bytes'], 'Actual main prompt bytes')
        require(size <= 24000 and len(chosen['fine']) <= 24 and len(chosen['submitted']) <= 24, 'Fixed capacity')
        return dict(search_id=trace['search_id'], status=status, original_main_bytes=original_bytes,
                    actual_main_bytes=size, totals=dict(totals))


def main(args):
    run = args.run.resolve(); plan = read(run.parent/'plan.json')
    same(plan['toggle'], 'FINE_SOURCE_RRF', 'RRF experiment')
    for name, digest in plan['source_files'].items():
        same(file_digest(run.parent/'source'/name), digest, 'Frozen source '+name)
    sys.path.insert(0, str(run.parent/'source/memory_system'))
    from app import cascade_rerank, config, fine_evidence, schemas, fine_source_rrf, cross_encoder, answer_choice
    modules = cascade_rerank,config,fine_evidence,schemas,fine_source_rrf,cross_encoder,answer_choice
    results = {r['search_id']:dict(qa_id=r['qa_id'],packet_hash=r['packet_hash'])
               for r in json_records(run/'completed/data/results/personamem-v2-32k.jsonl') if r.get('search_id')}
    rows=[]; seen=set(); totals=Counter()
    with patch('socket.socket.connect', side_effect=AssertionError('Offline audit')):
        for trace in json_records(run/'server-search-debug.jsonl'):
            sid=trace['search_id']; require(sid in results and sid not in seen, 'Unique completed search'); seen.add(sid)
            same(trace['packet_hash'],results[sid]['packet_hash'],'Result packet binding')
            row=audit_trace(trace,modules); row['qa_id']=results[sid]['qa_id']; rows.append(row); totals.update(row['totals'])
    same(seen,set(results),'All completed searches audited')
    ledger=[]
    for raw in physical_records(run/'server.log'):
        text=raw.decode(errors='replace')
        if 'provider_call ' not in text or 'stage=search.fine_source_rrf ' not in text: continue
        fields=dict(re.findall(r'(\w+)=([^\s]+)',text)); ledger.append(fields)
        require(fields['model']=='qwen3-rerank' and fields['attempt']=='1' and fields['retries']=='0','Fixed source provider, no retries')
    same(len(ledger),totals['provider_calls'],'Trace/ledger source call total')
    same(sum(int(r['total_tokens']) for r in ledger),totals['accounted_source_tokens'],'Trace/ledger token total')
    paths=[run.parent/'plan.json',run/'server-search-debug.jsonl',run/'server.log',run/'completed/data/results/personamem-v2-32k.jsonl']
    report=dict(status='passed',summary=dict(totals,statuses=dict(Counter(r['status'] for r in rows))),rows=rows,
                provider_calls_by_audit=0,limitation=__doc__,input_sha256={str(p):file_digest(p) for p in paths})
    with (run/'fine-source-rrf-audit.json').open('x') as stream: json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args())
