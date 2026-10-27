"""Reconstruct each live source admission from its own baseline and wire hashes.

No model calls. Coverage and syntactic admission checks are not answer accuracy
or proof of the semantic reviewer's judgment.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

from batched_records import file_digest, json_records, physical_records
from audit_retained_porter_https import selection_code, same, require
from probe_assistant_admission_order import replay
from source_nomination import wire_body


def read(path):
    with path.open('rb') as f:
        return json.loads(b''.join(iter(lambda: f.read(4194304), b'')))


def gate_wire(config, gate):
    name = 'emit_json_result'
    return dict(model=config.LLM_MODEL,
        messages=[dict(role='system', content=gate['system']), dict(role='user', content=gate['prompt'])],
        tools=[dict(type='function', function=dict(name=name,
            description='Return the requested structured result.', parameters=gate['schema']))],
        parallel_tool_calls=False, tool_choice=dict(type='function', function=dict(name=name)),
        temperature=config.LLM_TEMPERATURE,
        max_tokens=min(gate['max_tokens'], config.LLM_JSON_MAX_TOKENS), num_retries=0)


def gate_payload(raw, llm):
    calls = raw['choices'][0]['message'].get('tool_calls') or []
    require(len(calls) == 1 and calls[0]['function']['name'] == 'emit_json_result', 'Complete gate tool response')
    arguments = calls[0]['function']['arguments']
    return arguments if isinstance(arguments, dict) else llm.extract_json(arguments)


def audit_trace(trace, modules):
    cascade, config, fine_evidence, schemas, source, review, policy, ce, llm = modules
    settings = dict(trace['versions']['settings'], LLM_MODEL=trace['versions']['model'],
                    CE_MODEL=trace['versions']['cross_encoder_model'], EMBED_MODEL=trace['versions']['embedding_model'])
    require((settings['LLM_MODEL'], settings['CE_MODEL'], settings['EMBED_MODEL']) ==
            ('gpt-4o-mini', 'qwen3-rerank', 'openai/text-embedding-v4'), 'Fixed models')
    code, _ = selection_code(cascade)
    c = trace['cascade']; details = c.get('option_margin_source')
    totals = Counter(questions=1)
    with patch.multiple(config, **settings):
        baseline, _, original_bytes = replay(trace, (cascade, config, fine_evidence, schemas), code)
        req, plan = baseline['req'], baseline['plan']
        plan['_cascade'] = baseline['trace']
        old_fine, old_main = baseline['fine'], baseline['submitted']
        fine, main, selected = old_fine, old_main, False
        if not config.OPTION_MARGIN_SOURCE_ADMISSION:
            require(details is None, 'Default-off source stage absent')
            status = 'disabled'
        else:
            require(isinstance(details, dict), 'Enabled source stage recorded')
            status = details['status']
            require(status in ('skipped', 'fallback', 'rejected', 'selected'), 'Terminal source stage')
            require('original_headroom' in details, 'Supported experimental policy')
            head = details['original_headroom']
            for key in ('token_headroom', 'call_headroom'):
                same(head[key], c['admission_budget'][key], 'Original available '+key)
            job = source.prepare(req, plan, baseline['coarse'], old_fine, old_main,
                                 baseline['carried_ids'], head, cascade)
            same(details['slot'], job['slot'], 'Source-preserving slot')
            same(details['source_view_ids'], list(job['views']), 'Complete source view inventory')
            same(details['nominee_ids'], [n['id'] for n in job['nominees']], 'Complete projection inventory')
            if job['reason']:
                same(status, 'skipped', 'Ineligible source proposal skipped')
                same(details['reason'], job['reason'], 'Exact preflight skip')
                require(not details['batches'] and not details['responses'], 'Skipped stage has no source calls')
            else:
                same(details.get('main_reservation', job['main_reservation']), job['main_reservation'], 'Protected main reservation')
                responses = details['responses']
                raw_ce = [r for r in responses if r['kind'] == 'rerank']
                raw_gate = [r for r in responses if r['kind'] == 'llm']
                same([r['kind'] for r in responses], ['rerank']*len(raw_ce)+['llm']*len(raw_gate), 'Serial source then gate responses')
                require(len(raw_ce) <= len(details['batches']) <= len(job['batches']), 'No extra source batches')
                require(len(raw_gate) <= 1, 'At most one winner review')
                scores = {i: {} for i in range(len(job['queries']))}
                spent = 0; initial_tokens = initial_calls = None
                for i, batch in enumerate(details['batches']):
                    expected = job['batches'][i]
                    for key in ('option', 'view_ids', 'bound'):
                        same(batch[key], expected[key], 'Exact source batch '+key)
                    if initial_tokens is None:
                        initial_tokens, initial_calls = batch['tokens_before'], batch['calls_before']
                    same(batch['tokens_before']-initial_tokens, spent, 'Serial source token ledger')
                    same(batch['calls_before']-initial_calls, i, 'Serial source call ledger')
                    require(batch['bound']+job['main_reservation']+spent <= head['token_headroom'], 'Incremental main token protection')
                    require(i+3 <= head['call_headroom'], 'Main and gate call protection')
                    totals['started_source_batches'] += 1
                    if i < len(raw_ce):
                        raw = raw_ce[i]
                        query = job['queries'][batch['option']]
                        docs = [job['views'][key] for key in batch['view_ids']]
                        same(raw['request_sha256'], source.digest(wire_body(config.CE_MODEL, config.CE_API_FORMAT, query, docs)), 'Actual source wire body hash')
                        usage = ce._usage(raw['response'], batch['bound'])['usage']['total_tokens']
                        totals['archived_source_responses'] += 1; totals['accounted_source_tokens'] += usage
                    if batch['status'] == 'ok':
                        require(i < len(raw_ce), 'Successful source response archived')
                        values = ce._scores(raw_ce[i]['response'], len(batch['view_ids']))
                        same(values, batch['scores'], 'Raw response score mapping')
                        same(batch['tokens_after']-batch['tokens_before'], usage, 'Provider usage charged')
                        same(batch['calls_after'], batch['calls_before']+1, 'One wire call per source batch')
                        scores[batch['option']].update(zip(batch['view_ids'], values, strict=True))
                        spent += usage
                    else:
                        require(status == 'fallback' and i == len(details['batches'])-1, 'Only terminal failed batch may be incomplete')
                complete = len(details['batches']) == len(job['batches']) and all(b['status'] == 'ok' for b in details['batches'])
                if not complete:
                    require(status == 'fallback' and not raw_gate, 'Partial source matrix keeps baseline')
                else:
                    winner = policy.rank(job['nominees'], scores, len(job['queries']))[0]
                    if winner['option_score_range'] <= 0:
                        require(status == 'skipped' and details['reason'] == 'zero_score_range' and not raw_gate, 'Zero range baseline')
                    else:
                        same(details['winner'], {k: v for k, v in winner.items() if k != 'unit'}, 'Deterministic sole winner')
                        gate = review.build(req.query, req.options, [dict(id=winner['source_view_id'],
                            text=winner['unit']['_packet_item']['sources'][0]['content'])])
                        if 'gate_request_sha256' in details:
                            same(details['gate_request_sha256'], source.digest(gate), 'Full semantic gate request')
                            totals['prepared_gates'] += 1
                        else:
                            require(status == 'fallback' and not raw_gate and len(gate['prompt'].encode()) > review.MAX_PROMPT_BYTES,
                                    'Oversized gate falls back before dispatch')
                        if raw_gate:
                            same(raw_gate[0]['request_sha256'], source.digest(gate_wire(config, gate)), 'Actual semantic gate wire hash')
                            totals['archived_gate_responses'] += 1
                            totals['accounted_gate_tokens'] += int((raw_gate[0]['response'].get('usage') or {}).get('total_tokens', 0))
                        if 'judgments' in details:
                            require(len(raw_gate) == 1, 'Validated gate response archived')
                            payload = gate_payload(raw_gate[0]['response'], llm)
                            same(payload, details['gate_payload'], 'Gate parsed payload')
                            judgments = review.validate(payload, gate)
                            same(judgments, details['judgments'], 'Gate validation decisions')
                            if status == 'rejected': require(not judgments[0]['accepted'], 'Negative gate retained baseline')
                        if status == 'selected':
                            require(len(raw_gate) == 1 and details['judgments'][0]['accepted'], 'Positive sole gate before admission')
                            require(spent+gate['reservation']+winner['reservation'] <= head['token_headroom'], 'Gate conservative reservation with main protected')
                            require(details['remaining_budget']['token_headroom'] >= winner['reservation'] and
                                    details['remaining_budget']['call_headroom'] >= 1, 'Actual main headroom retained')
                            unit = winner['unit']
                            fine = [unit if v['id'] == job['slot'] else v for v in old_fine]
                            main = old_main+[unit]; selected = True
                            actual = next(v for v in trace['ranked'] if v['id'] == unit['id'])
                            same({k: actual[k] for k in unit}, unit, 'Exact synthetic source unit in final candidate universe')
                            require(unit['id'] not in c['coarse']['selected_ids'], 'Projection is not original coarse retrieval')
                            same(details['original_stages']['fine'], baseline['trace']['fine'], 'Original fine snapshot')
                            same(details['proposed_fine_ids'], [v['id'] for v in fine], 'Recorded new fine')
                            same(details['proposed_main_ids'], [v['id'] for v in main], 'Recorded new main')
            same(details['display_ids'], selected, 'Display changes only on admitted projection')
        same([v['id'] for v in fine], c['fine']['selected_ids'], 'Actual fine membership/order')
        if not selected: same(c['fine'], baseline['trace']['fine'], 'Unchanged baseline fine metadata')
        same([v['id'] for v in main], c['listwise']['candidate_ids'], 'Actual main prefix plus at most one source')
        same(sorted(baseline['baseline_ids']), c['listwise']['baseline_candidate_ids'], 'Original mandatory main identity')
        same(sorted(baseline['carried_ids']), c['listwise']['carried_ids'], 'Original carried reservations')
        request = cascade.listwise_request(req, plan, main, display_ids=selected)
        same(len(request['prompt'].encode()), c['listwise']['input_bytes'], 'Actual main prompt bytes')
        require(len(fine) <= 24 and len(main) <= 24 and set(v['id'] for v in main) <= set(v['id'] for v in fine), 'Fine/main capacity')
        require(c['listwise']['input_bytes'] <= 24000, 'Original main byte bound')
        totals['admitted'] += int(selected)
        return dict(search_id=trace['search_id'], status=status, admitted=selected,
                    original_main_bytes=original_bytes, actual_main_bytes=c['listwise']['input_bytes'], totals=dict(totals))


def main(args):
    run = args.run.resolve(); plan = read(run.parent/'plan.json')
    require(plan['toggle'] == 'OPTION_MARGIN_SOURCE_ADMISSION', 'Option-margin experiment')
    for name, sha in plan['source_files'].items(): same(file_digest(run.parent/'source'/name), sha, 'Frozen HTTPS source '+name)
    sys.path.insert(0, str(run.parent/'source/memory_system'))
    from app import cascade_rerank, config, fine_evidence, schemas, option_margin_source, option_margin_review, option_margin_policy, cross_encoder, llm
    modules = cascade_rerank, config, fine_evidence, schemas, option_margin_source, option_margin_review, option_margin_policy, cross_encoder, llm
    results = {r['search_id']: dict(qa_id=r['qa_id'], packet_hash=r['packet_hash'])
               for r in json_records(run/'completed/data/results/personamem-v2-32k.jsonl') if 'search_id' in r}
    seen = set(); rows = []; totals = Counter()
    with patch('socket.socket.connect', side_effect=AssertionError('Offline source audit')):
        for trace in json_records(run/'server-search-debug.jsonl'):
            sid = trace['search_id']; require(sid in results and sid not in seen, 'Unique terminal request'); seen.add(sid)
            same(trace['packet_hash'], results[sid]['packet_hash'], 'Result packet binding')
            row = audit_trace(trace, modules); row['qa_id'] = results[sid]['qa_id']
            rows.append(row); totals.update(row['totals'])
    same(seen, set(results), 'All completed Search responses audited')
    ledger = []
    for raw in physical_records(run/'server.log'):
        text = raw.decode(errors='replace')
        if 'provider_call ' not in text or not any('stage='+s+' ' in text for s in ('search.option_margin_source', 'search.option_margin_gate')): continue
        fields = dict(re.findall(r'(\w+)=([^\s]+)', text)); ledger.append(fields)
        expected = 'qwen3-rerank' if fields['stage'] == 'search.option_margin_source' else 'gpt-4o-mini'
        require(fields['model'] == expected and fields['attempt'] == '1' and fields['retries'] == '0', 'Fixed provider and no selective retries')
    require(len(ledger) <= totals['started_source_batches']+totals['prepared_gates'], 'No untraced extra provider calls')
    require(len(ledger) >= totals['archived_source_responses']+totals['archived_gate_responses'], 'Archived responses have provider ledger entries')
    same(sum(int(r['total_tokens']) for r in ledger), totals['accounted_source_tokens']+totals['accounted_gate_tokens'], 'Source/gate raw response accounting matches ledger')
    report = dict(status='passed', summary=dict(totals, statuses=dict(Counter(r['status'] for r in rows)),
        provider_calls=len(ledger), provider_errors=sum(r['status'] != 'ok' for r in ledger)), rows=rows,
        provider_calls_by_audit=0, limitation=__doc__, input_sha256={str(p): file_digest(p) for p in (
            run.parent/'plan.json', run/'server-search-debug.jsonl', run/'server.log', run/'completed/data/results/personamem-v2-32k.jsonl')})
    with (run/'option-margin-source-audit.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args())
