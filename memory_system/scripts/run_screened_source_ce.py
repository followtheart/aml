"""Execute preregistered complete source batches serially, then freeze nominations.

No reference data is read by this executor. Complete source scoring is only a
nomination signal; frozen-main recall and live HTTPS still require later audits.
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

from batched_records import file_digest, json_records
from source_nomination import nominate, wire_body


def read(path):
    data = bytearray()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            data.extend(block)
    return json.loads(data)


def body_digest(body):
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


class Ledger(logging.Handler):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.context = None

    def emit(self, record):
        message = record.getMessage()
        if message.startswith('provider_call '):
            row = dict(self.context or {}, fields=dict(re.findall(r'(\w+)=([^\s]+)', message)))
            self.stream.write(json.dumps(row)+'\n')
            self.stream.flush()


async def execute(args):
    root = args.experiment.resolve()
    plan = read(root/'plan.json')
    for name, digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        if file_digest(Path(name)) != digest:
            raise ValueError('Preregistered input changed: '+name)
    cohort = plan['cohorts'][args.cohort]
    sys.path.insert(0, cohort['runtime_source'])
    from app import budget, config, cross_encoder
    import httpx
    for key, value in plan['provider_settings'].items():
        setattr(config, key, value)
    if config.FAKE or config.LLM_MODEL != 'gpt-4o-mini' or config.CE_MODEL != 'qwen3-rerank':
        raise ValueError('Fixed real models required')
    if hashlib.sha256(config.CE_API_URL.encode()).hexdigest() != plan['provider_url_sha256']:
        raise ValueError('Provider endpoint changed')
    if not config.CE_API_KEY:
        raise ValueError('Provider credential missing')
    for name, digest in cohort['runtime_sha256'].items():
        if file_digest(Path(cohort['runtime_source'])/name) != digest:
            raise ValueError('Frozen provider code changed: '+name)
    outputs = {name: root/(args.cohort+'-'+name+'.jsonl') for name in ('selections','requests','ledger')}
    if any(p.exists() for p in outputs.values()):
        raise FileExistsError('Paid outputs already exist; no automatic rerun')
    expected = {(r['search_id'], r['batch']): r for r in cohort['requests']}
    seen, emitted, counters = set(), set(), dict(questions=0, nominated=0, skipped=0, fallback=0, calls=0, tokens=0)
    with outputs['selections'].open('x') as selected, outputs['requests'].open('x') as events, outputs['ledger'].open('x') as ledger_file:
        ledger = Ledger(ledger_file)
        logger = logging.getLogger('aml.metrics')
        logger.setLevel(logging.INFO)
        logger.addHandler(ledger)
        def event(row):
            events.write(json.dumps(row, ensure_ascii=False)+'\n')
            events.flush()
        try:
            for job in json_records(Path(cohort['inventory'])):
                sid = job['search_id']
                if sid in seen:
                    raise ValueError('Duplicate inventory question')
                seen.add(sid)
                scores, failure, dispatched = {}, None, 0
                limits = None
                if job['scheduling']['eligible']:
                    # Gate before each question, outside its provider deadline.
                    while True:
                        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                             if line.startswith('MemAvailable:')))/1024
                        if available >= 1536 and os.getloadavg()[0] <= 1.25:
                            break
                        print(json.dumps(dict(status='waiting_for_resources', qa_id=job['qa_id'], available_mib=available)),flush=True)
                        await asyncio.sleep(15)
                    main_bound = job['scheduling']['main_reservation']
                    with budget.scope(seconds=job['scheduling']['ce_deadline_seconds'],
                            calls=job['original_headroom']['call_headroom'] or len(job['batches'])+1,
                            tokens=job['original_headroom']['token_headroom'] or main_bound+sum(job['batch_bounds'])) as limits:
                        limits.reserve_calls(1)
                        limits.reserve_tokens(main_bound)
                        try:
                            async with asyncio.timeout(job['scheduling']['ce_deadline_seconds']):
                                for index, batch in enumerate(job['batches']):
                                    texts = [job['views'][key]['text'] for key in batch]
                                    body = wire_body(job['ce_model'], job['ce_api_format'], job['ce_query'], texts)
                                    registered = expected[sid, index]
                                    digest = body_digest(body)
                                    if digest != registered['body_sha256'] or batch != registered['view_ids']:
                                        raise ValueError('Exact registered request changed')
                                    bound = cross_encoder.request_size(job['ce_query'], texts)
                                    if bound != registered['bound']:
                                        raise ValueError('Request bound changed')
                                    context = dict(search_id=sid, qa_id=job['qa_id'], batch=index)
                                    ledger.context = context
                                    original_post = httpx.AsyncClient.post
                                    wire_calls = 0
                                    async def checked_post(client, url, **kwargs):
                                        nonlocal wire_calls
                                        if url != config.CE_API_URL or kwargs.get('json') != body or wire_calls:
                                            raise ValueError('Provider wire mismatch or repeat dispatch')
                                        wire_calls += 1
                                        event(dict(context, status='dispatch', body_sha256=digest, bound=bound))
                                        return await original_post(client, url, **kwargs)
                                    start_calls, start_tokens = limits.calls, limits.tokens
                                    try:
                                        with patch.object(httpx.AsyncClient, 'post', checked_post):
                                            values = await cross_encoder.rerank(job['ce_query'], texts,
                                                stage='diagnostic.screened_source_ce')
                                        if len(values) != len(batch) or wire_calls != 1:
                                            raise ValueError('Incomplete source scoring')
                                        if limits.tokens + main_bound > limits.max_tokens:
                                            raise ValueError('Reported usage consumed protected main budget')
                                        scores.update(zip(batch, values, strict=True))
                                        event(dict(context, status='ok', view_ids=batch, scores=values,
                                            body_sha256=digest, bound=bound, calls=limits.calls-start_calls,
                                            tokens=limits.tokens-start_tokens))
                                    except BaseException as exc:
                                        event(dict(context, status='error', error=type(exc).__name__,
                                            body_sha256=digest, bound=bound, calls=limits.calls-start_calls,
                                            tokens=limits.tokens-start_tokens, wire_calls=wire_calls))
                                        raise
                                    finally:
                                        dispatched += wire_calls
                                        emitted.add((sid,index))
                        except (Exception, asyncio.CancelledError) as exc:
                            failure = type(exc).__name__
                            scores = {}
                            # External task cancellation must not start further questions.
                            if isinstance(exc, asyncio.CancelledError):
                                raise
                        finally:
                            limits.release_calls(1)
                            limits.release_tokens(main_bound)
                decision = nominate(job, scores)
                if failure:
                    decision = dict(status='fallback', reason=failure, addition=None)
                row = dict(search_id=sid, conversation_id=job['conversation_id'], qa_id=job['qa_id'],
                    decision=decision, scores=scores, provider_calls=dispatched,
                    budget_calls=limits.calls if limits else 0, reported_tokens=limits.tokens if limits else 0,
                    original_main_ids=job['original_main_ids'],
                    proposed_main_ids=job['original_main_ids']+([decision['addition']] if decision['addition'] else []),
                    original_display_preserved=decision['addition'] is None)
                selected.write(json.dumps(row,ensure_ascii=False)+'\n');selected.flush()
                counters['questions'] += 1
                counters[decision['status']] += 1
                counters['calls'] += dispatched
                counters['tokens'] += row['reported_tokens']
                print(json.dumps(dict(qa_id=job['qa_id'],status=decision['status'],calls=dispatched)),flush=True)
        finally:
            logger.removeHandler(ledger)
    if seen != set(cohort['search_ids']):
        raise ValueError('Incomplete source nomination cohort')
    result = dict(status='complete_fixed_nominations', summary=counters,
        attempted_batches=len(emitted), registered_batches=len(expected),
        output_sha256={name:file_digest(path) for name,path in outputs.items()},
        plan_sha256=file_digest(root/'plan.json'), limitation=__doc__)
    with (root/(args.cohort+'-selections.manifest.json')).open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result['summary']),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--cohort',choices=('regression','validation'),required=True)
    asyncio.run(execute(parser.parse_args()))
