"""Replay deleted evidence review on sealed candidates; no ingestion or retrieval."""
import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import budget, cascade_rerank, config, eval_scoring, listwise_recovery, llm, run_metadata, search_pipeline
from app.schemas import SearchRequest


def read_archive(path):
    with zipfile.ZipFile(path) as archive:
        def rows(name):
            return [json.loads(line) for line in archive.read('memory_system/' + name).splitlines() if line.strip()]
        searches = {s['search_id']: s for s in rows('logs/search-debug.jsonl')}
        results = rows('data/results/personamem-v2-32k.jsonl')
        prepared = {p['conversation_id']: p for p in rows('data/prepared/personamem-v2-32k.jsonl')}
    return {r['qa_id']: dict(trace=searches[r['search_id']], result=r,
        qa=next(q for q in prepared[r['conversation_id']]['qa'] if q['id'] == r['qa_id'])) for r in results}


def prepare(fixture):
    trace = deepcopy(fixture['trace'])
    candidates = {c['id']: c for c in trace['ranked']}
    original = trace['cascade']['listwise']
    submitted = [candidates[mid] for mid in original['candidate_ids']]
    selected = [candidates[mid] for mid in original['selected_ids']]
    irrelevant = {i for i, c in enumerate(submitted) if c['id'] not in original['selected_ids']}
    rules = [c for c in trace['ranked'] if cascade_rerank.protected(c)]
    req = SearchRequest(**{k: trace[k] for k in
        ('query', 'options', 'user_id', 'top_k', 'reference_time', 'evidence_token_budget')})
    plan = dict(trace['plan'], _cascade=trace['cascade'], _evidence_groups=trace['evidence_groups'])
    packet, packet_hash, _ = search_pipeline._pack_evidence(None, req, plan, rules + selected, trace['anchor_time'])
    # Establish byte-identical baseline packing before making any provider call.
    if packet_hash != trace['packet_hash']:
        raise ValueError(f"Baseline packet mismatch: {fixture['qa']['id']}")
    return trace, req, plan, submitted, selected, irrelevant, rules


async def main(args):
    import litellm
    fixtures = read_archive(args.archive)
    qids = sorted(fixtures, key=int) if args.questions == 'all' else args.questions.split(',')
    if args.output.resolve() == args.archive.resolve():
        raise ValueError('Output must not overwrite input')
    versions = run_metadata.versions()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    original = llm.complete_json
    transport = litellm.acompletion
    archive_hash = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    # Validate every requested baseline before paying for any replay.
    for qid in qids:
        prepare(fixtures[qid])
    with args.output.open('x', encoding='utf-8') as output:
        for repetition in range(1, args.repeat + 1):
            for qid in qids:
                fixture = fixtures[qid]
                trace, req, plan, submitted, selected, irrelevant, rules = prepare(fixture)
                requests = []
                async def recorded(prompt, *positional, **kwargs):
                    record = dict(prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), kwargs=kwargs)
                    requests.append(record)
                    started = time.monotonic()
                    async def recorded_transport(**params):
                        response = await transport(**params)
                        # Do not persist provider credentials, headers or hidden metadata.
                        clean = response.model_dump()
                        record.setdefault('provider_responses', []).append(
                            {key: clean[key] for key in ('choices', 'usage') if key in clean})
                        return response
                    try:
                        with patch.object(litellm, 'acompletion', recorded_transport):
                            result = await original(prompt, *positional, **kwargs)
                        record['response'] = result
                        return result
                    except BaseException as exc:
                        record['error_type'] = type(exc).__name__
                        raise
                    finally:
                        record['seconds'] = time.monotonic() - started
                started = time.monotonic()
                with budget.scope(seconds=config.SEARCH_DEADLINE_SECONDS, calls=config.SEARCH_MAX_CALLS,
                                  tokens=config.SEARCH_MAX_TOKENS) as usage:
                    with patch.object(llm, 'complete_json', recorded):
                        recovered, recovery = await listwise_recovery.recover(req, submitted, selected, irrelevant)
                recovery_seconds = time.monotonic() - started
                for candidate in recovered:
                    if candidate.get('_final') is None:
                        candidate['_score_kind'] = 'listwise'
                packet, packet_hash, manifest = search_pipeline._pack_evidence(
                    None, req, plan, rules + recovered, trace['anchor_time'])
                baseline_ids = {item['id'] for item in trace['returned']}
                if not baseline_ids <= {item['id'] for item in packet}:
                    raise ValueError('Recovery displaced baseline evidence')
                row = dict(qid=qid, repetition=repetition, archive_sha256=archive_hash, versions=versions,
                    fake=config.FAKE, baseline_packet_hash=trace['packet_hash'], baseline_hash_verified=True,
                    baseline_selected_ids=[c['id'] for c in selected], selected_ids=[c['id'] for c in recovered],
                    recovery=recovery, recovery_seconds=recovery_seconds, provider_calls=usage.calls,
                    tokens=usage.tokens, packet_hash=packet_hash, packet=packet, manifest=manifest)
                if args.answer:
                    with budget.scope(seconds=240, calls=8, tokens=128000) as answer_usage:
                        with patch.object(llm, 'complete_json', recorded):
                            prediction, score, diagnostics = await eval_scoring.evaluate(fixture['qa'], packet)
                    row.update(prediction=prediction, score=score, diagnostics=diagnostics,
                        baseline_prediction=fixture['result']['prediction'], baseline_score=fixture['result']['score'],
                        answer_provider_calls=answer_usage.calls, answer_tokens=answer_usage.tokens)
                row['requests'] = requests
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
                summary = {k: row[k] for k in ('qid', 'repetition', 'provider_calls', 'tokens', 'recovery_seconds')}
                summary.update(status=recovery['status'], restored=len(recovery['restored_ids']),
                               unreviewed=len(recovery['unreviewed_ids']), evidence=manifest['evidence_count'])
                if args.answer:
                    summary.update(prediction=prediction, score=score, error=diagnostics.get('error_type'))
                print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'runs/personamem-20260920-log-audit/inputs.zip')
    parser.add_argument('--questions', default='13,15,16,22')
    parser.add_argument('--repeat', type=int, choices=range(1, 4), default=1)
    parser.add_argument('--answer', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
