"""Diagnose cited-scope gating with verified archived prefixes and live tails.

Each archive arm is a separate fixed-input stratum, not a causal source-removal
pair. This selected-case probe cannot estimate cohort or HTTPS accuracy.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import logging
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, budget, config, eval_scoring, evidence_packet, llm, run_metadata
from replay_query_context import records, recorded_response


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def execute(case, enabled, offline=False):
    row, trace, qa = case
    calls = row['answer_calls']
    boundary = next((i for i, c in enumerate(calls) if c['stage'] == 'eval.choice_premise_scope'), None)
    if boundary is None and not offline:
        raise ValueError('Live prefix probe requires an archived scope boundary')
    pending = copy.deepcopy(calls if offline else calls[:boundary])
    live, replayed, violations, scope_inputs = [], [], [], []
    provider, scope = llm.complete_json, answer_choice.reclassify_uncited

    async def capture(entries, diagnostics):
        scope_inputs.append(copy.deepcopy(entries))
        return await scope(entries, diagnostics)

    async def replay(prompt, *args, **kwargs):
        identity = dict(stage=kwargs['stage'], prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        if pending:
            old = pending.pop(0)
            if any(old[k] != v for k, v in identity.items()):
                violations.append(identity)
                raise ValueError('Archived stage/prompt mismatch')
            budget.current.get().before_call()
            replayed.append(identity)
            return recorded_response(old)
        if violations or offline:
            violations.append(identity)
            raise ValueError('Unverified continuation forbidden')
        live.append(dict(**identity, prompt=prompt))
        return await provider(prompt, *args, **kwargs)

    settings = dict(trace['versions']['settings'], CHOICE_CITED_SCOPE=enabled)
    with (patch.multiple(config, **settings), patch.object(llm, 'complete_json', replay),
          patch.object(answer_choice, 'reclassify_uncited', capture)):
        prediction, score, diagnostics = await eval_scoring.evaluate(qa, copy.deepcopy(trace['returned']))
        actual = {key: getattr(config, key) for key in settings}
    # Pydantic error locations are tuples in memory and lists in JSON archives.
    diagnostics = json.loads(json.dumps(diagnostics, ensure_ascii=False))
    if pending or violations or len(scope_inputs) != 1:
        raise ValueError('Prefix/scope reconstruction failed')
    if diagnostics.get('answer_source_catalog') != row['answer_source_catalog']:
        raise ValueError('Catalog differs from archive')
    if offline:
        for key in ('choice_alignment', 'answer_premise_scope', 'answer_eligible_options', 'answer_calls'):
            if diagnostics.get(key) != row.get(key):
                if key == 'answer_calls':
                    differences = [dict(index=i, fields={k: [a.get(k), b.get(k)]
                        for k in a.keys() | b.keys() if a.get(k) != b.get(k)})
                        for i, (a, b) in enumerate(zip(diagnostics[key], row[key])) if a != b]
                    raise ValueError('Offline calls differ: ' + json.dumps(differences, ensure_ascii=False))
                raise ValueError('Offline replay differs: ' + key)
        if (prediction, score) != (row['prediction'], row['score']):
            raise ValueError('Offline answer differs')
    return dict(prediction=prediction, score=score, diagnostics=diagnostics,
                actual_settings=actual, fixed_calls=replayed, live_requests=live,
                scope_inputs=scope_inputs[0], packet_hash=row['packet_hash'])


async def main(args):
    if config.FAKE or config.LLM_MODEL != 'gpt-4o-mini' or config.CE_MODEL != 'qwen3-rerank':
        raise ValueError('Require real gpt-4o-mini and unchanged qwen3-rerank')
    root = args.run.resolve()
    if (root / 'exit-code').read_text().strip() != '0':
        raise ValueError('Archive must have completed successfully')
    plan = json.loads((root / 'plan.json').read_text())
    if digest(args.prepared) != plan['data_sha256']:
        raise ValueError('Prepared data hash differs')
    questions = {(d['conversation_id'], q['id']): q for d in records(args.prepared) for q in d['qa']}
    cases = []
    archive_arms = ('control', 'treatment') if args.archive_arms == 'both' else (args.archive_arms,)
    for arm in archive_arms:
        run = root / arm
        traces = {t['search_id']: t for t in records(run / 'server-search-debug.jsonl')}
        reader_versions = run / 'reader-versions.json'
        if reader_versions.exists():
            actual_reader = json.loads(reader_versions.read_text())
            if actual_reader['model'] != 'gpt-4o-mini':
                raise ValueError('Archive reader model differs')
            for trace in traces.values():
                trace['versions']['settings'] = actual_reader['settings']
        results = {r['qa_id']: r for r in records(run / 'completed/data/results/personamem-v2-32k.jsonl')}
        selected = sorted(results, key=int) if args.questions == 'all' else args.questions.split(',')
        for qid in selected:
            row = results[qid]
            trace = traces[row['search_id']]
            if evidence_packet.digest(trace['returned']) != row['packet_hash']:
                raise ValueError('Archive packet hash differs')
            if trace['versions']['model'] != 'gpt-4o-mini':
                raise ValueError('Archive model differs')
            cases.append((arm, (row, trace, questions[row['conversation_id'], qid])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_handler = logging.FileHandler(args.output.with_suffix('.provider.log'), mode='x')
    logging.getLogger('aml.metrics').addHandler(log_handler)
    logging.getLogger('aml.metrics').setLevel(logging.INFO)
    with args.output.open('x') as out:
        def emit(value):
            out.write(json.dumps(value, ensure_ascii=False) + '\n')
            out.flush()
        emit(dict(event='plan', run=str(root), questions=args.questions, repeats=args.repeat, archive_arms=archive_arms,
                  versions=run_metadata.versions(), source_sha256=digest(Path(__file__)),
                  prepared_sha256=digest(args.prepared), protocol=__doc__,
                  limitation='Frozen calls count against call budget; their original tokens and latency are not simulated. Live tail usage logged separately.'))
        # Preflight all cases without any network dispatch before the live phase.
        for arm, case in cases:
            with patch('socket.socket.connect', side_effect=AssertionError('Offline preflight')):
                replay = await execute(case, case[1]['versions']['settings']['CHOICE_CITED_SCOPE'], offline=True)
            emit(dict(event='preflight', archive_arm=arm, qa_id=case[0]['qa_id'],
                      prediction=replay['prediction'], scope_inputs=replay['scope_inputs'],
                      packet_hash=replay['packet_hash'], verified_calls=len(replay['fixed_calls'])))
            print(f"preflight {arm} Q{case[0]['qa_id']} passed", flush=True)
        if args.offline_only:
            return
        for repetition in range(1, args.repeat + 1):
            for index, (arm, case) in enumerate(cases):
                for enabled in ([False, True] if (index + repetition) % 2 else [True, False]):
                    result = await execute(case, enabled)
                    emit(dict(event='result', archive_arm=arm, qa_id=case[0]['qa_id'],
                              repetition=repetition, enabled=enabled, **result))
                    print(json.dumps(dict(archive_arm=arm, repetition=repetition,
                        qa_id=case[0]['qa_id'], enabled=enabled, prediction=result['prediction'],
                        score=result['score'], live_calls=len(result['live_requests']))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared/personamem-v2-32k.jsonl'))
    parser.add_argument('--questions', required=True)
    parser.add_argument('--archive-arms', choices=('both', 'control', 'treatment'), default='both')
    parser.add_argument('--repeat', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--offline-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
