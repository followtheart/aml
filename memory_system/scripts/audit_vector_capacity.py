"""Offline full provider-input and outcome replay for vector capacity experiment."""
import argparse
import asyncio
import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

import numpy as np
import replay_vector_capacity as replay
from app import budget, cross_encoder, embeddings, llm, metrics


def outcome(row):
    search = row['search']
    cascade = search['cascade']
    return dict(qa_id=row['qa_id'], arm=row['arm'], prediction=row['prediction'], score=row['score'],
                packet_hash=search['packet_hash'], status=search['status'],
                degraded=search['coverage_manifest']['search_degraded'],
                routes=[(r['channel'], [c['id'] for c in r['candidates']]) for r in search['routes']],
                fused=[r['id'] for r in search['fused']], coarse=cascade['coarse']['selected_ids'],
                fine=cascade['fine']['selected_ids'], admitted=cascade['listwise']['candidate_ids'],
                selected=cascade['listwise']['selected_ids'],
                eligible=row['diagnostics'].get('answer_eligible_options'))


async def audit(path):
    plan = json.loads(path.with_suffix('.plan.json').read_text())
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = [(qid, arm) for i, qid in enumerate(plan['qa_ids']) for arm in
                (['control', 'candidate'] if i % 2 == 0 else ['candidate', 'control'])]
    assert [(r['qa_id'], r['arm']) for r in rows] == expected
    assert replay.digest(Path(plan['prepared']).read_text()) == plan['prepared_sha256']
    assert replay.digest(Path(plan['probe']).read_text()) == plan['probe_sha256']
    assert replay.digest(Path(replay.__file__).read_text()) == plan['script_sha256']
    pending = []
    for row in rows:
        for original_event in row['provider_events']:
            event = copy.deepcopy(original_event)
            if event['kind'] == 'llm' and 'response' not in event:
                matches = [c for c in row['diagnostics'].get('answer_calls', [])
                           if c.get('stage') == event.get('stage')
                           and c.get('prompt_sha256') == event.get('prompt_sha256')
                           and c.get('error_type') == event.get('error_type')]
                phases = {c.get('failure_phase') for c in matches}
                if phases == {'provider_parse'}:
                    event['failure_phase'] = 'provider_parse'
            pending.append(event)
    count = len(pending)
    violations = []

    async def frozen(kind, *pos, **kw):
        if kind == 'llm':
            identity = dict(prompt_sha256=replay.digest(pos[0]), stage=kw.get('stage'),
                            schema_sha256=replay.digest(json.dumps(kw.get('schema'), sort_keys=True)))
        elif kind == 'embedding':
            identity = dict(texts=list(pos[0]), stage=kw.get('stage'))
        else:
            identity = dict(query=pos[0], document_hashes=[replay.digest(t) for t in pos[1]], stage=kw.get('stage'))
        if not pending:
            violations.append('extra_call')
            raise ValueError('Unexpected provider call')
        event = pending.pop(0)
        if event['kind'] != kind or any(event.get(k) != v for k, v in identity.items()):
            violations.append(identity.get('stage'))
            raise ValueError('Provider input changed')
        limits = budget.current.get()
        if limits:
            limits.before_call()
        if 'response' not in event:
            if event.get('failure_phase') == 'provider_parse' and event['error_type'] == 'LLMError':
                raise llm.LLMError(event['error_detail']) from metrics.ResponseParseError(event['error_detail'])
            classes = {'LLMError': llm.LLMError, 'ResponseParseError': metrics.ResponseParseError,
                       'RerankUnavailable': cross_encoder.RerankUnavailable, 'BudgetExceeded': budget.BudgetExceeded,
                       'ValueError': ValueError, 'TimeoutError': TimeoutError}
            if event['error_type'] not in classes:
                violations.append('unreproducible_error')
                raise ValueError('Unsupported recorded error')
            raise classes[event['error_type']](event['error_detail'])
        return np.asarray(event['response'], dtype=np.float32) if kind == 'embedding' else copy.deepcopy(event['response'])

    async def frozen_llm(*pos, **kw): return await frozen('llm', *pos, **kw)
    async def frozen_embed(*pos, **kw): return await frozen('embedding', *pos, **kw)
    async def frozen_ce(*pos, **kw): return await frozen('ce', *pos, **kw)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / 'replay.jsonl'
        with patch.object(llm, 'complete_json', frozen_llm), patch.object(embeddings, 'embed', frozen_embed), \
             patch.object(cross_encoder, 'rerank', frozen_ce), contextlib.redirect_stdout(io.StringIO()):
            await replay.main(SimpleNamespace(run=Path(plan['run']), prepared=Path(plan['prepared']),
                                             probe=Path(plan['probe']), output=output, capacity=plan['capacities'][1]))
        rebuilt = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    assert not pending and not violations, 'Provider trace mismatch'
    assert [outcome(r) for r in rebuilt] == [outcome(r) for r in rows], 'Search/reader outcome mismatch'
    controls = {r['qa_id']: r for r in rows if r['arm'] == 'control'}
    candidates = {r['qa_id']: r for r in rows if r['arm'] == 'candidate'}
    return dict(status='passed', questions=len(controls), provider_events=count,
                scores={arm: sum(r['score'] for r in rows if r['arm'] == arm) for arm in ['control', 'candidate']},
                wins=[q for q in controls if candidates[q]['score'] > controls[q]['score']],
                losses=[q for q in controls if candidates[q]['score'] < controls[q]['score']],
                provider_errors=[dict(qa_id=r['qa_id'], arm=r['arm'], kind=e['kind'], error=e['error_type'])
                                 for r in rows for e in r['provider_events'] if 'response' not in e],
                caveat=plan['caveat'] + ' Audit checks every provider input and rebuilt route/fusion/admission/packet/eligibility/answer. Replay omits original network latency and token consumption; it proves resulting trace agreement, not identical cost/deadline execution.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(audit(args.replay))
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'caveat'}))
