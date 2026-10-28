"""Rebuild all pairwise decisions and reader paths using exact recorded calls."""
import argparse
import asyncio
import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, llm, metrics
from probe_question_source_filter import load_cohort, sha
from replay_query_context import recorded_response
from replay_pairwise_generic_selection import main as replay


async def audit(path):
    report = json.loads(path.read_text())
    if hashlib.sha256(Path(report['prepared']).read_bytes()).hexdigest() != report['prepared_sha256']:
        raise ValueError('Prepared data changed')
    for name, digest in report['scripts'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError('Script changed')
    cohort = load_cohort(Path(report['run']), Path(report['prepared']))
    expected = [(data[0]['qa_id'], arm) for index, data in enumerate(cohort)
                for arm in (['control', 'candidate'] if index % 2 == 0 else ['candidate', 'control'])]
    if [(r['qa_id'], r['arm']) for r in report['rows']] != expected:
        raise ValueError('Population/order changed')
    pending = [copy.deepcopy(c) for r in report['rows'] for c in r['live_calls']]
    errors = []

    async def provider(prompt, *pos, **kw):
        if not pending:
            errors.append('extra')
            raise ValueError('Unexpected call')
        call = pending.pop(0)
        if (call['stage'] != kw['stage'] or call['prompt_sha256'] != sha(prompt)
                or call['prompt'] != prompt or call['schema'] != kw.get('schema')
                or call['system'] != kw.get('system')):
            errors.append('changed')
            raise ValueError('Provider input changed')
        budget.current.get().before_call()
        if call.get('provider_parse') and call['error_type'] == 'LLMError':
            raise llm.LLMError(call.get('error_detail', 'parse error')) from metrics.ResponseParseError('parse error')
        return recorded_response(call)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'audit.json'
        with patch.object(llm, 'complete_json', provider), contextlib.redirect_stdout(io.StringIO()):
            await replay(SimpleNamespace(run=Path(report['run']), prepared=Path(report['prepared']), output=out))
        rebuilt = json.loads(out.read_text())
    if pending or errors:
        raise ValueError('Provider calls incomplete')
    fields = ['qa_id', 'arm', 'prediction', 'score', 'comparisons', 'eligible']
    for old, new in zip(report['rows'], rebuilt['rows']):
        if any(old[k] != new[k] for k in fields):
            raise ValueError('Outcome changed')
        for key in ['choice_alignment', 'answer_eligible_options', 'answer_blocked_options',
                    'answer_source_catalog', 'answer_constraint_catalog', 'answer_selection_review']:
            if old['diagnostics'].get(key) != new['diagnostics'].get(key):
                raise ValueError('Reader state changed: ' + key)
    if any(report[k] != rebuilt[k] for k in ['scores', 'live_calls']):
        raise ValueError('Totals changed')
    control = {r['qa_id']: r for r in report['rows'] if r['arm'] == 'control'}
    return dict(status='passed', questions=len(cohort), scores=report['scores'],
                live_calls=report['live_calls'],
                wins=[r['qa_id'] for r in report['rows'] if r['arm'] == 'candidate' and r['score'] > control[r['qa_id']]['score']],
                losses=[r['qa_id'] for r in report['rows'] if r['arm'] == 'candidate' and r['score'] < control[r['qa_id']]['score']],
                triggered=[r['qa_id'] for r in report['rows'] if r['comparisons']],
                errors=[dict(qa_id=r['qa_id'], arm=r['arm'], stage=c['stage'], error=c.get('error_type'))
                        for r in report['rows'] for c in r['live_calls'] if c['status'] != 'ok'],
                caveat=report['caveat'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = asyncio.run(audit(args.replay))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
