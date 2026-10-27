"""Paired scope-only diagnosis over every affected verified archive input.

No source retrieval, support extraction, downstream answers, or gold selection.
Both modes see identical complete options and claims. Only cited target
eligibility changes. Removed assertions require inspection, not score credit.
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
from app import answer_choice, budget, config, run_metadata
from replay_query_context import records


def cited_targets(entries):
    return [f"{e['letter']}:{i}" for e in entries
            if e.get('validation_status') == 'valid' and not e.get('validation_errors')
            for i, c in enumerate(e['claims']) if c.get('citations')
            and not c.get('validation_errors') and not c.get('dropped_citations')]


async def main(args):
    if config.FAKE or config.LLM_MODEL != 'gpt-4o-mini' or config.CE_MODEL != 'qwen3-rerank':
        raise ValueError('Real fixed models required')
    preflight = records(args.preflight)
    inputs = [r for r in preflight if r['event'] == 'preflight']
    plan = preflight[0]
    root = Path(plan['run'])
    expected = json.loads((root / 'plan.json').read_text())['questions']
    if len(inputs) != expected * 2 or len({(r['archive_arm'], r['qa_id']) for r in inputs}) != len(inputs):
        raise ValueError('Require complete verified archive input cohort')
    traces = {}
    for arm in ('control', 'treatment'):
        by_search = {t['search_id']: t for t in records(root / arm / 'server-search-debug.jsonl')}
        for row in records(root / arm / 'completed/data/results/personamem-v2-32k.jsonl'):
            traces[arm, row['qa_id']] = by_search[row['search_id']]
    jobs = [r for r in inputs if cited_targets(r['scope_inputs'])]
    handler = logging.FileHandler(args.output.with_suffix('.provider.log'), mode='x')
    logging.getLogger('aml.metrics').addHandler(handler)
    logging.getLogger('aml.metrics').setLevel(logging.INFO)
    with args.output.open('x') as out:
        def emit(row):
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            out.flush()
        emit(dict(event='plan', run=str(root), inputs=len(inputs), affected=len(jobs),
                  preflight_sha256=hashlib.sha256(args.preflight.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  versions=run_metadata.versions(), limitation=__doc__,
                  expected_rows=len(jobs)*2, selection='all structurally eligible cited claims; no scores or gold'))
        for index, row in enumerate(jobs):
            for enabled in ([False, True] if index % 2 == 0 else [True, False]):
                settings = dict(traces[row['archive_arm'], row['qa_id']]['versions']['settings'],
                                CHOICE_CITED_SCOPE=enabled)
                diagnostics = {}
                with patch.multiple(config, **settings), budget.scope(seconds=240, calls=28, tokens=256000) as limits:
                    result = await answer_choice.reclassify_uncited(copy.deepcopy(row['scope_inputs']), diagnostics)
                output = dict(event='result', archive_arm=row['archive_arm'], qa_id=row['qa_id'],
                    enabled=enabled, affected_claims=cited_targets(row['scope_inputs']),
                    entries=result, diagnostics=diagnostics, calls=limits.calls, tokens=limits.tokens,
                    actual_settings=settings)
                emit(output)
                scope = diagnostics.get('answer_premise_scope', {})
                print(json.dumps(dict(archive_arm=row['archive_arm'], qa_id=row['qa_id'], enabled=enabled,
                    calls=limits.calls, status=scope.get('status'), removed=scope.get('removed', []))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
