"""Verify and archive the completed target sentence layout diagnostic, offline."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
from unittest.mock import patch

from probe_target_sentence_reviews import load_jobs, records, sha
from app import config


def main(root):
    root = root.resolve()
    project = Path(__file__).resolve().parents[1]
    run = root / 'target-sentence-review'
    read = lambda path: json.loads(path.read_text())
    plan = read(run / 'plan.json')
    preflight = read(root / 'target-sentence-review-preflight/plan.json')
    audit = read(run / 'audit.json')
    summary = audit['summary']
    assert (run / 'exit-code').read_text().strip() == '0'
    for key in ('jobs', 'schedule', 'source_run', 'source_trace_sha256', 'script_sha256', 'loader_sha256'):
        assert plan[key] == preflight[key], key
    assert audit['plan_sha256'] == sha(run / 'plan.json')
    assert audit['responses_sha256'] == sha(run / 'live.jsonl')
    assert audit['auditor_sha256'] == sha(project / 'scripts/audit_target_sentence_reviews.py')
    jobs = load_jobs(Path(plan['source_run']))
    assert [meta for meta, *_ in jobs] == plan['jobs']
    for meta, before, after, _ in jobs:
        original = json.loads(before['prompt'].rsplit('\n', 1)[1])
        transformed = json.loads(after['prompt'].rsplit('\n', 1)[1])
        assert list(original['targets']) == list(transformed['targets'])
        for key, text in original['targets'].items():
            parts = transformed['targets'][key]
            assert text == (''.join(parts) if isinstance(parts, list) else parts)
        for key in ('sources', 'candidates'):
            assert original[key] == transformed[key]
        for key in ('schema', 'system', 'max_tokens'):
            assert before[key] == after[key]
        if not meta['applied']:
            assert before == after
    assert len(jobs) == 158 and sum(meta['applied'] for meta, *_ in jobs) == 120
    assert summary['decoded_responses'] == summary['provider_calls'] == 316
    assert summary['historical_selections_verified'] == 56
    assert summary['recorded_tokens'] == 1043436
    assert summary['provider_statuses'] == {'ok': 316}
    changes = summary['reference_transitions']
    gains = [c for c in changes if c['control'] == 'negative' and c['treatment'] == 'invalid']
    assert {c['qa_id'] for c in gains} == {'25', '31', '37'}
    assert not any(c['treatment'] == 'positive' for c in changes)
    assert summary['transport_changes']['main']['lost'] == ['4']
    assert summary['transport_changes']['overflow']['lost'] == ['4']
    changed = summary['by_layout']['changed']
    assert changed['control']['judgments']['positive'] == changed['treatment']['judgments']['positive'] == 47
    q47 = [r for r in records(run / 'live.jsonl') if r['qa_id'] == '47']
    assert len(q47) == 6
    assert all(not (j['valid'] and j['useful']) for r in q47 for j in r['judgments'])
    required = {'LLM_MODEL': 'gpt-4o-mini', 'CE_MODEL': 'qwen3-rerank',
                'EMBED_MODEL': 'openai/text-embedding-v4', 'EMBED_DIM': 256}
    assert all(getattr(config, key) == value for key, value in required.items())
    cfg = project / 'app/config.py'
    module = ast.parse(cfg.read_text())
    module.body = [n for n in module.body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name) and n.value.func.id == '_load_dotenv')]
    defaults = {'__file__': str(cfg)}
    with patch.dict(os.environ, {}, clear=True):
        exec(compile(module, str(cfg), 'exec'), defaults)
    historical = records(project / 'runs/https-fine-context-fill-20261023/control/server-search-debug.jsonl')[0]['versions']['settings']
    bools = {k: v for k, v in historical.items() if type(v) is bool}
    assert len(bools) == 65 and all(defaults[k] == v for k, v in bools.items())
    app_files = {k: v for k, v in read(root / 'plan.json')['source_files'].items()
                 if k.startswith('memory_system/app/')}
    assert len(app_files) == 81
    assert all(sha(project / name.removeprefix('memory_system/')) == expected
               for name, expected in app_files.items())
    archive = root / 'target-sentence-study-scripts'
    archive.mkdir(exist_ok=False)
    for name in ('probe_target_sentence_reviews.py', 'audit_target_sentence_reviews.py',
                 'probe_candidate_sources.py', 'audit_candidate_sources.py',
                 'audit_https_funnel.py', 'audit_https_stages.py',
                 'evaluation_model_guard.py', 'finalize_target_sentence_study.py'):
        shutil.copy2(project / 'scripts' / name, archive / name)
    paths = list(archive.iterdir()) + list(run.iterdir())
    paths += list((root / 'target-sentence-review-preflight').iterdir())
    paths += [root / 'target-sentence-review.log', root / 'target-sentence-review-preflight.log']
    artifacts = {str(p.relative_to(root)): sha(p) for p in paths if p.is_file()}
    result = dict(source_run=plan['source_run'], summary=summary, artifacts=artifacts,
        exact_source_and_target_layout_pairs_verified=len(jobs),
        unchanged_app_files=len(app_files), historical_boolean_defaults_verified=len(bools),
        required_models=required, runtime_promoted=False, new_https_run=False,
        finalizer_provider_calls=0,
        decision='Reject: reference gains are invalid-judgment retention, Q4 loses reference, changed-layout valid positives do not increase.',
        limitations=[plan['limitation'],
            'Diagnostic uses the earlier fine-semantic-reserve treatment review cohort, not latest Search inputs.',
            'Sentence layout includes an explanatory note; it is not a layout-only causal isolation.',
            'Validation success does not itself establish semantic relevance.'])
    with (root / 'target-sentence-study.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('exact_source_and_target_layout_pairs_verified',
        'unchanged_app_files', 'historical_boolean_defaults_verified', 'decision')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run)
