"""Archive an audited paired review diagnostic and verify runtime invariants."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(args):
    project = Path(__file__).resolve().parents[1]
    root = args.run.resolve()
    run = root / args.diagnostic
    read = lambda path: json.loads(path.read_text())
    plan = read(run / 'plan.json')
    audit = read(run / 'audit.json')
    summary = audit['summary']
    if (run / 'exit-code').read_text().strip() != '0':
        raise ValueError('Require completed successful diagnostic')
    for key, path in [('plan_sha256', run / 'plan.json'), ('responses_sha256', run / 'live.jsonl'),
                      ('auditor_sha256', project / 'scripts' / args.auditor)]:
        if audit[key] != sha(path):
            raise ValueError('Audit provenance mismatch: ' + key)
    if plan['script_sha256'] != sha(project / 'scripts' / args.producer):
        raise ValueError('Producer changed')
    if plan['loader_sha256'] != sha(project / 'scripts/probe_candidate_sources.py'):
        raise ValueError('Historical loader changed')
    source = Path(plan['source_run'])
    if plan['source_trace_sha256'] != sha(source / 'server-search-debug.jsonl'):
        raise ValueError('Historical trace changed')
    if summary['decoded_responses'] != plan['expected_calls'] or summary['historical_selections_verified'] != 56:
        raise ValueError('Incomplete response or historical selection replay')
    preflight = root / (args.diagnostic + '-preflight')
    if plan['preflight_plan_sha256'] != sha(preflight / 'plan.json'):
        raise ValueError('Preflight changed')
    before = read(preflight / 'plan.json')
    for key in ('jobs', 'schedule', 'source_run', 'source_trace_sha256', 'script_sha256', 'loader_sha256'):
        if before[key] != plan[key]:
            raise ValueError('Live plan differs from preflight: ' + key)
    models = {'LLM_MODEL': 'gpt-4o-mini', 'CE_MODEL': 'qwen3-rerank',
              'EMBED_MODEL': 'openai/text-embedding-v4', 'EMBED_DIM': 256}
    if any(getattr(config, k) != v for k, v in models.items()):
        raise ValueError('Runtime model changed')
    cfg = project / 'app/config.py'
    module = ast.parse(cfg.read_text())
    module.body = [n for n in module.body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name) and n.value.func.id == '_load_dotenv')]
    defaults = {'__file__': str(cfg)}
    with patch.dict(os.environ, {}, clear=True):
        exec(compile(module, str(cfg), 'exec'), defaults)
    with (project / 'runs/https-fine-context-fill-20261023/control/server-search-debug.jsonl').open() as stream:
        historical = json.loads(next(stream))['versions']['settings']
    bools = {k: v for k, v in historical.items() if type(v) is bool}
    if len(bools) != 65 or any(defaults[k] != v for k, v in bools.items()):
        raise ValueError('Historical boolean defaults changed')
    app_files = {k: v for k, v in read(root / 'plan.json')['source_files'].items()
                 if k.startswith('memory_system/app/')}
    if len(app_files) != 81 or any(sha(project / k.removeprefix('memory_system/')) != v
                                  for k, v in app_files.items()):
        raise ValueError('Application changed from latest full HTTPS')
    archive = root / (args.diagnostic + '-scripts')
    archive.mkdir(exist_ok=False)
    names = {args.producer, args.auditor, Path(__file__).name, 'probe_candidate_sources.py',
             'audit_candidate_sources.py', 'audit_https_funnel.py', 'audit_https_stages.py',
             'evaluation_model_guard.py'}
    for name in sorted(names):
        shutil.copy2(project / 'scripts' / name, archive / name)
    paths = list(run.iterdir()) + list(preflight.iterdir()) + list(archive.iterdir())
    paths += [root / (args.diagnostic + '.log'), root / (args.diagnostic + '-preflight.log')]
    result = dict(source_run=plan['source_run'], summary=summary,
        required_models=models, historical_boolean_defaults_verified=len(bools),
        unchanged_app_files=len(app_files), runtime_promoted=False, new_https_run=False,
        archive_provider_calls=0, decision=args.decision, limitations=[plan['limitation']],
        artifacts={str(p.relative_to(root)): sha(p) for p in paths if p.is_file()})
    with (root / (args.diagnostic + '-study.json')).open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('historical_boolean_defaults_verified',
        'unchanged_app_files', 'runtime_promoted', 'new_https_run', 'decision')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--diagnostic', required=True)
    parser.add_argument('--producer', required=True)
    parser.add_argument('--auditor', required=True)
    parser.add_argument('--decision', required=True)
    main(parser.parse_args())
