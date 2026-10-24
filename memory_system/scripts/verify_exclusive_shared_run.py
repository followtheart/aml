"""Final immutable-source, settings, accounting and process checks for this trial."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import socket
from unittest.mock import patch

from evaluation_model_guard import check_provider_log, check_versions


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(args):
    root = args.run.resolve()
    plan = json.loads((root/'plan.json').read_text())
    assert plan['toggle'] == 'PACKET_EXCLUSIVE_SHARED_CONTEXT'
    assert (root/'exit-code').read_text().strip() in ('0', '1')
    for name, expected in plan['source_files'].items():
        assert sha(root/'source'/name) == expected, name
    assert sha(root/'fixture.json') == plan['fixture_sha256']
    assert sha(root/'frozen-memory.db') == plan['database_sha256']
    app_files = {name: h for name, h in plan['source_files'].items() if name.startswith('memory_system/app/')}
    project = Path(__file__).resolve().parents[1]
    current_app_hashes = {name:sha(project/name.removeprefix('memory_system/')) for name in app_files}
    assert current_app_hashes == app_files
    historical = records(args.defaults/'server-search-debug.jsonl')[0]['versions']['settings']
    booleans = {k: v for k, v in historical.items() if type(v) is bool}
    assert len(booleans) == 65
    config_path = project/'app/config.py'
    module = ast.parse(config_path.read_text())
    module.body = [n for n in module.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name) and n.value.func.id == '_load_dotenv')]
    defaults = {'__file__': str(config_path)}
    with patch.dict(os.environ, {}, clear=True):
        exec(compile(module, str(config_path), 'exec'), defaults)
    assert all(defaults[k] == v for k, v in booleans.items())
    assert defaults['PACKET_EXCLUSIVE_SHARED_CONTEXT'] is False
    assert defaults['LLM_MODEL'] == 'gpt-4o-mini'
    settings, arms = {}, {}
    provider_events = []
    for arm in ('control', 'treatment'):
        run = root/arm
        traces = records(run/'server-search-debug.jsonl')
        results = records(run/'completed/data/results/personamem-v2-32k.jsonl')
        assert len(traces) == len(results) == plan['questions']
        assert {t['search_id'] for t in traces} == {r['search_id'] for r in results}
        for t in traces:
            check_versions(t['versions'], arm)
            s = t['versions']['settings']
            assert all(s[k] == v for k, v in booleans.items())
            assert s['PACKET_EXCLUSIVE_SHARED_CONTEXT'] == (arm == 'treatment')
            assert t['cascade']['cross_encoder']['model'] == 'qwen3-rerank'
            assert t['cascade']['conversation_context']['reserve_bytes'] == 5120
            if arm in settings:
                assert settings[arm] == s
            settings[arm] = s
        manifest = json.loads((run/'isolated-manifest.json').read_text())
        assert manifest['audit_exit_code'] == 0 and manifest['client_exit_code'] in (0, 1)
        assert 'server_exit_code' in manifest
        cmdline = Path('/proc')/str(manifest['server_pid'])/'cmdline'
        assert not cmdline.exists() or str(root).encode() not in cmdline.read_bytes()
        log = (run/'server.log').read_text()
        llm_count = check_provider_log(log, arm)
        provider_events.extend(dict(re.findall(r'(\w+)=([^\s]+)', line)) for line in log.splitlines() if 'provider_call ' in line)
        arms[arm] = dict(questions=len(traces), score=sum(r['score'] for r in results),
            terminal_errors=[r['qa_id'] for r in results if r.get('error_stage')],
            ce_partial=[t['search_id'] for t in traces if t['cascade']['cross_encoder']['status'] != 'ok'],
            search_llm_dispatches=llm_count, server_exit_code=manifest['server_exit_code'])
    differences = {k: [settings['control'].get(k), settings['treatment'].get(k)]
        for k in settings['control'].keys() | settings['treatment'].keys()
        if settings['control'].get(k) != settings['treatment'].get(k)}
    assert differences == {'PACKET_EXCLUSIVE_SHARED_CONTEXT': [False, True]}
    reader_log = (root/'reader-provider.log').read_text()
    reader_calls = check_provider_log(reader_log, 'reader')
    provider_events.extend(dict(re.findall(r'(\w+)=([^\s]+)', line)) for line in reader_log.splitlines() if 'provider_call ' in line)
    unknown = [e for e in provider_events if e.get('usage_known') == 'false' or not e.get('total_tokens', '').isdigit()]
    ports = {}
    for port in (args.port, args.port+1):
        with socket.socket() as sock:
            ports[str(port)] = sock.connect_ex(('127.0.0.1', port)) != 0
    assert all(ports.values())
    effect = json.loads((root/'exclusive-shared-effect-audit.json').read_text())
    for arm in arms:
        assert effect['arms'][arm]['summary']['actual_packets_verified'] == plan['questions']
        assert effect['arms'][arm]['summary']['reader_catalogs_verified'] == plan['questions']
        assert not effect['arms'][arm]['summary']['progress_snapshot']
    result = dict(source_files_verified=len(plan['source_files']), current_app_files_verified=len(app_files),
        current_app_changes_from_frozen=[],
        historical_boolean_defaults_verified=len(booleans), settings_differences=differences,
        new_flag_default=False, arms=arms, provider_events=len(provider_events), reader_dispatches=reader_calls,
        unknown_usage_events=unknown, ports_released=ports,
        source_plan_sha256=sha(root/'plan.json'), auditor_sha256=sha(Path(__file__)))
    with (root/'final-verification.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--port', type=int, default=18599)
    p.add_argument('--defaults', type=Path, default=Path('runs/https-fine-context-fill-20261023/control'))
    main(p.parse_args())
