"""Run the HTTPS experiment against a frozen, isolated local TLS service.

Uses current provider configuration, a fresh database and a localhost-only
server. Does not restart the existing service. Forward evaluator options after
--, for example -- --no-progress. Run artifacts include source and log snapshots.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import config  # Load the existing provider environment without copying secrets.
from evaluation_model_guard import check_runtime


def main():
    check_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18443)
    parser.add_argument('--min-free-mb', type=int, default=512,
                        help='Required free disk space before creating experiment artifacts')
    parser.add_argument('eval_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.min_free_mb < 0:
        parser.error('--min-free-mb must be nonnegative')
    existing = run.parent
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < args.min_free_mb * 1024 * 1024:
        parser.error(f'Insufficient disk space: {free // (1024 * 1024)} MiB free; '
                     f'{args.min_free_mb} MiB required')
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / 'source'
    project = snapshot / 'memory_system'
    for directory in ('app', 'scripts'):
        shutil.copytree(ROOT / directory, project / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ROOT.parent / 'prompts', snapshot / 'prompts')
    lexicon = ROOT / 'data/lexicons/wordnet.zip'
    if lexicon.is_file():
        target = project / 'data/lexicons/wordnet.zip'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lexicon, target)
    (project / '.venv').symlink_to(ROOT / '.venv', target_is_directory=True)
    cert, key = run / 'localhost.crt', run / 'localhost.key'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                    '-keyout', str(key), '-out', str(cert), '-days', '2',
                    '-subj', '/CN=localhost', '-addext', 'subjectAltName=DNS:localhost'],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    key.chmod(0o600)
    env = os.environ.copy()
    env.update(AML_API_KEY=secrets.token_urlsafe(32), AML_DB_PATH=str(run / 'memory.db'),
               AML_MEMORY_DEBUG_LOG=str(run / 'server-memory-debug.jsonl'),
               AML_SEARCH_DEBUG_LOG=str(run / 'server-search-debug.jsonl'),
               AML_ACCESS_LOG=str(run / 'server-access.jsonl'),
               AML_HTTPS_URL=f'https://localhost:{args.port}', AML_HTTPS_CA=str(cert),
               AML_EXPERIMENT_DIR=str(run))
    env['AML_EVAL_DATA'] = str(Path(env.get('AML_EVAL_DATA',
                                         ROOT / 'data/prepared/personamem-v2-32k.jsonl')).resolve())
    env.pop('AML_HTTPS_TOKEN_FILE', None)
    env.pop('PYTHONPATH', None)
    files = sorted(p for directory in (project / 'app', project / 'scripts',
        snapshot / 'prompts', project / 'data/lexicons') if directory.exists()
        for p in directory.rglob('*') if p.is_file())
    manifest = dict(url=env['AML_HTTPS_URL'], source='frozen current worktree',
                    data=env['AML_EVAL_DATA'], fake=config.FAKE,
                    data_sha256=hashlib.sha256(Path(env['AML_EVAL_DATA']).read_bytes()).hexdigest(),
                    evaluation={k: int(env.get(k, default)) for k, default in (
                        ('AML_EVAL_CONVS', 1), ('AML_EVAL_LIMIT', 30),
                        ('AML_EVAL_CHUNK_MESSAGES', 20), ('AML_EVAL_CHUNK_WORDS', 2000))},
                    files={str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in files})
    manifest_path = run / 'isolated-manifest.json'

    def save_manifest():
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(manifest_path)

    save_manifest()
    server = None
    try:
        with (run / 'server.log').open('w') as log:
            server = subprocess.Popen(
                [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1',
                 '--port', str(args.port), '--ssl-certfile', str(cert), '--ssl-keyfile', str(key),
                 '--no-proxy-headers'], cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
            manifest['server_pid'] = server.pid
            save_manifest()
            context = ssl.create_default_context(cafile=str(cert))
            for _ in range(60):
                if server.poll() is not None:
                    raise RuntimeError('Isolated server exited; inspect server.log')
                try:
                    with urllib.request.urlopen(env['AML_HTTPS_URL'] + '/health',
                                                context=context, timeout=1) as response:
                        if json.load(response).get('status') == 'ok':
                            break
                except OSError:
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError('Isolated server did not become healthy')
            extra = args.eval_args[1:] if args.eval_args[:1] == ['--'] else args.eval_args
            print(f'isolated_server_pid={server.pid} experiment_dir={run}', flush=True)
            client = subprocess.run(['bash', 'scripts/run_experiment_https.sh', *extra],
                                    cwd=project, env=env)
            manifest['client_exit_code'] = client.returncode
            save_manifest()
            results = run / 'completed/data/results' / env.get('AML_EVAL_RESULT_NAME', 'personamem-v2-32k.jsonl')
            if results.exists() and results.stat().st_size:
                with open(env['AML_EVAL_DATA']) as stream:
                    dataset = json.loads(next(line for line in stream if line.strip())).get('dataset')
                audit_script = 'audit_temporal_https.py' if dataset in ('personamem-v3', 'perma') else 'audit_https_funnel.py'
                audit_args = ['--dataset', dataset] if dataset in ('personamem-v3', 'perma') else []
                audit = subprocess.run([sys.executable, str(ROOT / 'scripts' / audit_script),
                                        '--run', str(run), '--prepared', env['AML_EVAL_DATA'],
                                        '--memory-log', env['AML_MEMORY_DEBUG_LOG'],
                                        '--search-log', env['AML_SEARCH_DEBUG_LOG'], *audit_args], cwd=ROOT)
                manifest['audit_exit_code'] = audit.returncode
                save_manifest()
                if audit.returncode:
                    return audit.returncode
            return client.returncode
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            manifest['server_exit_code'] = server.returncode
            save_manifest()


if __name__ == '__main__':
    sys.exit(main())
