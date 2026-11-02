"""Exercise ASGI startup and first search on the previous isolated snapshot."""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parent
cfg = json.loads(Path('/root/memory-system-runtime/service.json').read_text())
env = dict(cfg['env'])
env.update(AML_DB_PATH=str(root.parent / 'source-join-hotfix-20261102/snapshot.sqlite'),
           AML_ACCESS_LOG=str(root / 'isolated-access.jsonl'),
           AML_MEMORY_DEBUG_LOG='', AML_SEARCH_DEBUG_LOG='',
           AML_SOURCE_FTS_BACKFILL='0', AML_SESSION_FTS_BACKFILL='0')
command = list(cfg['command'])
command[command.index('--host') + 1] = '127.0.0.1'
command[command.index('--port') + 1] = '18680'
os.chdir(cfg['cwd'])
os.execve(command[0], command, env)
