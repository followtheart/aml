"""Copy one committed production snapshot for isolated hotfix measurements."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time

root = Path(__file__).resolve().parent
path = root / 'snapshot.sqlite'
if path.exists():
    raise RuntimeError('Snapshot already exists')
source = sqlite3.connect('file:/root/aml/memory_system/memory.db?mode=ro', uri=True)
source.execute('PRAGMA query_only=ON')
source.execute('BEGIN')
source.execute('SELECT rowid FROM source_messages LIMIT 1').fetchone()
started_at = datetime.now(timezone.utc).isoformat()
started = time.monotonic()
last = started

def progress(status, remaining, total):
    global last
    now = time.monotonic()
    if now - started > 240:
        raise TimeoutError('Snapshot exceeded four minutes')
    if now - last > 10 or remaining == 0:
        print(json.dumps({'remaining_pages': remaining, 'total_pages': total,
                          'elapsed_s': round(now-started, 2)}), flush=True)
        last = now

with sqlite3.connect(path) as target:
    source.backup(target, pages=1024, progress=progress)
source.close()
report = {'source': '/root/aml/memory_system/memory.db', 'snapshot_at': started_at,
          'bytes': path.stat().st_size, 'copy_elapsed_s': round(time.monotonic()-started, 2),
          'method': 'Pinned read-only SQLite online backup; no production writes'}
(root / 'snapshot.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
