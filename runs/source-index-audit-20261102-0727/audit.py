"""Read-only source_fts coverage/content audit using a separate digest database."""
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/root/aml/memory_system/memory.db')


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    start = time.monotonic()
    report = {'started_at': now(), 'database': str(SOURCE),
              'scope': 'source_messages versus source_fts keys, ownership and content',
              'method': 'single read-only snapshot; SHA-256 of JSON [user_id, content]; indexed joins in separate database',
              'limitations': ['Does not verify internal FTS5 posting-list integrity.']}
    source = sqlite3.connect(SOURCE.as_uri() + '?mode=ro', uri=True, timeout=2)
    source.execute('PRAGMA query_only=ON')
    source.execute('PRAGMA cache_size=-8192')
    source.execute('BEGIN')
    # Establish one committed snapshot for both scans.
    source.execute('SELECT rowid FROM source_messages LIMIT 1').fetchone()
    report['snapshot_at'] = now()
    target_path = ROOT / 'digests.sqlite'
    if target_path.exists():
        raise RuntimeError('Refusing to overwrite an existing audit database')
    target = sqlite3.connect(target_path)
    target.execute('PRAGMA cache_size=-16384')
    target.execute('PRAGMA temp_store=FILE')
    target.execute('PRAGMA synchronous=OFF')
    for table in ('original', 'indexed'):
        target.execute(f'CREATE TABLE {table}(request_id TEXT, message_index INTEGER, digest BLOB)')
    counts = {}
    try:
        for table, dest in [('source_messages', 'original'), ('source_fts', 'indexed')]:
            scan_start = time.monotonic()
            cursor = source.execute(f'SELECT request_id,message_index,user_id,content FROM {table}')
            count = 0
            last_log = time.monotonic()
            while batch := cursor.fetchmany(128):
                transformed = [(rid, idx, hashlib.sha256(json.dumps(
                    [uid, content], ensure_ascii=False, separators=(',', ':')).encode('utf-8')).digest())
                    for rid, idx, uid, content in batch]
                target.executemany(f'INSERT INTO {dest} VALUES (?,?,?)', transformed)
                count += len(batch)
                if time.monotonic() - last_log >= 10:
                    print(json.dumps({'table': table, 'scanned': count,
                                      'elapsed_s': round(time.monotonic()-scan_start, 2)}), flush=True)
                    last_log = time.monotonic()
            target.commit()
            counts[table] = count
            print(json.dumps({'table': table, 'complete': True, 'rows': count,
                              'elapsed_s': round(time.monotonic()-scan_start, 2)}), flush=True)
        source.rollback()
        source.close()
        for table in ('original', 'indexed'):
            target.execute(f'CREATE INDEX {table}_key ON {table}(request_id,message_index)')
        target.commit()
        queries = {
            'missing_index_rows': '''SELECT COUNT(*) FROM original o WHERE NOT EXISTS
                (SELECT 1 FROM indexed i WHERE i.request_id=o.request_id AND i.message_index=o.message_index)''',
            'extra_index_rows': '''SELECT COUNT(*) FROM indexed i WHERE NOT EXISTS
                (SELECT 1 FROM original o WHERE o.request_id=i.request_id AND o.message_index=i.message_index)''',
            'duplicate_index_keys': '''SELECT COUNT(*) FROM
                (SELECT 1 FROM indexed GROUP BY request_id,message_index HAVING COUNT(*)>1)''',
            'duplicate_index_extra_rows': '''SELECT COALESCE(SUM(n-1),0) FROM
                (SELECT COUNT(*) n FROM indexed GROUP BY request_id,message_index HAVING COUNT(*)>1)''',
            'ownership_or_content_mismatches': '''SELECT COUNT(*) FROM indexed i JOIN original o
                ON i.request_id=o.request_id AND i.message_index=o.message_index WHERE i.digest!=o.digest''',
            'null_index_keys': 'SELECT COUNT(*) FROM indexed WHERE request_id IS NULL OR message_index IS NULL',
        }
        findings = {name: target.execute(query).fetchone()[0] for name, query in queries.items()}
        target.close()
        report.update(counts=counts, findings=findings,
                      complete_and_matching=all(n == 0 for n in findings.values()) and len(set(counts.values())) == 1,
                      finished_at=now(), elapsed_s=round(time.monotonic()-start, 2))
        (ROOT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        try:
            source.close()
        finally:
            target.close()


if __name__ == '__main__':
    main()
