"""Follow up the initial candidate timeout with a longer, bounded cold-copy run."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import threading

import benchmark

root = Path(__file__).resolve().parent
path = root / 'snapshot.sqlite'
with sqlite3.connect(path) as c:
    c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
fd = os.open(path, os.O_RDONLY)
try:
    os.fsync(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
finally:
    os.close(fd)
cache_state = subprocess.check_output(['fincore', '--bytes', str(path)], text=True)
print(cache_state, flush=True)
benchmark.TIMEOUT = 120
st = benchmark.store.Store.__new__(benchmark.store.Store)
st.conn = sqlite3.connect(path)
st.conn.row_factory = sqlite3.Row
st.conn.execute('PRAGMA query_only=ON')
st.conn.execute('PRAGMA cache_size=-16384')
st._lock = threading.RLock()
report = {'scope': 'candidate follow-up on the same copy, 120-second per-query cap',
          'cache_preparation': 'POSIX_FADV_DONTNEED requested for isolated copy only; no global cache changes',
          'fincore_before': cache_state, 'cases': []}
prior = json.loads((root / 'benchmark.json').read_text())
try:
    for case, original in zip(json.loads((root / 'private-cases.json').read_text()), prior['cases']):
        result = benchmark.measure(st, benchmark.store.Store.source_search, case)
        row = {'label': case['label'], **result}
        if result['status'] == original['candidate']['status'] == 'ok':
            row['matches_previous_candidate'] = result['digest'] == original['candidate']['digest']
        report['cases'].append(row)
        (root / 'candidate-recheck.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    report['all_completed'] = all(r['status'] == 'ok' for r in report['cases'])
    report['all_comparable_results_match'] = all(r.get('matches_previous_candidate', True) for r in report['cases'])
    (root / 'candidate-recheck.json').write_text(json.dumps(report, indent=2)+'\n')
    assert report['all_completed'] and report['all_comparable_results_match']
finally:
    st.conn.close()
