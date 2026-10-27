"""Serial, resource-gated launcher for the preregistered paired HTTPS study."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from batched_records import file_digest, json_records

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/https-screened-source-20261026'
PLAN = json.loads((OUT / 'preregistered-plan.json').read_bytes())
STATE = dict(status='starting', pid=os.getpid(), completed=[], child_pid=None)


def save():
    group = Path('/sys/fs/cgroup') / next(row.split(':', 2)[2].lstrip('/') for row in
        Path('/proc/self/cgroup').read_text().splitlines() if row.startswith('0::'))
    STATE['resources'] = {name: (group/name).read_text().strip() for name in
        ('memory.current', 'memory.peak', 'memory.events', 'cpu.stat') if (group/name).exists()}
    STATE['updated_at'] = time.time()
    temporary = OUT/'state.tmp'
    temporary.write_text(json.dumps(STATE, indent=2)+'\n')
    temporary.replace(OUT/'state.json')


def gate():
    while True:
        values = {row.split(':')[0]: int(row.split()[1]) for row in Path('/proc/meminfo').read_text().splitlines()}
        STATE.update(status='waiting_for_resources', available_memory_mib=values['MemAvailable']/1024,
                     load1=os.getloadavg()[0])
        save()
        if STATE['available_memory_mib'] >= 1536 and STATE['load1'] <= 1.25:
            return
        time.sleep(15)


def execute(name, command, cohort):
    gate()
    with (OUT/(name+'.log')).open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        STATE.update(status='running', stage=name, cohort=cohort, child_pid=child.pid)
        save()
        while child.poll() is None:
            STATE['progress'] = {}
            for arm in ('control', 'treatment'):
                path = OUT/cohort/arm/'completed/data/results/personamem-v2-32k.jsonl'
                if path.exists():
                    count = correct = errors = 0
                    for row in json_records(path, partial=True):
                        count += 1
                        correct += row['score']
                        errors += bool(row.get('error_stage'))
                    STATE['progress'][arm] = dict(completed=count, score=correct, errors=errors)
            save()
            time.sleep(15)
        code = child.returncode
    (OUT/(name+'.exit-code')).write_text(str(code)+'\n')
    STATE['completed'].append(dict(stage=name, exit_code=code))
    STATE['child_pid'] = None
    save()
    return code


def main():
    try:
        for name, digest in PLAN['prerequisite_sha256'].items():
            if file_digest(Path(name)) != digest:
                raise ValueError('Prerequisite artifact changed: '+name)
        for cohort in ('regression', 'validation'):
            for name, expected in PLAN['source_files'].items():
                if file_digest(ROOT.parent/name) != expected:
                    raise ValueError('Preregistered source changed: '+name)
            spec = PLAN['cohorts'][cohort]
            if file_digest(Path(spec['source_run'])/'server-search-debug.jsonl') != spec['source_trace_sha256']:
                raise ValueError('Source trace changed')
            if file_digest(Path(spec['prepared'])) != spec['prepared_sha256']:
                raise ValueError('Prepared data changed')
            if file_digest(Path(spec['source_run'])/'memory.db') != spec['database_sha256']:
                raise ValueError('Source database changed')
            command = [sys.executable, 'scripts/evaluate_frozen_https.py', '--run', spec['source_run'],
                '--prepared', spec['prepared'], '--output', str(OUT/cohort), '--port', str(spec['port']),
                '--concurrency', '1', '--sequential-arms', '--toggle', 'SOURCE_CE_LATE_ADMISSION']
            execute(cohort+'-eval', command, cohort)
            # Preserve every failure; no selected-question or arm reruns.
            execute(cohort+'-audit', [sys.executable, 'scripts/analyze_overflow_https.py',
                '--run', str(OUT/cohort), '--include-reader-errors'], cohort)
        STATE['status'] = 'complete' if all(row['exit_code'] == 0 for row in STATE['completed']) else 'complete_with_failures'
        save()
    except BaseException as exc:
        STATE.update(status='failed', error=type(exc).__name__+': '+str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
