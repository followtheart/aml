"""Run offline self-tests with isolated settings, storage and subprocesses."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tests', nargs='*', help='selftest filenames (default: all)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tests = [root / 'scripts' / name for name in args.tests] if args.tests else sorted(
        (root / 'scripts').glob('selftest_*.py'))
    wrapper = '''import runpy, sys
from pathlib import Path
from unittest.mock import patch
read = Path.read_text
def isolated(path, *args, **kwargs):
    return '' if path.name == '.env' else read(path, *args, **kwargs)
target = sys.argv[1]
sys.path.insert(0, str(Path(target).parent))
sys.argv = [target]
with patch.object(Path, 'read_text', isolated):
    runpy.run_path(target, run_name='__main__')
'''
    failed = []
    with tempfile.TemporaryDirectory(prefix='aml-selftest-') as temp:
        for test in tests:
            if test.parent != root / 'scripts' or not test.is_file():
                parser.error(f'Invalid test: {test.name}')
            env = {key: value for key, value in os.environ.items() if not key.startswith('AML_')}
            env.update(AML_FAKE='1', AML_DB_PATH=str(Path(temp) / (test.stem + '.db')),
                       AML_MEMORY_DEBUG_LOG='', AML_SEARCH_DEBUG_LOG='',
                       PYTHONIOENCODING='utf-8', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            result = subprocess.run([sys.executable, '-c', wrapper, str(test)],
                                    cwd=root, env=env, capture_output=True, text=True,
                                    encoding='utf-8', timeout=120)
            output = result.stdout + result.stderr
            print(f'{test.name}: {"PASS" if result.returncode == 0 else "FAIL"}', flush=True)
            if result.returncode:
                failed.append(test.name)
                print(output, flush=True)
            else:
                for line in output.splitlines():
                    if line.startswith('Ran ') or 'passed' in line:
                        print(line, flush=True)
    print(f'{len(tests) - len(failed)}/{len(tests)} suites passed', flush=True)
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
