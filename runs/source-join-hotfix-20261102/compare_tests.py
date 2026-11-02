"""Identify pre-existing test failures without changing the checkout or service."""
import argparse
import json
import os
from pathlib import Path
import sys
import unittest

os.environ['AML_FAKE'] = '1'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/root/aml/memory_system')
sys.path.insert(0, '/root/aml/memory_system/scripts')
from app import store

parser = argparse.ArgumentParser()
parser.add_argument('arm', choices=['baseline', 'candidate'])
args = parser.parse_args()
if args.arm == 'baseline':
    exec(compile((ROOT / 'baseline_store.py').read_text(), str(ROOT / 'baseline_store.py'), 'exec'), store.__dict__)
modules = ['selftest_evidence_recovery', 'selftest_recall_context', 'selftest_search_storage',
           'selftest_porter_recall', 'selftest_retrieval_quality']
suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = {'arm': args.arm, 'tests_run': result.testsRun,
          'failures': sorted(test.id() for test, _ in result.failures),
          'errors': sorted(test.id() for test, _ in result.errors)}
(ROOT / (args.arm + '-tests.json')).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
