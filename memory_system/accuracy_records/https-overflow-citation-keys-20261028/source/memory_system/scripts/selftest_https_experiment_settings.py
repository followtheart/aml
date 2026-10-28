"""Ensure numeric HTTPS interventions cannot silently become boolean toggles."""
import json
import asyncio
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, search_pipeline, store
from https_experiment_settings import arm_values, fixture_values, run_arms
import evaluate_frozen_https as runner


class ExperimentSettingsTests(unittest.TestCase):
    def test_batched_jsonl_boundaries(self):
        rows = [dict(text='中文🙂\n引文', value=1), dict(text='长记录' * 100), dict(last=True)]
        payload = b'\r\n' + b'\r\n \t\n'.join(json.dumps(row, ensure_ascii=False).encode()
                                              for row in rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'records.jsonl'
            path.write_bytes(payload)
            for size in (1, 2, 7, 31, 4 * 1024 * 1024):
                with self.subTest(block_bytes=size):
                    self.assertEqual(list(runner.records(path, size)), rows)
            with self.assertRaises(ValueError):
                list(runner.records(path, 0))
            path.write_bytes(b'\n\r\n \t')
            self.assertEqual(list(runner.records(path, 2)), [])
            path.write_bytes(b'{"ok":1}\n{"truncated":')
            with self.assertRaises(json.JSONDecodeError):
                list(runner.records(path, 3))

    def test_legacy_boolean_fixture(self):
        self.assertEqual(fixture_values(dict(settings={}, toggle='LISTWISE_OVERFLOW_REVIEW')),
                         dict(control=False, treatment=True))

    def test_numeric_values(self):
        self.assertEqual(arm_values('RERANK_MAX_PROMPT_BYTES', {'RERANK_MAX_PROMPT_BYTES':24000}),
                         dict(control=24000,treatment=32000))

    def test_numeric_baseline_is_not_inferred(self):
        for value in (True,False,24000.0,32000,None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                arm_values('RERANK_MAX_PROMPT_BYTES',{'RERANK_MAX_PROMPT_BYTES':value})

    def test_tampered_values_rejected(self):
        for values in ({'control':24000,'treatment':64000},{'control':False,'treatment':True},
                       {'control':24000.0,'treatment':32000}, {'control':24000}):
            with self.subTest(values=values),self.assertRaises(ValueError):
                fixture_values(dict(settings={'RERANK_MAX_PROMPT_BYTES':24000},
                    toggle='RERANK_MAX_PROMPT_BYTES',setting_values=values))

    def test_server_applies_values_before_starting_app(self):
        fixture=dict(settings={'RERANK_MAX_PROMPT_BYTES':24000},requests=[],
                     toggle='RERANK_MAX_PROMPT_BYTES',setting_values=dict(control=24000,treatment=32000))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.json';path.write_text(json.dumps(fixture))
            for arm,expected in fixture['setting_values'].items():
                observed=[]
                def observe(*args,**kwargs):
                    observed.append(config.RERANK_MAX_PROMPT_BYTES)
                with patch.object(config,'RERANK_MAX_PROMPT_BYTES'), \
                     patch.object(search_pipeline,'_understand'),patch.object(search_pipeline,'_anchor_time'), \
                     patch.object(store.Store,'record_recall'), \
                     patch.dict(sys.modules,{'app.main':SimpleNamespace(app=object())}), \
                     patch('uvicorn.run',side_effect=observe):
                    runner.serve(SimpleNamespace(fixture=path,arm=arm,port=12345,cert='unused',key='unused'))
                self.assertEqual(observed,[expected])
                self.assertIs(type(observed[0]),int)


class ArmScheduleTests(unittest.IsolatedAsyncioTestCase):
    async def test_serial_schedule_closes_control_before_starting_treatment(self):
        events = []
        async def arm(name, port):
            events.append(('start', name, port))
            await asyncio.sleep(0)
            events.append(('closed', name, port))
            return name == 'control'
        self.assertEqual(await run_arms(arm, 19001, sequential=True), [True, False])
        self.assertEqual(events, [('start', 'control', 19001), ('closed', 'control', 19001),
                                  ('start', 'treatment', 19002), ('closed', 'treatment', 19002)])

    async def test_parallel_option_remains_available(self):
        started = []
        async def arm(name, port):
            started.append(name)
            await asyncio.sleep(0)
            self.assertEqual(len(started), 2)
            return False
        self.assertEqual(await run_arms(arm, 19001), [False, False])


if __name__=='__main__':
    unittest.main()
