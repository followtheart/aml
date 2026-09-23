"""Offline regressions for the dependency-free dotenv loader."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


CONFIG_PATH = Path(__file__).resolve().parents[1] / 'app' / 'config.py'


class DotenvTests(unittest.TestCase):
    def load(self, contents, existing=None):
        # Import an isolated module with synthetic files and environment only.
        # No test reads or edits the developer's real .env or credentials.
        spec = importlib.util.spec_from_file_location('dotenv_test_config', CONFIG_PATH)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, existing or {}, clear=True), \
                patch.object(Path, 'exists', return_value=True), \
                patch.object(Path, 'read_text', return_value=contents):
            spec.loader.exec_module(module)
            values = dict(os.environ)
        return module, values

    def test_numeric_and_boolean_inline_comments_allow_startup(self):
        module, values = self.load(
            'AML_CHOICE_JEV_MIN_CONFIDENCE=0.8 # review below this threshold\n'
            'AML_JEV_TIMEOUT_SECONDS=20\t# seconds\n'
            'AML_JEV_MAX_REQUEST_BYTES=48000 # request size\n'
            'AML_CHOICE_JEV_SUPPORT=1 # enable classification\n')
        self.assertEqual(module.CHOICE_JEV_MIN_CONFIDENCE, .8)
        self.assertEqual(module.JEV_TIMEOUT_SECONDS, 20)
        self.assertEqual(module.JEV_MAX_REQUEST_BYTES, 48000)
        self.assertTrue(module.CHOICE_JEV_SUPPORT)
        self.assertEqual(values['AML_CHOICE_JEV_SUPPORT'], '1')

    def test_quoted_values_keep_hashes_and_inner_spaces(self):
        _, values = self.load(
            'EXAMPLE_SINGLE=\'keep # this\' # remove this\n'
            'EXAMPLE_DOUBLE="  keep # this  " # remove this\n'
            'EXAMPLE_QUOTED_EMPTY="" # nothing\n'
            'EXAMPLE_LITERAL_QUOTE="\'literal\'"\n')
        self.assertEqual(values['EXAMPLE_SINGLE'], 'keep # this')
        self.assertEqual(values['EXAMPLE_DOUBLE'], '  keep # this  ')
        self.assertEqual(values['EXAMPLE_QUOTED_EMPTY'], '')
        self.assertEqual(values['EXAMPLE_LITERAL_QUOTE'], "'literal'")

    def test_unspaced_hashes_are_literal(self):
        _, values = self.load(
            'EXAMPLE_TOKEN=synthetic#hash # trailing comment\n'
            'EXAMPLE_URL=https://example.invalid/page#section\n'
            'EXAMPLE_HASH=#literal\n'
            'EXAMPLE_EMPTY= # comment only\n')
        self.assertEqual(values['EXAMPLE_TOKEN'], 'synthetic#hash')
        self.assertEqual(values['EXAMPLE_URL'], 'https://example.invalid/page#section')
        self.assertEqual(values['EXAMPLE_HASH'], '#literal')
        self.assertEqual(values['EXAMPLE_EMPTY'], '')

    def test_windows_backslashes_are_not_decoded(self):
        _, values = self.load(
            'EXAMPLE_PATH=C:\\new\\test # trailing comment\n'
            'EXAMPLE_QUOTED_PATH="C:\\new folder\\test\\" # trailing comment\n')
        self.assertEqual(values['EXAMPLE_PATH'], 'C:\\new\\test')
        self.assertEqual(values['EXAMPLE_QUOTED_PATH'], 'C:\\new folder\\test\\')

    def test_apostrophe_inside_unquoted_value_does_not_quote_comment(self):
        _, values = self.load(
            "EXAMPLE_PATH=C:\\Users\\O'Brien\\memory.db # local database\n")
        self.assertEqual(values['EXAMPLE_PATH'], "C:\\Users\\O'Brien\\memory.db")

    def test_existing_environment_wins_and_ignored_lines_stay_ignored(self):
        module, values = self.load(
            '\n# comment\nnot an assignment\n'
            ' AML_CHOICE_JEV_MIN_CONFIDENCE = 0.8 # default\n'
            'EXAMPLE_VALUE=file # default\n',
            {'AML_CHOICE_JEV_MIN_CONFIDENCE': '0.9', 'EXAMPLE_VALUE': 'environment'})
        self.assertEqual(module.CHOICE_JEV_MIN_CONFIDENCE, .9)
        self.assertEqual(values['EXAMPLE_VALUE'], 'environment')
        self.assertEqual(set(values), {'AML_CHOICE_JEV_MIN_CONFIDENCE', 'EXAMPLE_VALUE'})


if __name__ == '__main__':
    unittest.main()
