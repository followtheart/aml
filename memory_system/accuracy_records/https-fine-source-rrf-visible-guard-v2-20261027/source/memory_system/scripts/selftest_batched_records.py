import json
from pathlib import Path
import tempfile
import unittest

from batched_records import csv_records, file_digest, json_records


class BatchedReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'data'

    def test_utf8_crlf_long_record_and_unterminated_tail(self):
        values = [dict(text='中文🙂' * 100), dict(end=True)]
        self.path.write_bytes(('\r\n'.join(json.dumps(v, ensure_ascii=False) for v in values)).encode())
        self.assertEqual(list(json_records(self.path, block_bytes=3)), values)
        self.assertEqual(list(json_records(self.path, partial=True, block_bytes=3)), values[:1])

    def test_partial_skips_only_unterminated_tail(self):
        self.path.write_bytes(b'{"ok":1}\n\n{"bad"')
        self.assertEqual(list(json_records(self.path, partial=True, block_bytes=2)), [{'ok': 1}])
        with self.assertRaises(json.JSONDecodeError):
            list(json_records(self.path, block_bytes=2))
        self.path.write_bytes(b'bad\n{"ok":1}')
        with self.assertRaises(json.JSONDecodeError):
            list(json_records(self.path, partial=True, block_bytes=2))

    def test_csv_multiline_quotes_crlf_utf8_and_tail(self):
        self.path.write_bytes('a,b\r\n"中文\r\n\"\"quoted\"\"",2\r\nend,3'.encode())
        self.assertEqual(list(csv_records(self.path, block_bytes=2)),
                         [dict(a='中文\r\n"quoted"', b='2'), dict(a='end', b='3')])

    def test_empty_invalid_block_and_digest(self):
        import hashlib
        self.path.write_bytes(b'')
        self.assertEqual(list(json_records(self.path)), [])
        self.assertEqual(file_digest(self.path), hashlib.sha256(b'').hexdigest())
        with self.assertRaises(ValueError):
            list(json_records(self.path, block_bytes=0))


if __name__ == '__main__':
    unittest.main()
