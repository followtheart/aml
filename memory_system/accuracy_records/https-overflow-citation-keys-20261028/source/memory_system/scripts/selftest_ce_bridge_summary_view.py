"""Boundary/provenance checks for the optional bridge CE scoring view."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context
from ce_bridge_summary_view import PLACEHOLDER, view


def fixture():
    primary = '[inferred; not stated by the user] 主摘要 remains unchanged.'
    summary = 'Bridge summary with 中文 and a literal\n[support: other] marker.'
    prefix = '\n[support: bridge] [inferred; not stated by the user] '
    block = prefix + summary
    sources = [dict(request_id='r', message_index=0, role='user', timestamp=1700000000000,
                    content='Full source with Source messages: and\n[support: bridge] inside the quotation.')]
    body = primary + block
    c = dict(id='primary', content='主摘要 remains unchanged.', type='fact', _bridge_ids=['bridge'],
             _rank_text=answer_context.with_evidence(body, sources),
             _packet_item=dict(_evidence_body=body, sources=sources))
    support = dict(id='bridge', origin='retrieval_bridge', protected=False,
                   block_bytes=len(block.encode()), summary_bytes=len(summary.encode()),
                   block_sha256=hashlib.sha256(block.encode()).hexdigest())
    record = dict(candidate_id=c['id'], body_exact=True, rank_exact=True, supports=[support])
    return c, record, dict(bridge=copy.deepcopy(sources)), primary, prefix


class BridgeViewTests(unittest.TestCase):
    def test_preserves_primary_prefix_and_all_source_bytes(self):
        c, r, originals, primary, prefix = fixture()
        before = copy.deepcopy((c, r, originals))
        text, info = view(c, r, originals)
        self.assertTrue(info['changed'])
        self.assertEqual(text, answer_context.with_evidence(primary + prefix + PLACEHOLDER,
                                                          c['_packet_item']['sources']))
        self.assertEqual((c, r, originals), before)

    def test_protects_primary_rule_and_graph_obligations(self):
        for changes in ({'type': 'rule'}, {'_user_rule': True}, {'_relation_path_ids': ['bridge']},
                        {'_coverage_proofs': [{'seed_ids': ['bridge']}]}, {'_rank_excerpted': True},
                        {'_body_excerpted': True}):
            c, r, originals, *_ = fixture()
            c.update(changes)
            with self.subTest(changes=changes):
                self.assertEqual(view(c, r, originals)[0], c['_rank_text'])
        c, r, originals, *_ = fixture()
        c['_packet_item']['is_constraint'] = True
        self.assertFalse(view(c, r, originals)[1]['changed'])

    def test_protects_declared_and_nested_dependencies(self):
        for origin in ('primary_dependency', 'bridge_dependency'):
            c, r, originals, *_ = fixture()
            r['supports'][0]['origin'] = origin
            self.assertFalse(view(c, r, originals)[1]['changed'])
        c, r, originals, *_ = fixture()
        r['supports'][0]['protected'] = True
        self.assertFalse(view(c, r, originals)[1]['changed'])

    def test_requires_every_original_source_with_same_metadata(self):
        for field, value in [('role', 'assistant'), ('timestamp', 1700000000001),
                             ('message_index', 1), ('content', 'Full source')]:
            c, r, originals, *_ = fixture()
            c['_packet_item']['sources'][0][field] = value
            c['_rank_text'] = answer_context.with_evidence(c['_packet_item']['_evidence_body'],
                                                         c['_packet_item']['sources'])
            with self.subTest(field=field):
                self.assertFalse(view(c, r, originals)[1]['changed'])
        c, r, originals, *_ = fixture()
        originals['bridge'].append(dict(originals['bridge'][0], message_index=2))
        self.assertFalse(view(c, r, originals)[1]['changed'])
        self.assertFalse(view(c, r, {})[1]['changed'])

    def test_refuses_tampered_block_or_source_suffix(self):
        for target in ('block', 'suffix'):
            c, r, originals, *_ = fixture()
            if target == 'block':
                r['supports'][0]['block_sha256'] = '0' * 64
            else:
                c['_rank_text'] += 'tampered'
            with self.subTest(target=target), self.assertRaises(ValueError):
                view(c, r, originals)

    def test_multiple_supports_keep_protected_block_and_original_offsets(self):
        c, r, originals, primary, prefix = fixture()
        r['supports'][0]['protected'] = True
        original_body = c['_packet_item']['_evidence_body']
        summary = 'A separate complete summary containing additional context.'
        second = '\n[support: second] [valid since 2020] ' + summary
        c['_bridge_ids'].append('second')
        c['_packet_item']['_evidence_body'] += second
        c['_rank_text'] = answer_context.with_evidence(c['_packet_item']['_evidence_body'],
                                                     c['_packet_item']['sources'])
        originals['second'] = copy.deepcopy(originals['bridge'])
        r['supports'].append(dict(id='second', origin='retrieval_bridge', protected=False,
            block_bytes=len(second.encode()), summary_bytes=len(summary.encode()),
            block_sha256=hashlib.sha256(second.encode()).hexdigest()))
        text, info = view(c, r, originals)
        self.assertEqual([s['id'] for s in info['spans']], ['second'])
        self.assertEqual(text, answer_context.with_evidence(
            original_body + '\n[support: second] [valid since 2020] ' + PLACEHOLDER,
            c['_packet_item']['sources']))

    def test_missing_provenance_and_non_bridge(self):
        c, r, originals, *_ = fixture()
        with self.assertRaises(ValueError):
            view(c, None, originals)
        c['_bridge_ids'] = []
        self.assertEqual(view(c, None, originals)[0], c['_rank_text'])


if __name__ == '__main__':
    unittest.main()
