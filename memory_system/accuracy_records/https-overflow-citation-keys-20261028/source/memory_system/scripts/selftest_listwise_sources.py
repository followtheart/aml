"""Shared-source display preserves attribution, baseline admission and budgets."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context, budget, cascade_rerank as ranking, config, cross_encoder, llm
from app.listwise_sources import shared_candidates
from app.schemas import SearchRequest


def row(index, sources):
    body = f'Evidence {index}'
    text = answer_context.with_evidence(body, sources)
    return dict(id=f'm{index}', content=body, _rank_text=text, _fused=1-index/100,
                _coverage_ids=[], _packet_item=dict(content=text, sources=sources))


class Display(unittest.TestCase):
    def test_same_words_different_speakers_and_times_remain_distinct(self):
        sources = [dict(role=role, timestamp=time, content='I do not own a violin.')
                   for role, time in [('user', '2020'), ('assistant', '2020'), ('user', '2021')]]
        rows = [row(i, [s]) for i, s in enumerate(sources)]
        before = copy.deepcopy(rows)
        data = json.loads(shared_candidates(rows).split('\n', 1)[1])
        self.assertEqual(len(data['shared_sources']), 3)
        self.assertEqual(rows, before)
        for value in data['shared_sources'].values():
            self.assertIn('do not own', value)

    def test_omitted_sources_and_unmatched_rank_text_are_kept_whole(self):
        rows = [row(0, [dict(role='user', content_omitted='unit_budget')]),
                dict(id='raw', content='Distinct text', _rank_text='Changed display')]
        data = json.loads(shared_candidates(rows).split('\n', 1)[1])
        self.assertEqual([c['full_text'] for c in data['candidates']], ['Evidence 0', 'Changed display'])


class Admission(unittest.IsolatedAsyncioTestCase):
    async def test_additional_candidates_preserve_baseline_and_original_packets(self):
        source = dict(role='user', timestamp=None, content='A qualified recollection. ' * 70)
        rows = [row(i, [source]) for i in range(8)]
        before = copy.deepcopy(rows)
        req = SearchRequest(user_id='u', query='Evidence', top_k=8)
        ce_inputs, sent = [], []

        async def ce(query, docs, **kwargs):
            ce_inputs.append(list(docs))
            return [.5] * len(docs)

        async def rank(prompt, **kwargs):
            sent.append(prompt)
            n = kwargs['schema']['properties']['ranking']['minItems']
            return dict(ranking=list(range(n)), irrelevant=[], groups=[])

        traces = []
        with patch.multiple(config, LISTWISE_KEYED_OUTPUT=False, CASCADE_COARSE_LIMIT=8, CASCADE_FINE_LIMIT=8,
                            CASCADE_LLM_LIMIT=8, RERANK_MAX_PROMPT_BYTES=7500), \
                patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(llm, 'complete_json', side_effect=rank):
            for enabled in (False, True):
                plan = {}
                with patch.object(config, 'LISTWISE_SHARED_SOURCES', enabled), \
                        budget.scope(seconds=30, calls=20, tokens=100000):
                    await ranking.rank(req, plan, copy.deepcopy(rows))
                traces.append(plan['_cascade']['listwise'])
        self.assertEqual(rows, before)
        self.assertEqual(ce_inputs[0], ce_inputs[1])
        base, treatment = traces
        self.assertTrue(set(base['candidate_ids']) < set(treatment['candidate_ids']))
        self.assertEqual(set(base['candidate_ids']), set(treatment['baseline_candidate_ids']))
        self.assertEqual(len(treatment['candidate_ids']), 8)
        for trace, prompt in zip(traces, sent):
            self.assertEqual(trace['input_bytes'], len(prompt.encode()))
            self.assertLessEqual(trace['input_bytes'], 7500)
            self.assertEqual(trace['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
