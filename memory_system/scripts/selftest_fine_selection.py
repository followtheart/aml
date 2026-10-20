"""Visible-source and supported-coverage selection regressions; no providers."""
import copy
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_read_text = Path.read_text
with patch.object(Path, 'read_text', lambda path, *a, **kw:
                  '' if path.name == '.env' else _read_text(path, *a, **kw)):
    from app import cascade_rerank as cascade, config, cross_encoder, llm, schemas
from selftest_graph_cascade import documents_in, permutation


def source(sid, text):
    return dict(source_event_id=sid, request_id=sid, message_index=0,
                role='user', user_id='u', session_id='s', source_version=1, content=text)


def candidate(mid, score, sources=(), *, supported=(), soft=(), fusion_ids=None):
    sources = copy.deepcopy(list(sources))
    text = f'Unit {mid}: ' + '\n'.join(s['content'] for s in sources)
    return dict(id=mid, content=f'Unit {mid}', _rank_text=text,
                _fixture_score=score, _fused=score,
                _coverage_ids=list(soft), _supported_coverage_ids=list(supported),
                _fusion_source_ids=list(fusion_ids if fusion_ids is not None else
                                        [s['source_event_id'] for s in sources]),
                _packet_item=dict(id=mid, content=text, sources=sources),
                _payload=dict(original_marker=mid))


class VisibleSourceSelectionTests(unittest.TestCase):
    def bounded(self, rows, count=16, requirements=(), fused_ids=(), reserve_ids=()):
        trace = {}
        selected = cascade.shortlist(rows, count, requirements,
            lambda row: (row['_fixture_score'],), fused_ids=fused_ids,
            reserve_ids=reserve_ids, trace=trace, bound_soft_reservations=True)
        return {c['id'] for c in selected}, trace

    def test_far_below_cutoff_fused_hits_do_not_displace_top_sixteen(self):
        rows = [candidate(f'r{i}', 1-i/100, [source(str(i), f'Evidence {i}.')])
                for i in range(50)]
        selected, trace = self.bounded(rows, fused_ids=['r20', 'r26', 'r45'])
        self.assertEqual(selected, {f'r{i}' for i in range(16)})
        self.assertEqual({x['reason'] for x in trace['soft_reservation_policy']['suppressed']},
                         {'outside_ce_window'})

    def test_near_cutoff_fused_and_topic_reserves_share_one_budget(self):
        rows = [candidate(f'r{i}', 1-i/100, [source(str(i), f'Evidence {i}.')])
                for i in range(24)]
        rows[17] = candidate('topic', .83, [source('topic', 'I enjoy weekend surfing.')])
        selected, trace = self.bounded(rows,
            requirements=[dict(id='option:0', text='weekend surfing')],
            fused_ids=['r0', 'r1', 'r16', 'r18'])
        self.assertIn('r16', selected)
        self.assertIn('topic', selected)
        self.assertNotIn('r18', selected)
        self.assertEqual(len(selected), 16)
        self.assertTrue(any(x['candidate_id'] == 'r18' and x['reason'] == 'soft_reservation_budget'
                            for x in trace['soft_reservation_policy']['suppressed']))

    def test_low_rank_supported_witness_and_unscored_rescue_are_exempt(self):
        rows = [candidate(f'r{i}', 1-i/100, [source(str(i), f'Evidence {i}.')])
                for i in range(30)]
        rows[-1]['_supported_coverage_ids'] = ['need']
        selected, _ = self.bounded(rows, requirements=[dict(id='need', text='Required fact')],
                                   fused_ids=['r29'], reserve_ids=['r28'])
        self.assertTrue({'r28', 'r29'} <= selected)

    def test_zero_capacity_has_no_soft_seats(self):
        selected, trace = self.bounded([candidate('a', 1)], count=0, fused_ids=['a'])
        self.assertFalse(selected)
        self.assertEqual(trace['soft_reservation_policy']['limit'], 0)

    def select(self, rows, count, requirements=()):
        before = copy.deepcopy(rows)
        selected = cascade.shortlist(rows, count, requirements,
                                     lambda row: (row['_fixture_score'],))
        self.assertEqual(rows, before, 'shortlisting modified prepared evidence or its payload')
        return [row['id'] for row in selected]

    def test_higher_score_multi_source_candidate_keeps_new_original_evidence(self):
        shared = source('shared', 'I received a request for help from a student.')
        new = source('student', 'The student disclosed troubling circumstances and needs support.')
        rows = [candidate('top', .9, [shared]),
                candidate('multi', .299, [shared, new]),
                candidate('lower', .201, [source('other', 'I asked a general question.')])]
        self.assertEqual(self.select(rows, 2), ['top', 'multi'])

    def test_point_299_beats_point_201_when_same_source_has_a_new_passage(self):
        rows = [candidate('first', .9, [source('same', 'My morning routine includes stretching.')]),
                candidate('newpassage', .299, [source('same', 'I also practice slow breathing.')]),
                candidate('lower', .201, [source('other', 'I read about relaxation research.')])]
        self.assertEqual(self.select(rows, 2), ['first', 'newpassage'])

    def test_four_same_source_restated_facts_leave_room_for_independent_and_new_passage(self):
        shared = source('kitchen', 'I cook at home and want convenient recipes.')
        rows = [candidate(f'restate{i}', .99-i*.01, [shared]) for i in range(4)]
        rows += [candidate('independent', .3, [source('bread', 'I bake fresh bread on weekends.')]),
                 candidate('newpassage', .2, [source('kitchen', 'I do not use meal kits anymore.')])]
        self.assertEqual(self.select(rows, 3), ['restate0', 'independent', 'newpassage'])

    def test_fusion_source_ids_cannot_override_visible_source_content(self):
        shared = source('actual', 'I use Facebook for family updates.')
        rows = [candidate('first', .9, [shared], fusion_ids=['actual']),
                candidate('duplicate', .8, [shared], fusion_ids=['invented-new-origin']),
                candidate('newsource', .7, [source('second', 'I call my sister each week.')],
                          fusion_ids=['actual'])]
        self.assertEqual(self.select(rows, 2), ['first', 'newsource'])

    def test_hidden_source_metadata_does_not_count_as_visible_new_evidence(self):
        shared = source('shared', 'I prefer quiet mornings.')
        duplicate = candidate('hidden', .8, [shared])
        duplicate['_packet_item']['sources'].append(source('hidden-only', 'A distinct unseen statement.'))
        rows = [candidate('first', .9, [shared]), duplicate,
                candidate('visible', .7, [source('visible', 'I stretch before breakfast.')])]
        self.assertEqual(self.select(rows, 2), ['first', 'visible'])

    def test_empty_sources_receive_no_novelty_bonus_over_higher_score_new_passage(self):
        rows = [candidate('first', .9, [source('same', 'I mentor students.')]),
                candidate('passage', .299, [source('same', 'I requested help for a distressed student.')]),
                candidate('unknown', .201, [])]
        self.assertEqual(self.select(rows, 2), ['first', 'passage'])

    def test_negation_and_complementary_passage_are_not_collapsed(self):
        positive = source('same', 'I used to attend electronic music festivals.')
        rows = [candidate('old', .9, [positive]),
                candidate('duplicate', .88, [positive]),
                candidate('negated', .8, [source('same', 'Do not use my electronic music preference anymore.')]),
                candidate('context', .7, [source('other', 'I want a general summer celebration.')])]
        self.assertEqual(self.select(rows, 3), ['old', 'negated', 'context'])

    def assert_distinct_context_survives(self, field, first_value, second_value, text):
        shared = source('same', text)
        rows = [candidate('first', .9, [shared]), candidate('distinct', .8, [shared]),
                candidate('unrelated', .7, [source('other', 'I read a book on the weekend.')])]
        rows[0][field], rows[1][field] = first_value, second_value
        self.assertEqual(self.select(rows, 2), ['first', 'distinct'],
                         f'Different {field} must remain available despite identical source text')
        # An identical qualifier should still permit genuine restatement removal.
        rows[1][field] = copy.deepcopy(first_value)
        self.assertEqual(self.select(rows, 2), ['first', 'unrelated'])

    def test_same_source_different_temporal_ranges_survive_small_pool(self):
        self.assert_distinct_context_survives('temporal',
            dict(start='2020-01-01', end='2020-12-31', precision='year'),
            dict(start='2024-01-01', end='2024-12-31', precision='year'),
            'I taught in 2020 and worked as a librarian in 2024.')

    def test_same_source_different_event_times_survive_small_pool(self):
        self.assert_distinct_context_survives('event_time',
            '2024-03-01T09:00:00Z', '2024-03-08T09:00:00Z',
            'I attended the first workshop on March 1 and the second on March 8.')

    def test_same_source_different_states_survive_small_pool(self):
        self.assert_distinct_context_survives('state',
            dict(slot='occupation', value='teacher'),
            dict(slot='occupation', value='librarian'),
            'I changed occupations from teacher to librarian.')

    def test_same_source_different_knowledge_statuses_survive_small_pool(self):
        self.assert_distinct_context_survives('knowledge_status', 'known', 'unknown',
            'The first appointment is confirmed; the second appointment is still unknown.')

    def test_combined_restatement_is_deferred_only_when_all_its_sources_are_already_visible(self):
        first = source('a', 'I use Facebook for family updates.')
        second = source('b', 'I call my sister each weekend.')
        rows = [candidate('first', .99, [first]), candidate('second', .98, [second]),
                candidate('combined', .97, [first, second]),
                candidate('new', .5, [source('c', 'I meet my cousins each summer.')])]
        self.assertEqual(self.select(rows, 3), ['first', 'second', 'new'])

    def test_soft_coverage_cannot_force_an_unsupported_low_score_candidate(self):
        rows = [candidate('first', .9, [source('a', 'Useful evidence one.')]),
                candidate('second', .8, [source('b', 'Useful evidence two.')]),
                candidate('fake', .01, [source('c', 'Unrelated evidence.')], soft=['need'])]
        self.assertEqual(self.select(rows, 2, [dict(id='need', text='A supported personal premise')]),
                         ['first', 'second'])

    def test_supported_coverage_is_reserved_without_soft_match_metadata(self):
        rows = [candidate('first', .9, [source('a', 'Useful general context.')]),
                candidate('second', .8, [source('b', 'More general context.')]),
                candidate('support', .01, [source('c', 'I surf when visiting the coast.')],
                          supported=['surf'])]
        self.assertEqual(self.select(rows, 2, [dict(id='surf', text='Personal surfing experience')]),
                         ['first', 'support'])


class CascadeCoverageSelectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        settings = patch.multiple(config, LISTWISE_KEYED_OUTPUT=False, CASCADE_COARSE_LIMIT=50, CASCADE_FINE_LIMIT=12,
            CASCADE_LLM_LIMIT=10, CE_BATCH_SIZE=24, CE_MAX_DOCUMENT_BYTES=10000,
            CE_MAX_REQUEST_BYTES=90000, RERANK_MAX_PROMPT_BYTES=24000,
            RERANK_REPAIR_MAX_CALLS=0)
        settings.start()
        self.addCleanup(settings.stop)

    async def run_cascade(self, rows, requirements):
        score_by_id = {row['id']: row['_fixture_score'] for row in rows}
        prompts = []
        async def ce(query, documents, **kwargs):
            return [score_by_id[re.search(r'Unit (\w+):', text)[1]] for text in documents]
        async def listwise(prompt, *args, **kwargs):
            documents = documents_in(prompt)
            prompts.append(documents)
            return permutation(len(documents))
        plan = {'_coverage_requirements': copy.deepcopy(requirements)}
        with patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(llm, 'complete_json', side_effect=listwise):
            result = await cascade.rank(schemas.SearchRequest(user_id='u', query='q', top_k=50),
                                        plan, copy.deepcopy(rows))
        submitted = [re.search(r'\[id: (\w+)\]', text)[1] for text in prompts[0]] if prompts else []
        return result, plan, submitted

    def covered_pool(self, count=6):
        requirements = [dict(id=f'r{i}', text=f'Personal observation {i}') for i in range(count)]
        rows = [candidate(f'general{i}', 10-i*.1,
                          [source(f'general{i}', f'General supporting context {i}.')]) for i in range(20)]
        rows += [candidate(f'covered{i}', 1-i*.01,
                           [source(f'covered{i}', f'Personal observation {i} was reported by the user.')],
                           supported=[f'r{i}'], soft=[f'r{i}']) for i in range(count)]
        return rows, requirements

    async def test_six_supported_champions_survive_fine_to_listwise_for_both_fine_caps(self):
        rows, requirements = self.covered_pool()
        submitted_by_limit = []
        for limit in (12, 20):
            with self.subTest(fine_limit=limit), patch.object(config, 'CASCADE_FINE_LIMIT', limit):
                result, plan, submitted = await self.run_cascade(rows, requirements)
            expected = {f'covered{i}' for i in range(6)}
            self.assertTrue(expected <= set(plan['_cascade']['fine']['selected_ids']))
            self.assertTrue(expected <= set(submitted), 'downstream cap recomputed and lost a fine reservation')
            self.assertTrue(expected <= {row['id'] for row in result if row.get('_cascade_selected')})
            self.assertLessEqual(len(submitted), 10)
            self.assertFalse(plan['_cascade'].get('reservation_missing'))
            submitted_by_limit.append(submitted)
        self.assertEqual(submitted_by_limit[0], submitted_by_limit[1],
                         'expanding fine changed the already-reserved downstream core')

    async def test_insufficient_capacity_names_every_missing_supported_reservation(self):
        rows, requirements = self.covered_pool()
        with patch.object(config, 'CASCADE_LLM_LIMIT', 4):
            _, plan, submitted = await self.run_cascade(rows, requirements)
        missing = {f'covered{i}' for i in range(6)} - set(submitted)
        self.assertEqual(len(missing), 2)
        diagnostics = [item for item in plan['_cascade'].get('reservation_missing', [])
                       if item['stage'] == 'listwise' and item['reason'] == 'capacity']
        self.assertEqual({item['candidate_id'] for item in diagnostics}, missing)
        self.assertEqual({item['requirement_id'] for item in diagnostics},
                         {f'r{mid[len("covered"):]}' for mid in missing})

    async def test_byte_budget_diagnoses_whole_reserved_candidate_omission(self):
        requirements = [dict(id=f'r{i}', text=f'Independent detail {i}') for i in range(2)]
        rows = [candidate(f'long{i}', 1-i*.1, [source(f'long{i}', f'Original {i} ' + str(i)*2990)],
                          supported=[f'r{i}'], soft=[f'r{i}']) for i in range(2)]
        with patch.object(config, 'RERANK_MAX_PROMPT_BYTES', 6500):
            result, plan, submitted = await self.run_cascade(rows, requirements)
        self.assertEqual(len(submitted), 1)
        missing = {row['id'] for row in rows} - set(submitted)
        diagnostics = [item for item in plan['_cascade'].get('reservation_missing', [])
                       if item['stage'] == 'listwise' and item['reason'] == 'prompt_budget']
        self.assertEqual({item['candidate_id'] for item in diagnostics}, missing)
        originals = {row['id']: row for row in rows}
        for row in result:
            self.assertEqual(row['_rank_text'], originals[row['id']]['_rank_text'])
            self.assertEqual(row['_packet_item'], originals[row['id']]['_packet_item'])

    async def test_adaptive_fine_addition_reaches_listwise_without_evicting_core(self):
        rows = [candidate(f'general{i}', 1-i*.01, [source(f's{i}', f'General context {i}.')])
                for i in range(16)]
        rows.append(candidate('newdetail', .01, [source('new', 'I ask about stablecoins and digital currencies.')]))
        with patch.multiple(config, FINE_EVIDENCE_ADAPTIVE=True, FINE_EXTRA_LIMIT=2,
                            CASCADE_FINE_LIMIT=16, CASCADE_LLM_LIMIT=16,
                            RERANK_MAX_PROMPT_BYTES=90000):
            _, plan, submitted = await self.run_cascade(rows,
                [dict(id='question', text='emerging cryptocurrency markets')])
        self.assertEqual(len(submitted), 17)
        self.assertIn('newdetail', submitted)
        self.assertTrue({f'general{i}' for i in range(16)} <= set(submitted))
        self.assertEqual(plan['_cascade']['fine']['output_count'], 17)
        self.assertTrue(plan['_cascade']['adaptive_admission']['decisions'][0]['admitted'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
