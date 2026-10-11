"""Window indexing must never paraphrase evidence or duplicate source votes."""
import copy
import unittest
from replay_source_vector_route import validate_probe_granularity


class SourceWindowProbeTests(unittest.TestCase):
    def fixture(self):
        return dict(user_only=True, sentence_window=2, source_aggregation='max_window_per_source',
                    sources=[dict(role='user', content='A. B.'), dict(role='user', content='C. D.')],
                    documents=[dict(source_index=0, start=0, end=5, text='A. B.'),
                               dict(source_index=1, start=0, end=5, text='C. D.')],
                    rankings={'query': {k: [dict(document_index=0, source_index=0, score=.8),
                                            dict(document_index=1, source_index=1, score=.4)]
                                        for k in ('all', 'user')}})

    def test_valid_original_spans(self):
        validate_probe_granularity(self.fixture())

    def test_rewritten_or_bad_offsets_rejected(self):
        for update in [dict(text='A. E.'), dict(start=-1), dict(end=6)]:
            probe=self.fixture();probe['documents'][0].update(update)
            with self.assertRaises(ValueError):validate_probe_granularity(probe)

    def test_duplicates_missing_sources_and_wrong_identity_rejected(self):
        for mode in ('duplicate', 'missing', 'wrong_source'):
            probe=self.fixture();hits=probe['rankings']['query']['user']
            if mode=='duplicate':hits[1]=copy.deepcopy(hits[0])
            elif mode=='missing':hits.pop()
            else:hits[0]['document_index']=1
            with self.assertRaises(ValueError):validate_probe_granularity(probe)

    def test_nonfinite_or_unsorted_scores_rejected(self):
        for score in (float('nan'), float('inf'), .9):
            probe=self.fixture();probe['rankings']['query']['user'][1]['score']=score
            with self.assertRaises(ValueError):validate_probe_granularity(probe)

    def test_uncollapsed_or_assistant_probe_rejected(self):
        for update in [dict(source_aggregation='none'), dict(user_only=False)]:
            probe=self.fixture();probe.update(update)
            with self.assertRaises(ValueError):validate_probe_granularity(probe)


if __name__=='__main__':unittest.main()
