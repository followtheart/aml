"""Keep substantive topic words and original evidence while filtering request boilerplate."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as choice, config


class RequestTerms(unittest.TestCase):
    def test_request_words_cannot_supply_second_topic_match(self):
        option='If you’re looking for something funny, try stand-up.'
        source='I’m looking for performances that stand out.'
        with patch.object(config,'CHOICE_REVIEW_REQUEST_TERMS',False):
            self.assertIn('looking',choice.local_context_match(source,choice.contrastive_context_terms(option))['shared_terms'])
        with patch.object(config,'CHOICE_REVIEW_REQUEST_TERMS',True):
            match=choice.local_context_match(source,choice.contrastive_context_terms(option))
            self.assertEqual(match['shared_terms'],['stand'])
            self.assertEqual(match['excerpt'],source)
            self.assertEqual(source[match['source_span']['start']:match['source_span']['end']],source)

    def test_contraction_and_substantive_terms_preserved(self):
        with patch.object(config,'CHOICE_REVIEW_REQUEST_TERMS',True):
            self.assertIn('re',choice.contrastive_context_terms('They’re not alone.'))
            self.assertTrue({'invite','literary'} <= choice.contrastive_context_terms('I’m writing to invite you to a literary evening.'))
            self.assertIn('looking',choice.contrastive_context_terms('A looking-glass for an exhibit.'))
            self.assertIn('painting',choice.contrastive_context_terms('I am looking for painting workshops.'))


if __name__=='__main__':unittest.main()
