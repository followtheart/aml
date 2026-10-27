"""Scope invariants for conditional antecedents; whole-option checks stay mandatory."""
import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from app import answer_choice as ac
def conditional_claim(text, option, span):
    return ac._nonasserted_scope_reason(dict(text=text, option_span=span), option) == 'conditional_interest_not_asserted'

def unspecified_activity_advice(text, option, span):
    return ac._nonasserted_scope_reason(dict(text=text, option_span=span), option) == 'unspecified_activity_advice'

def rewrite(entries):
    diagnostics = {}
    result = ac.normalize_nonasserted_scopes(entries, diagnostics)
    return result, diagnostics['answer_scope_normalization']['removed']

class ScopeTests(unittest.TestCase):
    def recognize(self,option,text):
        a=option.index(text)
        return conditional_claim(text,option,dict(start=a,end=a+len(text)))
    def test_plain_conditional_interest_does_not_assert_its_antecedent(self):
        for text in ('If you are interested in pottery,','If you’re drawn to quiet landscapes,','If you like cooperative games,','If you prefer acoustic music,','If you are in the mood for a comedy,'):
            self.assertTrue(self.recognize('A. '+text+' try a local event.',text),text)
    def test_presupposed_and_causal_facts_are_retained(self):
        for text in ('Since you enjoy pottery,','If you are up for mixing travel with your love of music,','If you are still interested in pottery,','If you enjoy your regular game nights,','If you are interested in revisiting the place you previously lived,','If you enjoy daily runs,','If you own a telescope,'):
            self.assertFalse(self.recognize('A. '+text+' try a local event.',text),text)
    def test_claim_that_extends_into_body_is_retained(self):
        text='If you like games, because you own a game shop, try a new one.'
        self.assertFalse(self.recognize('A. '+text,text))
    def test_incorrect_offsets_are_not_normalized_into_generic(self):
        option='A. If you like games, try chess.'
        self.assertFalse(conditional_claim('you like games',option,dict(start=0,end=14)))
        self.assertFalse(conditional_claim('you like games',option,None))
    def test_whole_option_still_catches_an_unextracted_personal_assertion(self):
        option='A. If you like games, since you own a game shop, try stocking a new one.'
        text='If you like games,';a=option.index(text)
        entry=dict(letter='A',kind='personal',status='unsupported',option=option,primary_claim=0,
            claims=[dict(text=text,option_span=dict(start=a,end=a+len(text)),status='unsupported',premise_type='interest',reason='no_source',citations=[],validation_errors=[])],
            removed_claims=[],validation_status='valid',validation_errors=[],warnings=[])
        original=copy.deepcopy(entry);updated,removed=rewrite([entry])
        self.assertEqual(entry,original)
        self.assertEqual(updated[0]['kind'],'generic')
        checks=ac.entailment_checks(updated,{},[option])
        self.assertEqual(len(checks),1)
        self.assertEqual(checks[0]['option'],option)
        self.assertIn('since you own a game shop',checks[0]['option'])
        verified=ac.validate_entailments({'checks':[dict(claim_id='A:option',entailed=False)]},updated,checks)
        self.assertEqual(ac.eligible_choices(verified,set()),[])
    def test_invalid_entry_is_not_rescued_by_conditional_syntax(self):
        entry=dict(letter='A',kind='personal',status='unsupported',option='A. If you like games, try chess.',primary_claim=None,claims=[],validation_status='unresolved',validation_errors=['support_unresolved'])
        self.assertEqual(rewrite([entry]),([entry],[]))

class AdviceTests(unittest.TestCase):
 def recognized(self,text,option=None):
  option=option or 'D. '+text+'.';a=option.index(text)
  return unspecified_activity_advice(text,option,dict(start=a,end=a+len(text)))
 def test_unspecified_suggestions(self):
  for text in ('focus on a non-work-related activity you enjoy','try something you like','make time for an activity that you find relaxing','choose any activity you enjoy'):
   self.assertTrue(self.recognized(text),text)
 def test_specific_activities_and_habits_are_retained(self):
  for text in ('focus on the gardening you enjoy','do your weekly activity you enjoy','try the cycling you like','spend time on your hobby','focus on an activity you already enjoy','focus on an activity you enjoy daily'):
   self.assertFalse(self.recognized(text),text)
 def test_asserted_matrix_is_retained(self):
  text='focus on an activity you enjoy'
  for prefix in ('You ', 'Since you ', 'I know you ', 'Because you already ', 'They '):
   self.assertFalse(self.recognized(text,'A. '+prefix+text+'.'))
 def test_invalid_offsets_are_retained(self):
  text='try something you like'
  self.assertFalse(unspecified_activity_advice(text,text,None))
  self.assertFalse(unspecified_activity_advice(text,text,dict(start=True,end=len(text))))
  self.assertFalse(unspecified_activity_advice(text,text,dict(start=1,end=len(text))))
 def entry(self):
  text='try something you like';option='A. '+text+' because you own a pottery studio.';a=option.index(text)
  return dict(letter='A',kind='personal',status='unsupported',option=option,primary_claim=0,claims=[dict(text=text,option_span=dict(start=a,end=a+len(text)),status='unsupported',premise_type='interest',reason='no_source',citations=[],validation_errors=[])],removed_claims=[],validation_status='valid',validation_errors=[],warnings=[])
 def test_whole_option_still_blocks_hidden_assertion(self):
  entry=self.entry();original=copy.deepcopy(entry);updated,removed=rewrite([entry]);self.assertEqual(entry,original);self.assertEqual(len(removed),1)
  checks=ac.entailment_checks(updated,{},[entry['option']]);self.assertEqual(len(checks),1);self.assertEqual(checks[0]['option'],entry['option'])
  verified=ac.validate_entailments({'checks':[dict(claim_id='A:option',entailed=False)]},updated,checks)
  self.assertEqual(ac.eligible_choices(verified,set()),[])
 def test_unresolved_entry_is_not_rescued(self):
  entry=self.entry();entry.update(validation_status='unresolved',validation_errors=['support_unresolved'])
  self.assertEqual(rewrite([entry]),([entry],[]))
class InvalidClaimTests(unittest.TestCase):
    def test_invalid_or_contradicted_claims_are_not_removed(self):
        text = 'try something you like'
        for extra in (dict(validation_errors=['quote_not_in_source']), dict(reason='contradicted')):
            claim = dict(text=text, option_span=dict(start=0, end=len(text)), **extra)
            self.assertIsNone(ac._nonasserted_scope_reason(claim, text))

if __name__ == '__main__':
    unittest.main()
