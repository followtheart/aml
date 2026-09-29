import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from app import answer_choice as ac

def enrich(checks, sources):
    diagnostics = {}
    result = ac.enrich_habit_context(checks, sources, diagnostics)
    return result, diagnostics['answer_habit_context']['added']
class HabitContextTests(unittest.TestCase):
 def check(self,kind='habit'):
  return dict(claim_id='A:0',check_type='premise',claim='you practice yoga and meditation daily',premise_type=kind,sources=[dict(id='s0',role='user',quote='I start each morning with slow stretches and deep breaths.')])
 def sources(self):
  return dict(s0=dict(role='user',text='I start each morning with slow stretches and deep breaths.'),s1=dict(role='user',text='Why do yoga and meditation affect stress?'))
 def test_context_is_nonanchor_and_does_not_replace_citations(self):
  check=self.check();before=copy.deepcopy(check);result,added=enrich([check],self.sources())
  self.assertEqual(check,before);self.assertEqual(result[0]['sources'],before['sources']);self.assertEqual(result[0]['claim_id'],'A:0');self.assertEqual(result[0]['topic_context'][0]['anchor'],False);self.assertEqual(added[0]['source_ids'],['s1'])
 def test_other_claim_types_and_whole_options_unchanged(self):
  for kind in ('interest','experience','condition','ownership','unknown'):
   check=self.check(kind);self.assertEqual(enrich([check],self.sources()),([check],[]))
  check=self.check();check['check_type']='option';self.assertEqual(enrich([check],self.sources()),([check],[]))
 def test_no_new_checks_for_an_unsupported_unchecked_claim(self):
  self.assertEqual(enrich([],self.sources()),([],[]))
 def test_assistant_persona_and_personal_assertions_not_background_questions(self):
  for s1 in (dict(role='assistant',text='Why do yoga and meditation affect stress?'),dict(role='user',declared='persona',text='Why do yoga and meditation affect stress?'),dict(role='user',text='I practice yoga and meditation daily. Why does this help me?')):
   sources=self.sources();sources['s1']=s1;check=self.check();self.assertEqual(enrich([check],sources),([check],[]))
 def test_cited_source_not_duplicated(self):
  check=self.check();check['sources'].append(dict(id='s1',role='user',quote=self.sources()['s1']['text']));self.assertEqual(enrich([check],self.sources()),([check],[]))
 def test_limit_and_single_word_overlap(self):
  sources=self.sources();sources.update(s2=dict(role='user',text='How do yoga and meditation compare?'),s3=dict(role='user',text='What distinguishes yoga and meditation?'),s4=dict(role='user',text='Why is yoga popular?'))
  result,added=enrich([self.check()],sources);self.assertEqual(len(result[0]['topic_context']),2);self.assertNotIn('s4',added[0]['source_ids'])
if __name__=='__main__':unittest.main()
