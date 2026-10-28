"""Offline regressions for deterministic persona ingestion and its answer-layer acceptance."""
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['AML_FAKE'] = '1'
_read_text = Path.read_text


def _without_dotenv(path, *args, **kwargs):
    return '' if path.name == '.env' else _read_text(path, *args, **kwargs)


with patch.object(Path, 'read_text', _without_dotenv):
    from app import answer_choice, persona_source

PERSONA = '''[system] You are an AI assistant helping a user with the following persona:

{
  "short_persona": {
    "persona": "A liberal Democrat from Kansas."
  },
  "name": "Daniel Robert Whitaker",
  "age": 38,
  "gender": "Male",
  "sexual_orientation": "Heterosexual",
  "location": {
    "city": "Lawrence",
    "state": "Kansas",
    "type": "Mid-sized college town"
  },
  "occupation": {
    "title": "High School Social Studies Teacher",
    "employer": "Public School District in Douglas County",
    "years_in_role": 11
  },
  "family_background": {
    "marital_status": "Married",
    "spouse": {
      "name": "Melissa Whitaker",
      "age": 36,
      "occupation": "Librarian"
    },
    "children": [
      {
        "name": "Evan",
        "age": 8
      },
      {
        "name": "Sophie",
        "age": 5
      }
    ]
  },
  "hobbies_interests": [
    "Cycling along the Kansas River trail",
    "Cooking vegetarian meals"
  ],
  "technology_use": {
    "devices": [
      "Laptop"
    ],
    "social_media": [
      "Facebook for family updates"
    ]
  }
}'''


def request(content, role='user'):
    message = types.SimpleNamespace(role=role, content=content, timestamp=1700000000000)
    return types.SimpleNamespace(messages=[message], request_id='persona-test')


class ExtractionTests(unittest.TestCase):
    def facts(self, content=PERSONA, role='user'):
        return persona_source.persona_facts(request(content, role), [], [[0]])

    def test_persona_json_becomes_atomic_profile_and_preference_facts(self):
        facts = self.facts()
        contents = {f['content'] for f in facts}
        self.assertIn("The user's name is Daniel Robert Whitaker.", contents)
        self.assertIn('The user lives in the city of Lawrence.', contents)
        self.assertIn('The user works as a High School Social Studies Teacher.', contents)
        self.assertIn("The user's spouse is Melissa Whitaker (age 36, occupation Librarian).", contents)
        self.assertIn("The user's child is Evan (age 8).", contents)
        self.assertIn("The user's child is Sophie (age 5).", contents)
        self.assertIn('The user enjoys cycling along the Kansas River trail.', contents)
        self.assertIn('The user uses Facebook for family updates.', contents)
        self.assertNotIn('Mid-sized college town', ' '.join(contents))
        types_ = {f['content']: f['type'] for f in facts}
        self.assertEqual(types_['The user enjoys cooking vegetarian meals.'], 'preference')
        self.assertEqual(types_['The user lives in the city of Lawrence.'], 'profile')

    def test_every_fact_quotes_a_verbatim_span_and_is_tagged_persona(self):
        for fact in self.facts():
            with self.subTest(content=fact['content']):
                evidence = fact['evidence'][0]
                self.assertIn(evidence['quote'], PERSONA)
                self.assertEqual(evidence['source_role'], 'persona')
                self.assertEqual(fact['_sources'], [0])
                self.assertTrue(fact['_persona'])
                self.assertIn('persona', fact['keywords'])

    def test_sensitive_attributes_are_vaulted_without_graph_triples(self):
        fact = next(f for f in self.facts() if 'Heterosexual' in f['content'])
        self.assertEqual(fact['sensitivity'], 'sensitive')
        self.assertEqual(fact['triples'], [])
        interest = next(f for f in self.facts() if 'cycling' in f['content'])
        self.assertEqual(interest['triples'][0]['relation'], 'interested_in')

    def test_non_persona_or_assistant_messages_produce_nothing(self):
        self.assertEqual(self.facts('I live in Lawrence. {"city": "Lawrence"}'), [])
        self.assertEqual(self.facts(PERSONA, role='assistant'), [])
        self.assertEqual(self.facts('[system] persona: not json {'), [])

    def test_existing_extracted_fact_is_not_duplicated(self):
        existing = [dict(content="The user's name is Daniel Robert Whitaker.", type='profile')]
        facts = persona_source.persona_facts(request(PERSONA), existing, [[0]])
        self.assertEqual(sum(f['content'] == "The user's name is Daniel Robert Whitaker." for f in facts), 1)


class AnswerAcceptanceTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        from selftest_answer_choice import memory, seal
        self.packet = seal(memory(PERSONA, summary='The user lives in the city of Lawrence.'))

    def test_persona_quote_supports_a_personal_premise_as_self_report(self):
        sources, _ = answer_choice.build_catalog(self.packet)
        sid, card = next(iter(sources.items()))
        self.assertEqual(card['declared'], 'persona')
        payload = {'options': [
            dict(letter='A', kind='personal', claims=[dict(text='you live in Lawrence', status='supported', citations=[
                dict(source_id=sid, quote='"city": "Lawrence"', basis='self_report', subject='current_user')])]),
            dict(letter='B', kind='generic', claims=[])]}
        entries = answer_choice.validate_assessments(payload, ['A. Since you live in Lawrence, visit the river.', 'B. Rest.'], sources)
        self.assertEqual(entries[0]['status'], 'supported')
        self.assertEqual(entries[0]['claims'][0]['citations'][0]['source_role'], 'persona')
        self.assertEqual(answer_choice.eligible_choices(entries, set()), ['A'])
        self.assertEqual(answer_choice._cards(sources)[0]['role'], 'persona (first-party profile)')

    def test_persona_source_cannot_prove_a_third_party_subject(self):
        sources, _ = answer_choice.build_catalog(self.packet)
        sid = next(iter(sources))
        payload = {'options': [
            dict(letter='A', kind='personal', claims=[dict(text='you live in Lawrence', status='supported', citations=[
                dict(source_id=sid, quote='"city": "Lawrence"', basis='self_report', subject='third_party')])]),
            dict(letter='B', kind='generic', claims=[])]}
        entries = answer_choice.validate_assessments(payload, ['A. Since you live in Lawrence, visit the river.', 'B. Rest.'], sources)
        self.assertEqual(answer_choice.eligible_choices(entries, set()), ['B'])


if __name__ == '__main__':
    unittest.main()
