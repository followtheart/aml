import json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac
from constraint_term_forms import normalize,wrapper
for a,b in [('libraries','library'),('activities','activity'),('hobbies','hobby'),('theatre','theater'),('THEATRES','theaters')]: assert normalize(a)==b
for word in ['movies','cookies','series','species','ties','pies']: assert normalize(word)==word
original=ac._topic_terms
with patch.object(ac,'_topic_terms',wrapper(original)):
 assert ac._scope_overlap('Please forget that I like visiting public libraries.','You could read at your local public library.')
 assert ac._scope_overlap('Please forget that I’m interested in contemporary theater performances.','Try that contemporary theatre piece.')
 assert not ac._scope_overlap('Please forget that I like visiting public libraries.','Spend time in a public park.')
 assert not ac._scope_overlap('Please forget that I cook vegetable soups.','Eat vegetable soups at a restaurant.')
 assert not ac._scope_overlap('Please forget that I like documentaries.','Read a thought-provoking book.')
print(json.dumps(dict(status='passed',checks=16,caveat='Constraint proposal normalization only, not semantic violation labels.')))
