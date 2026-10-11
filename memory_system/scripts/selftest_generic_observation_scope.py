import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generic_observation_scope import supports
positive=["I’ve also found that reusable cloths work wonders on stubborn spots.","I have found that short breaks can help concentration.","I've found that a simple checklist helps organize tasks."]
negative=["I’ve found that your old cloths work well.","I’ve found that Maria is improving.","I’ve found that the old injury is healing.","I’ve found that my cats are calming.","I’ve found that running helped after surgery last year.","I’ve found that I enjoy swimming.","I found solace in sharing memories."]
for c in positive:assert supports(c,'A. '+c),c
for c in negative:assert not supports(c,'A. '+c),c
c=positive[0]
assert not supports(c,'A. You could write: “'+c+'”')
assert not supports(c,'A. '+c+' '+c)
assert not supports(c,'A. '+c+' because you already own them.')
print(json.dumps(dict(status='passed',positive=len(positive),negative=len(negative)+3,caveat='Bounded construction tests; personal scope and answer effects still need full reader validation.')))
