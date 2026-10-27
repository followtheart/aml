import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from advice_purpose_scope import supports
positive=[('you have a steady source of connection','It might help to create a plan for social time, so you have a steady source of connection.'),('you know you have a safety net in place','It might help to create a plan for essentials, so you know you have a safety net in place.'),('you feel more secure','You could build a plan so you feel more secure.'),('you feel relaxed','Try taking a break so you feel relaxed.')]
negative=[('you have a safety net','You saved money last year, so you have a safety net.'),('you keep those routines','Create a plan so you keep those routines.'),('you know you have a diagnosis','Create a record so you know you have a diagnosis.'),('you have your old savings','Create a plan so you have your old savings.'),('you have a safety net','I suggested creating a plan so you have a safety net.'),('you have a safety net','Create a plan so you have a safety net from before.'),('you feel better','Create a plan so you feel better because you already own a home.'),('you know you have a degree','Create a resume so you know you have a degree.'),('you have a plan','Create a plan. You have a plan.'),('you have a plan','Create a plan so you have a plan; create a plan so you have a plan.')]
# Exact direct-advice forms only; taking is deliberately outside this version.
positive=positive[:-1]
for claim,option in positive:assert supports(claim,option),(claim,option)
for claim,option in negative:assert not supports(claim,option),(claim,option)
print(json.dumps(dict(status='passed',positive=len(positive),negative=len(negative),caveat='Construction boundaries only. Runtime unchanged; whole-option verification still required.')))
