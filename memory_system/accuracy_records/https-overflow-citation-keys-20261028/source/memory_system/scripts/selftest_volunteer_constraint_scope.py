import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from volunteer_constraint_scope import augment,POLICY
pairs=[dict(pair_id='A:r0',constraint_quote='Please forget that I volunteer for local cleanup projects.',option_quote='Visit the park for a picnic.'),dict(pair_id='B:r0',constraint_quote='Please forget that I volunteer for local cleanup projects.',option_quote='Help out picking up litter at the park.')]
p='Question and instructions unchanged\n<pairs>\n'+json.dumps(pairs)+'\n</pairs>\nOutput format unchanged'
changed,ids=augment(p);assert ids==['A:r0','B:r0'];assert changed.replace(POLICY+'\n','')==p
assert json.loads(changed.split('<pairs>\n')[1].split('\n</pairs>')[0])==pairs
assert augment('unrelated prompt')==('unrelated prompt',[])
other=p.replace('I volunteer for','I enjoy visiting');assert augment(other)==(other,[])
large=p+'x'*48000;assert augment(large)==(large,[])
assert 'helping out' in POLICY and 'not a ban on that place' in POLICY
print(json.dumps(dict(status='passed',checks=6,caveat='Prompt boundary/immutability checks only; semantic behavior requires full replay.')))
