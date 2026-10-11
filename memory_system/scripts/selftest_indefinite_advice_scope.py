import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from indefinite_advice_scope import supports
positives=[(f'{verb} something you enjoy',f'Try {verb} something you enjoy to relax.') for verb in ['cooking','reading','watching','making']]
positives += [('cooking something you enjoy','C. When you’re feeling completely drained, try to make time for rest and simple activities that bring you comfort, like reading, gentle exercise, or cooking something you enjoy. Talking to supportive friends or family members can also help.')]
negatives=[('cooking something you enjoy',x) for x in ['You spend hours cooking something you enjoy.','Since cooking something you enjoy is your daily habit, take a break.','Try cooking something you enjoy from your childhood.','Try cooking something you enjoy as you always do.']]
negatives += [(x,'Try '+x+' to relax.') for x in ['cooking the pasta you enjoy','your interest in cooking','cooking something you enjoyed last year','cooking something your friend enjoys','cooking something you enjoy and own a restaurant']]
for c,o in positives:assert supports(c,o),(c,o)
for c,o in negatives:assert not supports(c,o),(c,o)
print(json.dumps(dict(status='passed',positives=len(positives),negatives=len(negatives))))
