"""Scope-fragment controls: public entities versus asserted user relationships."""
import argparse
import asyncio
from pathlib import Path

import probe_verification_scope as probe


GUIDANCE = (
    ' For a fragment rather than a complete sentence, recover the grammatical '
    'subject and its relationship from the complete option. A named country, '
    'team, venue or product is not itself a fact about the user. Public events '
    'and their expected outcomes remain general facts even if geographically '
    'relevant. Return true when the option actually asserts a user relationship '
    'to that entity, such as residence, ownership, membership or an existing '
    'preference. Do not infer that relationship from mere mention. A fragment '
    'can still carry a personal assertion through its surrounding clause.'
)


def cases(_):
    rows = [
        ('This year promises thrilling moments for India’s national team.', 'India’s national team', False),
        ('India’s team will be aiming for top honors.', 'India’s team will be aiming for top honors', False),
        ('Since you support India’s national team, consider following the finals.', 'India’s national team', True),
        ('Because you live in Jamshedpur, local events may be convenient.', 'Jamshedpur', True),
        ('Jamshedpur will host an exhibition this winter.', 'Jamshedpur', False),
        ('You could visit the gallery on Saturday.', 'the gallery', False),
        ('Use your existing gallery membership to book a visit.', 'gallery membership', True),
        ('The latest camera offers improved stabilization.', 'camera', False),
        ('Since you own this camera, try its stabilization mode.', 'this camera', True),
        ('You do not enjoy cricket, so consider another sport.', 'cricket', True),
        ('If you enjoy cricket, you could follow the tournament.', 'cricket', False),
        ('Your favorite cricket tournament starts next month.', 'cricket tournament', True),
    ]
    return [dict(claim_id=f'anchor_{i}', option='A. '+o, claim=c,
                 expected=e, origin='synthetic_scope_control')
            for i, (o, c, e) in enumerate(rows)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.inventory = None
    probe.cases = cases
    probe.GUIDANCE = GUIDANCE
    asyncio.run(probe.main(args))
