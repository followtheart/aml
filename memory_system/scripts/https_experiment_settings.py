"""Explicit values for the single setting varied by a frozen HTTPS experiment."""


def arm_values(toggle, settings):
    if toggle == 'RERANK_MAX_PROMPT_BYTES':
        if type(settings.get(toggle)) is not int or settings[toggle] != 24000:
            raise ValueError('Main-window experiment requires a recorded 24000-byte baseline')
        return dict(control=24000, treatment=32000)
    return dict(control=False, treatment=True)


def fixture_values(fixture):
    expected = arm_values(fixture.get('toggle', 'LISTWISE_OVERFLOW_REVIEW'), fixture['settings'])
    actual = fixture.get('setting_values', expected)
    if (actual != expected or set(actual) != set(expected)
            or any(type(actual[k]) is not type(expected[k]) for k in expected)):
        raise ValueError('Experiment setting values differ from its declared policy')
    return expected


async def run_arms(arm, port, *, sequential=False):
    """Finish and close one arm before starting the other on a small host."""
    if sequential:
        control = await arm('control', port)
        treatment = await arm('treatment', port + 1)
        return [control, treatment]
    import asyncio
    return await asyncio.gather(arm('control', port), arm('treatment', port + 1))
