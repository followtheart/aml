"""Enforce the fixed LLM requirement for this recall evaluation series."""

REQUIRED_MODEL = 'gpt-4o-mini'


def check_model(model, origin):
    if model != REQUIRED_MODEL:
        raise ValueError(f'{origin}: evaluation requires {REQUIRED_MODEL}')


def check_versions(versions, origin):
    check_model(versions.get('model'), origin)
    settings = versions.get('settings', {})
    if 'LLM_MODEL' in settings:
        check_model(settings['LLM_MODEL'], origin + ' settings')


def check_runtime():
    from app import config
    check_model(config.LLM_MODEL, 'runtime')


def check_provider_log(log, origin):
    """Check recorded dispatch names; cannot attest a proxy's hidden routing."""
    import re
    count = 0
    for line in log.splitlines():
        if 'provider_call ' not in line:
            continue
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
        if fields.get('kind') != 'llm':
            continue
        check_model(fields.get('model'), origin + ' LLM dispatch')
        count += 1
    if not count:
        raise ValueError(origin + ': no recorded LLM dispatches')
    return count
