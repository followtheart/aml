"""Non-secret code/prompt fingerprints for reproducible local evaluations."""
import hashlib
from functools import lru_cache
from pathlib import Path
from . import config


@lru_cache(maxsize=1)
def versions():
    app = Path(__file__).resolve().parent
    prompt_dir = app.parents[1] / 'prompts'
    files = sorted(app.glob('*.py')) + sorted(prompt_dir.glob('*.txt'))
    hashes = {str(p.relative_to(app.parents[1])).replace('\\', '/'):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    fingerprint = hashlib.sha256(repr(sorted(hashes.items())).encode()).hexdigest()
    return {'pipeline_version': fingerprint, 'file_hashes': hashes,
            'model': config.LLM_MODEL, 'embedding_model': config.EMBED_MODEL,
            'settings': {key: getattr(config, key) for key in (
                'CORE_PROFILE_INJECT', 'CORE_PROFILE_TOKEN_BUDGET', 'CORE_PROFILE_MAX_ITEMS',
                'SEARCH_SOURCE_EXCERPT_CHARS', 'SEARCH_SOURCE_MESSAGES_PER_ITEM',
                'RERANK_MAX_CANDIDATES', 'PERSONA_VIEW_FILTER', 'CHOICE_ALIGN_ENABLED',
                'CHOICE_AUTOPICK', 'SEARCH_MAX_ROUNDS')}}
