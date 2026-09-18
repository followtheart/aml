"""Non-secret code/prompt fingerprints for reproducible local evaluations."""
import hashlib
from functools import lru_cache
from pathlib import Path
from . import config

SEARCH_POLICY = 'graph_cascade_v8'


@lru_cache(maxsize=1)
def versions():
    app = Path(__file__).resolve().parent
    prompt_dir = app.parents[1] / 'prompts'
    files = sorted(app.glob('*.py')) + sorted(prompt_dir.glob('*.txt'))
    hashes = {str(p.relative_to(app.parents[1])).replace('\\', '/'):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    fingerprint = hashlib.sha256(repr(sorted(hashes.items())).encode()).hexdigest()
    return {'pipeline_version': fingerprint, 'search_policy': SEARCH_POLICY,
            'answer_policy': 'direct_evidence_v2', 'file_hashes': hashes,
            'model': config.LLM_MODEL, 'embedding_model': config.EMBED_MODEL,
            'cross_encoder_model': config.CE_MODEL,
            'settings': {key: getattr(config, key) for key in (
                'CORE_PROFILE_MAX_ITEMS', 'QUERY_PROFILE_DIGEST',
                'SEARCH_SOURCE_EXCERPT_CHARS', 'SEARCH_SOURCE_MESSAGES_PER_ITEM',
                'SEARCH_DEADLINE_SECONDS', 'RERANK_MAX_PROMPT_BYTES',
                'RERANK_CONCURRENCY', 'RERANK_DEADLINE_SECONDS',
                'RERANK_REPAIR_MAX_CALLS',
                'CASCADE_COARSE_LIMIT', 'CASCADE_FINE_LIMIT', 'CASCADE_LLM_LIMIT',
                'CE_BATCH_SIZE', 'CE_TIMEOUT_SECONDS', 'CE_DEADLINE_SECONDS',
                'CE_MAX_DOCUMENT_BYTES', 'CE_MAX_REQUEST_BYTES', 'CE_API_FORMAT',
                'GRAPH_FUSION_ENABLED', 'GRAPH_FUSION_MAX_CANDIDATES',
                'SEARCH_FOLLOWUP_QUERIES', 'SEARCH_FOLLOWUP_SECONDS',
                'SEARCH_ITEM_MAX_BYTES', 'SEARCH_LOCAL_CONCURRENCY',
                'VECTOR_CHUNK_SIZE', 'VECTOR_CACHE_BYTES', 'VECTOR_APPROXIMATE',
                'VECTOR_EXACT_LIMIT', 'VECTOR_CANDIDATE_LIMIT',
                'SEARCH_MAX_CALLS', 'SEARCH_MAX_TOKENS',
                'EVIDENCE_FALLBACK_ITEMS',
                'RECALL_VECTOR_LIMIT', 'RECALL_SOURCE_LIMIT', 'RECALL_FTS_LIMIT',
                'RECALL_PROFILE_LIMIT', 'RECALL_EXPANSION_LIMIT',
                'RECALL_EXPANSION_MIN_DIRECT')}}
