"""Non-secret code/prompt fingerprints for reproducible local evaluations."""
import hashlib
from functools import lru_cache
from pathlib import Path
from . import answer_choice, config

SEARCH_POLICY = 'graph_cascade_v13-optional-recovery-span-refs'


@lru_cache(maxsize=1)
def versions():
    app = Path(__file__).resolve().parent
    prompt_dir = app.parents[1] / 'prompts'
    files = sorted(app.glob('*.py')) + sorted(prompt_dir.glob('*.txt'))
    hashes = {str(p.relative_to(app.parents[1])).replace('\\', '/'):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    fingerprint = hashlib.sha256(repr(sorted(hashes.items())).encode()).hexdigest()
    return {'pipeline_version': fingerprint, 'search_policy': SEARCH_POLICY,
            'answer_policy': answer_choice.VERSION, 'file_hashes': hashes,
            'model': config.LLM_MODEL, 'embedding_model': config.EMBED_MODEL,
            'cross_encoder_model': config.CE_MODEL,
            'settings': {key: getattr(config, key) for key in (
                'CORE_PROFILE_MAX_ITEMS', 'QUERY_PROFILE_DIGEST', 'EXTRACT_SPAN_REFS',
                'SEARCH_SOURCE_EXCERPT_CHARS', 'SEARCH_SOURCE_MESSAGES_PER_ITEM',
                'SEARCH_DEADLINE_SECONDS', 'RERANK_MAX_PROMPT_BYTES',
                'RERANK_CONCURRENCY', 'RERANK_DEADLINE_SECONDS',
                'RERANK_REPAIR_MAX_CALLS',
                'CASCADE_COARSE_LIMIT', 'CASCADE_FINE_LIMIT', 'CASCADE_LLM_LIMIT',
                'CHOICE_ALLOW_INFERRED', 'CHOICE_BOUNDED_SUPPORT_SPANS', 'CHOICE_TYPOGRAPHIC_QUOTES', 'CHOICE_REPAIR_CONTEXT', 'CHOICE_REPAIR_SPAN_REFS', 'CHOICE_ADAPTIVE_INTEREST_RETRY', 'CHOICE_ENTAILMENT_REVIEW', 'CHOICE_SEMANTIC_WITNESSES', 'LISTWISE_SPAN_REFS',
                'CE_BATCH_SIZE', 'CE_TIMEOUT_SECONDS', 'CE_DEADLINE_SECONDS',
                'CE_RETRY_BATCHES', 'CE_RETRY_BATCH_SIZE',
                'CE_MAX_DOCUMENT_BYTES', 'CE_MAX_REQUEST_BYTES', 'CE_API_FORMAT',
                'GRAPH_FUSION_ENABLED', 'GRAPH_FUSION_MAX_CANDIDATES',
                'SEARCH_FOLLOWUP_QUERIES', 'SEARCH_FOLLOWUP_SECONDS',
                'SEARCH_ITEM_MAX_BYTES', 'SEARCH_LOCAL_CONCURRENCY',
                'VECTOR_CHUNK_SIZE', 'VECTOR_CACHE_BYTES', 'VECTOR_APPROXIMATE',
                'VECTOR_EXACT_LIMIT', 'VECTOR_CANDIDATE_LIMIT',
                'SEARCH_MAX_CALLS', 'SEARCH_MAX_TOKENS',
                'EVIDENCE_FALLBACK_ITEMS',
                'RECALL_VECTOR_LIMIT', 'RECALL_SOURCE_LIMIT', 'RECALL_FTS_LIMIT',
                'SOURCE_RECALL_DIVERSITY',
                'RECALL_PROFILE_LIMIT', 'RECALL_EXPANSION_LIMIT',
                'RECALL_EXPANSION_MIN_DIRECT')}}
