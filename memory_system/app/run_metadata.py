"""Non-secret code/prompt fingerprints for reproducible local evaluations."""
import hashlib
from functools import lru_cache
from pathlib import Path
from . import answer_choice, config, direct_catalog_reader

SEARCH_POLICY = 'graph_cascade_v16-staged-listwise'


@lru_cache(maxsize=1)
def versions():
    app = Path(__file__).resolve().parent
    prompt_dir = app.parents[1] / 'prompts'
    files = sorted(app.glob('*.py')) + sorted(prompt_dir.glob('*.txt'))
    hashes = {str(p.relative_to(app.parents[1])).replace('\\', '/'):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    fingerprint = hashlib.sha256(repr(sorted(hashes.items())).encode()).hexdigest()
    return {'pipeline_version': fingerprint, 'search_policy': SEARCH_POLICY,
            'answer_policy': (direct_catalog_reader.VERSION if config.CHOICE_DIRECT_CATALOG_READER else answer_choice.VERSION), 'file_hashes': hashes,
            'model': config.LLM_MODEL, 'embedding_model': config.EMBED_MODEL,
            'cross_encoder_model': config.CE_MODEL,
            'settings': {key: getattr(config, key) for key in (
                'CORE_PROFILE_MAX_ITEMS', 'PROFILE_REQUIRE_USER_SUPPORT', 'QUERY_PROFILE_DIGEST', 'CHOICE_QUERY_CONTEXT', 'CHOICE_PRESERVE_REPAIR_PREMISES', 'CHOICE_CURRENT_QUESTION_SOURCE', 'CHOICE_SOURCE_SCOPE', 'CHOICE_CONSTRAINT_CONTENT_TERMS', 'CHOICE_CONSTRAINT_TERM_FORMS', 'CHOICE_VERIFICATION_SCOPE', 'EXTRACT_SPAN_REFS', 'EXTRACT_TRIPLE_ISOLATION', 'EXTRACT_COVERAGE_RETRY',
                'SEARCH_SOURCE_EXCERPT_CHARS', 'SEARCH_SOURCE_MESSAGES_PER_ITEM',
                'SEARCH_DEADLINE_SECONDS', 'RERANK_MAX_PROMPT_BYTES', 'LISTWISE_SHARED_SOURCES', 'CHOICE_SCOPE_ROLES', 'CHOICE_GROUPED_ENTAILMENT', 'CHOICE_REVIEW_REQUEST_TERMS',
                'CHOICE_CONDITIONAL_LIST_SCOPE', 'CHOICE_TEMPLATE_PATH_SCOPE', 'CHOICE_EXISTING_STATE_GUARD', 'CHOICE_REPAIR_VALID_CLAIMS', 'CHOICE_CITED_SCOPE', 'RERANK_CONCURRENCY', 'RERANK_DEADLINE_SECONDS',
                'RERANK_REPAIR_MAX_CALLS', 'LISTWISE_KEYED_OUTPUT', 'LISTWISE_REVIEW_MAX_BATCHES',
                'LISTWISE_STAGED', 'LISTWISE_PURPOSE_BATCH_SIZE', 'LISTWISE_CONTEXT_REVIEW_LIMIT',
                'LISTWISE_STAGED_MAX_PROMPT_BYTES', 'LISTWISE_CONTEXT_MAX_BYTES',
                'CHOICE_RECOVERY_CLAIM_ONLY',
                'CHOICE_RECOVERY_BEFORE_SCOPE',
                'CASCADE_COARSE_LIMIT', 'CASCADE_FINE_LIMIT', 'CASCADE_LLM_LIMIT',
                'FINE_EVIDENCE_ADAPTIVE', 'FINE_EXTRA_LIMIT', 'LISTWISE_UNREVIEWED_LIMIT',
                'CHOICE_ALLOW_INFERRED', 'CHOICE_COVERAGE_WITNESSES', 'CHOICE_BOUNDED_SUPPORT_SPANS', 'CHOICE_TYPOGRAPHIC_QUOTES', 'CHOICE_REPAIR_CONTEXT', 'CHOICE_REPAIR_SPAN_REFS', 'CHOICE_ADAPTIVE_INTEREST_RETRY', 'CHOICE_ENTAILMENT_REVIEW', 'CHOICE_SEMANTIC_WITNESSES', 'CHOICE_WITNESS_SPAN_REFS', 'CHOICE_INDEPENDENT_SELECTION_REVIEW', 'CHOICE_TYPED_SCOPE_CONSISTENCY', 'CHOICE_DIRECT_REQUEST_SELECTION', 'CHOICE_STRUCTURED_REQUEST_SELECTION', 'CHOICE_SOURCE_NOTES_SELECTION', 'CHOICE_DIRECT_CATALOG_READER', 'LISTWISE_SPAN_REFS',
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
