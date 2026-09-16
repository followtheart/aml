"""Deployment configuration. All knobs are environment variables.

AML_LLM_MODEL   : litellm model string for all LLM calls.
                  Default "gpt-4o-mini" (AML full-gate requirement in cycle 1).
                  Any litellm-supported model works. Examples:
                    OpenAI       gpt-4o-mini            (OPENAI_API_KEY)
                    SiliconFlow  siliconflow/Qwen/Qwen2.5-7B-Instruct
                                 (SILICONFLOW_API_KEY)
                    DashScope    dashscope/qwen3-14b   (DASHSCOPE_API_KEY,
                                 Alibaba Cloud Bailian OpenAI-compatible mode)
                    DeepSeek     deepseek/deepseek-chat (DEEPSEEK_API_KEY)
                    Ollama       ollama/qwen2.5
                    Any OpenAI-compatible endpoint:
                      AML_LLM_MODEL=openai/<model> + AML_LLM_API_BASE=<url>
AML_LLM_API_BASE / AML_LLM_API_KEY : optional custom endpoint / key override
                  for the chat model (OpenAI-compatible APIs).
AML_LLM_JSON_MAX_TOKENS : output cap for structured JSON calls. Default 2048.
AML_EXTRACT_BATCH_MESSAGES : maximum messages per extraction request.
                  Default 6 to keep small-model JSON generation reliable.
AML_EMBED_MODEL : litellm embedding model. Default "text-embedding-3-small".
                  SiliconFlow example: siliconflow/BAAI/bge-m3
                  DashScope example:   dashscope/text-embedding-v4
                  (or openai/<model> + AML_EMBED_API_BASE).
AML_EMBED_API_BASE / AML_EMBED_API_KEY : same override for embeddings.
AML_EMBED_DIM / AML_EMBEDDING_SPACE : stored vector dimension and explicit
                  compatibility-space label. Changing either requires re-embedding.
AML_API_KEY     : Bearer token required on /add and /search. Empty = no auth
                  (only acceptable for local smoke).
AML_DB_PATH     : SQLite file path. Default ./memory.db
AML_FAKE        : "1" forces offline FakeLLM/FakeEmbedding (no network, for
                  plumbing tests and CI).
AML_RERANK_MAX_CANDIDATES / AML_SEARCH_MIN_RELEVANCE : bounded rerank cost
                  and the minimum accepted relevance score.
"""
import os
from pathlib import Path


def _load_dotenv():
    """Tiny .env loader (no dependency). Does not override existing env."""
    for cand in (Path(__file__).resolve().parents[1] / ".env",
                 Path.cwd() / ".env"):
        if not cand.exists():
            continue
        for line in cand.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

LLM_MODEL = os.environ.get("AML_LLM_MODEL", "gpt-4o-mini")
LLM_API_BASE = os.environ.get("AML_LLM_API_BASE", "") or None
LLM_API_KEY = os.environ.get("AML_LLM_API_KEY", "") or None
EMBED_MODEL = os.environ.get("AML_EMBED_MODEL", "text-embedding-3-small")
EMBED_API_BASE = os.environ.get("AML_EMBED_API_BASE", "") or None
EMBED_API_KEY = os.environ.get("AML_EMBED_API_KEY", "") or None


# OpenAI-compatible providers that litellm may not know natively. Each prefix
# maps to its base URL, the environment variable holding the API key, the
# provider's hard ceiling on max_tokens and on embedding inputs per request
# (None when unknown/unbounded).
_COMPATIBLE_PROVIDERS = {
    "siliconflow/": {"base": "https://api.siliconflow.cn/v1",
                     "key_env": "SILICONFLOW_API_KEY", "max_output_tokens": None,
                     "embed_batch": None},
    "dashscope/": {"base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                   "key_env": "DASHSCOPE_API_KEY", "max_output_tokens": 8192,
                   "embed_batch": 10},
}


def _normalize_compatible(model, api_base, api_key):
    """Rewrite `<provider>/<m>` -> `openai/<m>` with the provider's base URL
    and API key so litellm talks to it through its OpenAI client. Also returns
    the provider spec (limits) or None."""
    for prefix, spec in _COMPATIBLE_PROVIDERS.items():
        if model and model.startswith(prefix):
            return ("openai/" + model[len(prefix):],
                    api_base or spec["base"],
                    api_key or os.environ.get(spec["key_env"]) or None,
                    spec)
    return model, api_base, api_key, None


LLM_MODEL, LLM_API_BASE, LLM_API_KEY, _LLM_SPEC = _normalize_compatible(
    LLM_MODEL, LLM_API_BASE, LLM_API_KEY)
EMBED_MODEL, EMBED_API_BASE, EMBED_API_KEY, _EMBED_SPEC = _normalize_compatible(
    EMBED_MODEL, EMBED_API_BASE, EMBED_API_KEY)
_LLM_OUTPUT_CAP = _LLM_SPEC["max_output_tokens"] if _LLM_SPEC else None
# Inputs per embedding request; larger lists are split into ordered batches.
EMBED_BATCH_SIZE = max(1, int(os.environ.get(
    "AML_EMBED_BATCH_SIZE",
    str((_EMBED_SPEC or {}).get("embed_batch") or 2048))))
API_KEY = os.environ.get("AML_API_KEY", "")
DB_PATH = os.environ.get("AML_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "memory.db"))
FAKE = os.environ.get("AML_FAKE", "") == "1"
LLM_TEMPERATURE = 0.0
# Additional retries for HTTP 429, independent of parsing/other error retries.
RATE_LIMIT_RETRIES = max(0, int(os.environ.get("AML_RATE_LIMIT_RETRIES", "6")))
RATE_LIMIT_BACKOFF_SECONDS = max(
    1.0, float(os.environ.get("AML_RATE_LIMIT_BACKOFF_SECONDS", "15")))
RATE_LIMIT_MAX_BACKOFF_SECONDS = max(
    RATE_LIMIT_BACKOFF_SECONDS,
    float(os.environ.get("AML_RATE_LIMIT_MAX_BACKOFF_SECONDS", "120")))
# Local soft thresholds; set to the account's limits with suitable headroom.
PROVIDER_SOFT_TPM = max(0, int(os.environ.get("AML_PROVIDER_SOFT_TPM", "60000")))
PROVIDER_RPM = max(0, int(os.environ.get("AML_PROVIDER_RPM", "30")))
# In-flight calls allowed per (kind, model); the RPM/TPM window still paces them.
PROVIDER_CONCURRENCY = max(1, int(os.environ.get("AML_PROVIDER_CONCURRENCY", "4")))
# Qwen3 chat models reason by default on OpenAI-compatible endpoints; the
# hidden reasoning tokens multiply latency for structured extraction, and
# DashScope rejects non-streaming requests while thinking is enabled.
LLM_DISABLE_THINKING = os.environ.get("AML_LLM_DISABLE_THINKING", "1") == "1"
LLM_MAX_TOKENS = 1200   # caps runaway repetition from small models
LLM_JSON_MAX_TOKENS = int(os.environ.get("AML_LLM_JSON_MAX_TOKENS", "2048"))
# Providers reject max_tokens above their ceiling with HTTP 400 instead of
# clamping; clamp locally so a generous default does not break every call.
if _LLM_OUTPUT_CAP:
    LLM_JSON_MAX_TOKENS = min(LLM_JSON_MAX_TOKENS, _LLM_OUTPUT_CAP)
    LLM_MAX_TOKENS = min(LLM_MAX_TOKENS, _LLM_OUTPUT_CAP)
EXTRACT_BATCH_MESSAGES = max(
    1, int(os.environ.get("AML_EXTRACT_BATCH_MESSAGES", "6")))
EMBED_DIM = max(32, int(os.environ.get("AML_EMBED_DIM", "256")))
# Stored with every vector so incompatible providers/fallbacks are never mixed.
EMBEDDING_SPACE = os.environ.get(
    "AML_EMBEDDING_SPACE",
    f"{'fake-hash' if FAKE else EMBED_MODEL}:float32:{EMBED_DIM}")
# Sensitive memories are excluded from every retrieval/index route unless both
# the deployment policy and an individual request explicitly opt in.
SENSITIVE_RECALL_ENABLED = os.environ.get("AML_SENSITIVE_RECALL_ENABLED", "0") == "1"
GOVERNANCE_NEIGHBORS = 5
RECALL_PER_ROUTE = 100
RERANK_CANDIDATES = 40
RERANK_SCORED = 30
# ULM §5.4: score only a small head of the fused list; the rest keeps its
# fusion order below the scored items instead of being penalised.
RERANK_MAX_CANDIDATES = max(
    10, int(os.environ.get("AML_RERANK_MAX_CANDIDATES", "40")))
SEARCH_MIN_RELEVANCE = float(os.environ.get("AML_SEARCH_MIN_RELEVANCE", "0.3"))

# ---- ULM lifecycle knobs (llm-memory-survey/memory-system-design.md) ----
# §3.2 semantic boundary segmentation (embedding drop between adjacent windows)
SEGMENT_MIN_MESSAGES = max(1, int(os.environ.get("AML_SEGMENT_MIN_MESSAGES", "2")))
SEGMENT_SIM_DROP = float(os.environ.get("AML_SEGMENT_SIM_DROP", "0.35"))
# §2.1 keep the raw chunk of every segment as an `episode` memory (Memory-Doc)
STORE_EPISODES = os.environ.get("AML_STORE_EPISODES", "1") == "1"
# §3.4 novelty gate: near-identical facts skip the governance LLM call
NOVELTY_DUP_THRESHOLD = float(os.environ.get("AML_NOVELTY_DUP_THRESHOLD", "0.97"))
SURPRISE_MOMENTUM = 0.7
# §2.2/§4.1 MemScene clustering (cosine + keyword Jaccard)
SCENE_JOIN_THRESHOLD = float(os.environ.get("AML_SCENE_JOIN_THRESHOLD", "0.55"))
SCENE_TOP_M = max(1, int(os.environ.get("AML_SCENE_TOP_M", "5")))
SCENE_CELLS_PER_SCENE = max(1, int(os.environ.get("AML_SCENE_CELLS_PER_SCENE", "20")))
SCENE_SUMMARY_LLM = os.environ.get("AML_SCENE_SUMMARY_LLM", "0") == "1"
# §4.4 heat = a*visits + b*interactions + c*recency + d*surprise
HEAT_WEIGHTS = (1.0, 0.5, 2.0, 1.0)
HEAT_RECENCY_HALFLIFE_DAYS = 14.0
HEAT_PROMOTE_THRESHOLD = float(os.environ.get("AML_HEAT_PROMOTE_THRESHOLD", "5"))
MAX_HOT_SCENES = max(1, int(os.environ.get("AML_MAX_HOT_SCENES", "500")))
# §4.5 Ebbinghaus retention R=exp(-t/S); 0 disables cold-tiering
FORGET_THRESHOLD = float(os.environ.get("AML_FORGET_THRESHOLD", "0"))
FORGET_STRENGTH_DAYS = 30.0
FORGET_RECALL_BONUS_DAYS = 15.0
# §4.2 profile: traits supported by >= N sessions count as stable
PROFILE_STABLE_SESSIONS = max(1, int(os.environ.get("AML_PROFILE_STABLE_SESSIONS", "2")))
PROFILE_TRANSIENT_TTL_DAYS = max(
    1, int(os.environ.get("AML_PROFILE_TRANSIENT_TTL_DAYS", "90")))
# §5.5 sufficiency verifier + iterative retrieval + abstention
SEARCH_MAX_ROUNDS = max(1, int(os.environ.get("AML_SEARCH_MAX_ROUNDS", "2")))
VERIFY_EVIDENCE_ITEMS = 15
ABSTAIN_CONFIDENCE = float(os.environ.get("AML_ABSTAIN_CONFIDENCE", "0.15"))
# §3.1 the rolling session summary is extraction context only (ACE collapse);
# the legacy summary recall route stays available for ablation.
SUMMARY_ROUTE = os.environ.get("AML_SUMMARY_ROUTE", "0") == "1"

# ---- Persona layer (IMPROVEMENT_PLAN P0-P3; MemoryBank/MIRIX/EverMemOS) ----
# P0: interest consolidation after each Add promotes repeated "user asked
# about X" signals into first-person `preference` AMUs.
PROFILE_CONSOLIDATION_ENABLED = os.environ.get("AML_PROFILE_CONSOLIDATION", "1") == "1"
# Support keys (Add request ids / supporting AMU ids) needed to confirm a
# consolidated preference and to promote it transient -> static.
PROFILE_MIN_SUPPORT = max(1, int(os.environ.get("AML_PROFILE_MIN_SUPPORT", "2")))
# P0: Core Profile is injected ahead of ranked results on every search,
# outside top_k (MemGPT/MIRIX core memory). 0 disables for ablation.
CORE_PROFILE_INJECT = os.environ.get("AML_CORE_PROFILE_INJECT", "1") == "1"
CORE_PROFILE_MAX_ITEMS = max(1, int(os.environ.get("AML_CORE_PROFILE_MAX_ITEMS", "12")))
CORE_PROFILE_MAX_CHARS = max(256, int(os.environ.get("AML_CORE_PROFILE_MAX_CHARS", "1600")))
# P1: a user "forget X" request invalidates the matching memories
# (Zep edge invalidation) instead of only logging the request.
INVALIDATE_ENABLED = os.environ.get("AML_INVALIDATE_ENABLED", "1") == "1"
INVALIDATE_MIN_SIM = float(os.environ.get("AML_INVALIDATE_MIN_SIM", "0.55"))
INVALIDATE_MAX_TARGETS = max(1, int(os.environ.get("AML_INVALIDATE_MAX_TARGETS", "3")))
# P1: histories without timestamps get synthetic monotonic ones seeded per
# session, so validity ordering follows revelation order instead of the
# service wall clock (Zep dual timeline; PersonaMem carries no timestamps).
SYNTHETIC_TIME_ENABLED = os.environ.get("AML_SYNTHETIC_TIME", "1") == "1"
SYNTHETIC_EPOCH_MS = int(os.environ.get("AML_SYNTHETIC_EPOCH_MS", "1577836800000"))  # 2020-01-01
SYNTHETIC_STEP_MS = max(1, int(os.environ.get("AML_SYNTHETIC_STEP_MS", "60000")))
# P2: choice questions get an explicit option<->persona-evidence alignment
# pass before answering (Memory-R1 answer-agent distillation).
CHOICE_ALIGN_ENABLED = os.environ.get("AML_CHOICE_ALIGN", "1") == "1"
# P3: query understanding sees a compact profile digest so expansions bind
# generic questions to known user traits.
QUERY_PROFILE_DIGEST = os.environ.get("AML_QUERY_PROFILE_DIGEST", "1") == "1"
# P3: preference/profile intents hide assistant world knowledge once the user
# actually owns persona memories (Structural Memory: per-task memory views).
PERSONA_VIEW_FILTER = os.environ.get("AML_PERSONA_VIEW_FILTER", "1") == "1"

# Full stored content for local debugging; empty path disables the JSONL log.
MEMORY_DEBUG_LOG = os.environ.get(
    "AML_MEMORY_DEBUG_LOG", "")

SEARCH_DEBUG_LOG = os.environ.get(
    "AML_SEARCH_DEBUG_LOG", "")

# Source evidence returned to the answer model. Full evidence remains in SQLite
# and the debug log; these limits prevent top_k=100 from producing megabytes.
SEARCH_SOURCE_MESSAGES_PER_ITEM = max(
    0, int(os.environ.get("AML_SEARCH_SOURCE_MESSAGES_PER_ITEM", "3")))
SEARCH_SOURCE_REFS_PER_ITEM = max(
    SEARCH_SOURCE_MESSAGES_PER_ITEM,
    int(os.environ.get("AML_SEARCH_SOURCE_REFS_PER_ITEM", "20")))
SEARCH_SOURCE_CONTEXT_CHARS = max(
    0, int(os.environ.get("AML_SEARCH_SOURCE_CONTEXT_CHARS", "12000")))

# Answer memory budget includes all memory types and their rendered evidence.
# These are character limits, not estimates of a provider-specific token count.
ANSWER_CONTEXT_MAX_CHARS = max(
    256, int(os.environ.get("AML_ANSWER_CONTEXT_MAX_CHARS", "24000")))
ANSWER_CONTEXT_ITEM_MAX_CHARS = max(
    256, int(os.environ.get("AML_ANSWER_CONTEXT_ITEM_MAX_CHARS", "2400")))
