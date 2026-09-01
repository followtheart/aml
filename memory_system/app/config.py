"""Deployment configuration. All knobs are environment variables.

AML_LLM_MODEL   : litellm model string for all LLM calls.
                  Default "gpt-4o-mini" (AML full-gate requirement in cycle 1).
                  Any litellm-supported model works. Examples:
                    OpenAI       gpt-4o-mini            (OPENAI_API_KEY)
                    SiliconFlow  siliconflow/Qwen/Qwen2.5-7B-Instruct
                                 (SILICONFLOW_API_KEY)
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
                  (or openai/<model> + AML_EMBED_API_BASE).
AML_EMBED_API_BASE / AML_EMBED_API_KEY : same override for embeddings.
AML_API_KEY     : Bearer token required on /add and /search. Empty = no auth
                  (only acceptable for local smoke).
AML_DB_PATH     : SQLite file path. Default ./memory.db
AML_FAKE        : "1" forces offline FakeLLM/FakeEmbedding (no network, for
                  plumbing tests and CI).
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


def _normalize_siliconflow(model, api_base, api_key):
    """litellm has no siliconflow provider in some versions; SiliconFlow is
    OpenAI-compatible, so rewrite `siliconflow/<m>` -> `openai/<m>` with the
    SiliconFlow base URL and SILICONFLOW_API_KEY."""
    if model and model.startswith("siliconflow/"):
        return ("openai/" + model[len("siliconflow/"):],
                api_base or "https://api.siliconflow.cn/v1",
                api_key or os.environ.get("SILICONFLOW_API_KEY") or None)
    return model, api_base, api_key


LLM_MODEL, LLM_API_BASE, LLM_API_KEY = _normalize_siliconflow(
    LLM_MODEL, LLM_API_BASE, LLM_API_KEY)
EMBED_MODEL, EMBED_API_BASE, EMBED_API_KEY = _normalize_siliconflow(
    EMBED_MODEL, EMBED_API_BASE, EMBED_API_KEY)
API_KEY = os.environ.get("AML_API_KEY", "")
DB_PATH = os.environ.get("AML_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "memory.db"))
FAKE = os.environ.get("AML_FAKE", "") == "1"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1200   # caps runaway repetition from small models
LLM_JSON_MAX_TOKENS = int(os.environ.get("AML_LLM_JSON_MAX_TOKENS", "2048"))
EXTRACT_BATCH_MESSAGES = max(
    1, int(os.environ.get("AML_EXTRACT_BATCH_MESSAGES", "6")))
EMBED_DIM = 256          # local/fake dim; litellm embeddings are truncated/padded to this
GOVERNANCE_NEIGHBORS = 5
RECALL_PER_ROUTE = 100
RERANK_CANDIDATES = 40
RERANK_SCORED = 30
