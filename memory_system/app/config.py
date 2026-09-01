"""Deployment configuration. All knobs are environment variables.

AML_LLM_MODEL   : litellm model string for all LLM calls.
                  Default "gpt-4o-mini" (AML full-gate requirement in cycle 1).
                  Any litellm-supported model works, e.g. "openai/gpt-4o-mini",
                  "azure/...", "deepseek/deepseek-chat", "ollama/qwen2.5".
AML_EMBED_MODEL : litellm embedding model. Default "text-embedding-3-small".
AML_API_KEY     : Bearer token required on /add and /search. Empty = no auth
                  (only acceptable for local smoke).
AML_DB_PATH     : SQLite file path. Default ./memory.db
AML_FAKE        : "1" forces offline FakeLLM/FakeEmbedding (no network, for
                  plumbing tests and CI).
"""
import os

LLM_MODEL = os.environ.get("AML_LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("AML_EMBED_MODEL", "text-embedding-3-small")
API_KEY = os.environ.get("AML_API_KEY", "")
DB_PATH = os.environ.get("AML_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "memory.db"))
FAKE = os.environ.get("AML_FAKE", "") == "1"
LLM_TEMPERATURE = 0.0
EMBED_DIM = 256          # local/fake dim; litellm embeddings are truncated/padded to this
GOVERNANCE_NEIGHBORS = 5
RECALL_PER_ROUTE = 100
RERANK_CANDIDATES = 40
RERANK_SCORED = 30
