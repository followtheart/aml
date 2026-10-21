#!/usr/bin/env bash
# Run an isolated HTTPS evaluation with the latest comparison settings.
# Usage: ./run_https_eval.sh [evaluator arguments, e.g. --no-progress]
# Overrides: AML_LISTWISE_STAGED=1 AML_EVAL_PORT=18443 ./run_https_eval.sh
set -euo pipefail

eval_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$eval_repo_dir/memory_system"

export AML_FAKE=0
export AML_LISTWISE_STAGED="${AML_LISTWISE_STAGED:-0}"
export AML_SEARCH_MAX_TOKENS="${AML_SEARCH_MAX_TOKENS:-128000}"

export AML_FINE_EVIDENCE_ADAPTIVE="${AML_FINE_EVIDENCE_ADAPTIVE:-1}"
export AML_FINE_EXTRA_LIMIT="${AML_FINE_EXTRA_LIMIT:-8}"
export AML_LISTWISE_SHARED_SOURCES="${AML_LISTWISE_SHARED_SOURCES:-1}"
export AML_LISTWISE_UNREVIEWED_LIMIT="${AML_LISTWISE_UNREVIEWED_LIMIT:-2}"
export AML_LISTWISE_KEYED_OUTPUT="${AML_LISTWISE_KEYED_OUTPUT:-1}"
export AML_RERANK_REPAIR_MAX_CALLS="${AML_RERANK_REPAIR_MAX_CALLS:-2}"
export AML_LISTWISE_SPAN_REFS="${AML_LISTWISE_SPAN_REFS:-1}"
export AML_LISTWISE_REVIEW_MAX_BATCHES="${AML_LISTWISE_REVIEW_MAX_BATCHES:-3}"
export AML_LISTWISE_PURPOSE_BATCH_SIZE="${AML_LISTWISE_PURPOSE_BATCH_SIZE:-10}"
export AML_LISTWISE_CONTEXT_REVIEW_LIMIT="${AML_LISTWISE_CONTEXT_REVIEW_LIMIT:-2}"
export AML_LISTWISE_STAGED_MAX_PROMPT_BYTES="${AML_LISTWISE_STAGED_MAX_PROMPT_BYTES:-48000}"
export AML_LISTWISE_CONTEXT_MAX_BYTES="${AML_LISTWISE_CONTEXT_MAX_BYTES:-12000}"

export AML_EVAL_DATA="${AML_EVAL_DATA:-$PWD/data/prepared/personamem-v2-32k.jsonl}"
export AML_EVAL_CONVS="${AML_EVAL_CONVS:-1}"
export AML_EVAL_LIMIT="${AML_EVAL_LIMIT:-30}"
export AML_EVAL_CHUNK_MESSAGES="${AML_EVAL_CHUNK_MESSAGES:-20}"
export AML_EVAL_CHUNK_WORDS="${AML_EVAL_CHUNK_WORDS:-2000}"

eval_run_dir="${AML_EXPERIMENT_DIR:-$PWD/runs/https-manual-$(date +%Y%m%d-%H%M%S)-$$}"
eval_port="${AML_EVAL_PORT:-18444}"

printf 'Evaluation directory: %s\n' "$eval_run_dir"
exec .venv/bin/python scripts/run_isolated_https_experiment.py \
  --run-dir "$eval_run_dir" \
  --port "$eval_port" \
  -- "$@"
