#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

eval_chunk_messages="${AML_EVAL_CHUNK_MESSAGES:-20}"
eval_chunk_words="${AML_EVAL_CHUNK_WORDS:-2000}"
eval_convs="${AML_EVAL_CONVS:-1}"
eval_limit="${AML_EVAL_LIMIT:-30}"
eval_data="${AML_EVAL_DATA:-./data/prepared/personamem-v2-32k.jsonl}"
eval_result_name="${AML_EVAL_RESULT_NAME:-personamem-v2-32k.jsonl}"
if [[ "$eval_result_name" != *.jsonl || "$eval_result_name" == */* ]]; then
  echo 'AML_EVAL_RESULT_NAME must be a .jsonl basename' >&2
  exit 2
fi
https_url="${AML_HTTPS_URL:-https://localhost:8443}"
https_timeout="${AML_HTTPS_TIMEOUT:-600}"

# Authenticate with AML_API_KEY or AML_HTTPS_TOKEN_FILE. If AML_HTTPS_CA is
# unset, use the system trust store; set it for a private/self-signed CA.
https_args=(--url "$https_url" --timeout "$https_timeout")
if [[ -n "${AML_HTTPS_CA:-}" ]]; then
  https_args+=(--ca "$AML_HTTPS_CA")
fi
if [[ -n "${AML_HTTPS_TOKEN_FILE:-}" ]]; then
  https_args+=(--token-file "$AML_HTTPS_TOKEN_FILE")
fi

# Add/Search and reranking configuration belong to the HTTPS server.
# Answer/Judge uses the client's AML_LLM_* / provider credentials.
run_dir="${AML_EXPERIMENT_DIR:-runs/experiment-https-$(date -u +%Y%m%dT%H%M%S)-$$}"
mkdir -p -- "$run_dir"
run_dir="$(cd -- "$run_dir" && pwd)"
mkdir -- "$run_dir/completed"
mkdir -p -- "$run_dir/completed/logs" "$run_dir/completed/data/results"
export AML_MEMORY_DEBUG_LOG="$run_dir/completed/logs/memory-debug.jsonl"
export AML_SEARCH_DEBUG_LOG="$run_dir/completed/logs/search-debug.jsonl"
exec > >(tee "$run_dir/output.log") 2>&1
printf 'experiment_dir=%s\n' "$run_dir"
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-code"' EXIT

# Extra arguments, such as --inspect or --no-progress, are passed through.
python -u scripts/https_eval.py "${https_args[@]}" \
  --data "$eval_data" --convs "$eval_convs" --limit "$eval_limit" \
  --chunk-messages "$eval_chunk_messages" --chunk-words "$eval_chunk_words" \
  --output "$run_dir/completed/data/results/$eval_result_name" "$@"
