#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

eval_chunk_messages="${AML_EVAL_CHUNK_MESSAGES:-20}"
eval_chunk_words="${AML_EVAL_CHUNK_WORDS:-2000}"

# Use the request bound validated against singleton scores for this provider.
# Keep configured mode available for baseline/custom-provider comparisons.
# A singleton run needs enough calls for the 50-document coarse pool plus the
# remaining search stages. Explicit caller budgets still take precedence.
rerank_mode="${AML_EXPERIMENT_RERANK_MODE:-bounded}"
case "$rerank_mode" in
  configured) ;;
  bounded)
    export AML_CE_BATCH_SIZE="${AML_CE_BATCH_SIZE:-24}"
    export AML_CE_MAX_REQUEST_BYTES="${AML_CE_MAX_REQUEST_BYTES:-16000}"
    export AML_SEARCH_MAX_CALLS="${AML_SEARCH_MAX_CALLS:-20}"
    ;;
  singleton)
    export AML_CE_BATCH_SIZE=1
    export AML_SEARCH_MAX_CALLS="${AML_SEARCH_MAX_CALLS:-80}"
    ;;
  *)
    printf 'Unknown AML_EXPERIMENT_RERANK_MODE: %s (use bounded, configured or singleton)\n' "$rerank_mode" >&2
    exit 2
    ;;
esac

# Keep evaluation artifacts separate from the running API and previous runs.
# local_eval already owns a temporary SQLite database.
run_dir="${AML_EXPERIMENT_DIR:-runs/experiment-$(date -u +%Y%m%dT%H%M%S)-$$}"
mkdir -p -- "$run_dir"
run_dir="$(cd -- "$run_dir" && pwd)"
mkdir -- "$run_dir/completed"
mkdir -p -- "$run_dir/completed/logs" "$run_dir/completed/data/results"
export AML_MEMORY_DEBUG_LOG="$run_dir/completed/logs/memory-debug.jsonl"
export AML_SEARCH_DEBUG_LOG="$run_dir/completed/logs/search-debug.jsonl"
exec > >(tee "$run_dir/output.log") 2>&1
printf 'experiment_dir=%s\n' "$run_dir"
printf 'rerank_mode=%s\n' "$rerank_mode"
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-code"' EXIT

python -u scripts/local_eval.py --data ./data/prepared/personamem-v2-32k.jsonl --convs 1 --limit 30 --chunk-messages "$eval_chunk_messages" --chunk-words "$eval_chunk_words" --output "$run_dir/completed/data/results/personamem-v2-32k.jsonl"

# python scripts/local_eval.py --data ./data/prepared/personamem-v2-32k.jsonl --convs 2 --limit 50 --output data/results/personamem-v2-32k.jsonl
