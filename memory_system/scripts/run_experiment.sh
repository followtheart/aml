#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

eval_chunk_messages="${AML_EVAL_CHUNK_MESSAGES:-20}"
eval_chunk_words="${AML_EVAL_CHUNK_WORDS:-2000}"

rm -f -- logs/* data/results/* memory.db

python scripts/local_eval.py --data ./data/prepared/personamem-v2-32k.jsonl --convs 1 --limit 30 --chunk-messages "$eval_chunk_messages" --chunk-words "$eval_chunk_words" --output data/results/personamem-v2-32k.jsonl

# python scripts/local_eval.py --data ./data/prepared/personamem-v2-32k.jsonl --convs 2 --limit 50 --output data/results/personamem-v2-32k.jsonl
