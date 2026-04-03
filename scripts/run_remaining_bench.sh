#!/bin/bash
set -e

export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_VISIBLE_DEVICES=2,3

BASE="uv run python tests/bench_dataset.py \
  --model-path meta-llama/Llama-3.1-8B \
  --draft-model-path meta-llama/Llama-3.2-1B \
  --max-new-tokens 512 \
  --context-length 110000 \
  --mem-fraction-static 0.40 \
  --tp 2 \
  --time-spec \
  --speculative-num-draft-tokens 16 \
  --speculative-num-steps 7 \
  --skip-original"

echo "=== RUN 2: gov_report topk=16 (skip-original) ==="
eval $BASE --dataset-path data/gov_report.jsonl --eagle-topk 16
echo "DONE"

echo "=== RUN 3: narrativeqa topk=8 (skip-original) ==="
eval $BASE --dataset-path data/narrativeqa.jsonl --eagle-topk 8
echo "DONE"

echo "=== RUN 4: narrativeqa topk=16 (skip-original) ==="
eval $BASE --dataset-path data/narrativeqa.jsonl --eagle-topk 16
echo "DONE"

echo "=== RUN 5: pg19 topk=8 (skip-original) ==="
eval $BASE --dataset-path data/pg19.jsonl --eagle-topk 8
echo "DONE"

echo "=== RUN 6: pg19 topk=16 (skip-original) ==="
eval $BASE --dataset-path data/pg19.jsonl --eagle-topk 16
echo "DONE"

echo "ALL REMAINING BENCHMARKS COMPLETE"
