#!/bin/bash
set -e

export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_VISIBLE_DEVICES=2,3

BASE="uv run python tests/bench_dataset.py \
  --model-path Qwen/Qwen3-4B \
  --draft-model-path brian920128/Qwen3-4B_eagle3 \
  --speculative-algorithm EAGLE3 \
  --eagle-topk 10 \
  --speculative-num-steps 4 \
  --max-new-tokens 512 \
  --context-length 110000 \
  --mem-fraction-static 0.40 \
  --tp 2 \
  --time-spec"

echo "=== RUN 1: gov_report (with original) ==="
eval $BASE --dataset-path data/gov_report.jsonl
echo "DONE"

echo "=== RUN 2: narrativeqa (with original) ==="
eval $BASE --dataset-path data/narrativeqa.jsonl
echo "DONE"

echo "=== RUN 3: pg19 (with original) ==="
eval $BASE --dataset-path data/pg19.jsonl
echo "DONE"

echo "ALL EAGLE3 BENCHMARKS COMPLETE"
