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
  --disable-cuda-graph \
  --result-prefix wo_graph_"

echo "=== RUN 1: narrativeqa topk=8 (with original) ==="
eval $BASE --dataset-path data/narrativeqa.jsonl --eagle-topk 8
echo "DONE"

echo "=== RUN 2: narrativeqa topk=16 (skip-original) ==="
eval $BASE --dataset-path data/narrativeqa.jsonl --eagle-topk 16 --skip-original
echo "DONE"

echo "=== RUN 3: pg19 topk=8 (with original) ==="
eval $BASE --dataset-path data/pg19.jsonl --eagle-topk 8
echo "DONE"

echo "=== RUN 4: pg19 topk=16 (skip-original) ==="
eval $BASE --dataset-path data/pg19.jsonl --eagle-topk 16 --skip-original
echo "DONE"

echo "ALL EAGLE NO-GRAPH BENCHMARKS COMPLETE"
