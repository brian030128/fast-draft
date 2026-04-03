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
  --speculative-num-steps 7"

DATASETS=(data/gov_report.jsonl data/narrativeqa.jsonl data/pg19.jsonl)
TOPKS=(8 16)

for i in "${!DATASETS[@]}"; do
  for j in "${!TOPKS[@]}"; do
    ds="${DATASETS[$i]}"
    topk="${TOPKS[$j]}"
    SKIP=""
    # Only the very first run (i=0, j=0) includes original
    if [ "$i" -ne 0 ] || [ "$j" -ne 0 ]; then
      SKIP="--skip-original"
    fi
    echo "========================================"
    echo "RUN: dataset=$ds topk=$topk skip_original=$SKIP"
    echo "========================================"
    eval $BASE --dataset-path "$ds" --eagle-topk "$topk" $SKIP
    echo "DONE: dataset=$ds topk=$topk"
    echo ""
  done
done

echo "ALL BENCHMARKS COMPLETE"
