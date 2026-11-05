#!/bin/bash
# End-to-end runs behind the A6000 x 2 rows of Table 1, the A6000 rows of Table 2,
# and Figure 4. Set CUDA_VISIBLE_DEVICES to two idle GPUs. CSVs go to results/.
#
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_paper_e2e.sh [table1|batch|all]
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

COMMON="--max-new-tokens 512 --context-length 110000 --mem-fraction-static 0.40 --tp 2
  --time-spec --result-prefix A6000_ --result-dir results"
LLAMA="--model-path meta-llama/Llama-3.1-8B --draft-model-path meta-llama/Llama-3.2-1B
  --eagle-topk 10 --speculative-num-steps 7 --speculative-num-draft-tokens 16"
QWEN="--model-path Qwen/Qwen3-4B --draft-model-path brian920128/Qwen3-4B_eagle3
  --speculative-algorithm EAGLE3 --eagle-topk 10 --speculative-num-steps 5
  --speculative-num-draft-tokens 16"

run() { uv run python tests/bench_dataset.py $COMMON "$@"; }

table1() {
  for ds in gov_report narrativeqa pg19; do
    for phase in original cascade flat fasttree hydragen deft; do
      run $LLAMA --dataset-path "data/$ds.jsonl" --only "$phase"
    done
    for phase in original cascade flat fasttree; do
      run $QWEN --dataset-path "data/$ds.jsonl" --only "$phase"
    done
  done
}

# Table 2 (A6000, B=4) and the Figure 4 ablation
batch() {
  for phase in original flat cascade cascade_per_step cascade_per_step_no_cg; do
    run $LLAMA --dataset-path data/narrativeqa.jsonl --batch-size 4 --only "$phase"
  done
  uv run --with matplotlib python scripts/plot_ablation.py --result-dir results --out ablation_study.svg
}

case "${1:-all}" in
  table1) table1 ;;
  batch) batch ;;
  all) table1; batch ;;
  *) echo "usage: $0 [table1|batch|all]"; exit 1 ;;
esac
