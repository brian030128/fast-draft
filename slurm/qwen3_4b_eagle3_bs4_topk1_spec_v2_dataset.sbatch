#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=qwen3_4b_eagle3_bs4_topk1_spec_v2_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
date

export CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/cuda-12.6
export GCC_HOME=/work/HPC_software/LMOD/gcc/12.5.0
export PATH=${GCC_HOME}/bin:${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${GCC_HOME}/lib64:${LD_LIBRARY_PATH:-}
export CC=${GCC_HOME}/bin/gcc
export CXX=${GCC_HOME}/bin/g++
export CUDAHOSTCXX=${GCC_HOME}/bin/g++
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ENABLE_SPEC_V2=True

cd /home/u4320956/fast-draft
mkdir -p results

echo
echo "=== bench_dataset.py Qwen3-4B EAGLE3 bs=4 topk=1 num_steps=5 spec_v2 on pg19 ==="
uv run python tests/bench_dataset.py \
    --dataset-path data/pg19_mid.jsonl \
    --model-path Qwen/Qwen3-4B \
    --draft-model-path brian920128/Qwen3-4B_eagle3 \
    --speculative-algorithm EAGLE3 \
    --speculative-num-steps 5 \
    --eagle-topk 1 \
    --max-new-tokens 256 \
    --batch-size 4 \
    --context-length 17500 \
    --mem-fraction-static 0.85 \
    --tp 1 \
    --skip cascade \
    --skip cascade_no_cg \
    --skip flat_no_cg \
    --skip cascade_per_step \
    --skip cascade_per_step_no_cg \
    --skip fasttree \
    --result-prefix qwen3_4b_bs4_topk1_specv2_pg19_${SLURM_JOB_ID}_

echo
echo "=== Done ==="
date
