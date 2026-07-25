#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=qwen3_4b_eagle3_bs1_topk5
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

cd /home/u4320956/fast-draft

echo
echo "=== Run bench_e2e.py (Qwen3-4B EAGLE3, bs=1, topk=5, num_steps=4, tree spec) ==="
uv run python tests/bench_e2e.py \
    --model-path Qwen/Qwen3-4B \
    --draft-model-path brian920128/Qwen3-4B_eagle3 \
    --speculative-algorithm EAGLE3 \
    --speculative-num-steps 4 \
    --eagle-topk 5 \
    --prompt-lengths 1024 \
    --max-new-tokens 256 \
    --num-requests 1 \
    --batch-size 1 \
    --context-length 2048 \
    --mem-fraction-static 0.85 \
    --tp 1 \
    --skip cascade_no_cg \
    --result-file results/qwen3_4b_eagle3_bs1_topk5_${SLURM_JOB_ID}.json

echo
echo "=== Done ==="
date
