#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=llama8b_1b_standalone_bs4_50k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
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
echo "=== Run bench_e2e.py (Llama-3.1-8B target + Llama-3.2-1B draft, STANDALONE, prompt=50k, bs=4, topk=5) ==="
uv run python tests/bench_e2e.py \
    --model-path meta-llama/Llama-3.1-8B \
    --draft-model-path meta-llama/Llama-3.2-1B \
    --speculative-algorithm STANDALONE \
    --speculative-num-steps 4 \
    --eagle-topk 5 \
    --prompt-lengths 50000 \
    --max-new-tokens 256 \
    --num-requests 2 \
    --batch-size 2 \
    --context-length 51200 \
    --mem-fraction-static 0.60 \
    --tp 1 \
    --result-file results/llama8b_1b_standalone_bs4_50k_${SLURM_JOB_ID}.json

echo
echo "=== Done ==="
date
