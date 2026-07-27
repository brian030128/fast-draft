#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=qwen3_4b_eagle3_bs64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-unset}"
date

echo
echo "=== nvidia-smi ==="
nvidia-smi

echo
echo "=== Environment setup ==="
export CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/cuda-12.6
export GCC_HOME=/work/HPC_software/LMOD/gcc/12.5.0
export PATH=${GCC_HOME}/bin:${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${GCC_HOME}/lib64:${LD_LIBRARY_PATH:-}
export CC=${GCC_HOME}/bin/gcc
export CXX=${GCC_HOME}/bin/g++
export CUDAHOSTCXX=${GCC_HOME}/bin/g++
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

cd /home/u4320956/fast-draft
echo "CUDA: $(nvcc --version | tail -1)"
echo "gcc:  $(gcc --version | head -1)"
echo "uv:   $(uv --version)"

echo
echo "=== Run bench_e2e.py (Qwen3-4B EAGLE3, batch=64, num_requests=64) ==="
uv run python tests/bench_e2e.py \
    --model-path Qwen/Qwen3-4B \
    --draft-model-path brian920128/Qwen3-4B_eagle3 \
    --speculative-algorithm EAGLE3 \
    --speculative-num-steps 4 \
    --eagle-topk 10 \
    --prompt-lengths 1024 \
    --max-new-tokens 256 \
    --num-requests 64 \
    --batch-size 64 \
    --context-length 2048 \
    --mem-fraction-static 0.85 \
    --tp 1 \
    --result-file results/qwen3_4b_eagle3_bs64_${SLURM_JOB_ID}.json

echo
echo "=== Done ==="
date
