#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=hydragen_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=/home/u4320956/fast-draft/.claude/worktrees/hydragen-compare/slurm/logs/%x_%j.out
#SBATCH --error=/home/u4320956/fast-draft/.claude/worktrees/hydragen-compare/slurm/logs/%x_%j.err

set -uo pipefail

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
export HF_HUB_OFFLINE=1

WT=/home/u4320956/fast-draft/.claude/worktrees/hydragen-compare
# Run from the main checkout so uv uses the existing .venv, but resolve
# `sglang` from the worktree's submodule (meta-path editable finders sit
# after PathFinder, so PYTHONPATH wins).
export PYTHONPATH=${WT}/3rdparty/sglang/python
cd /home/u4320956/fast-draft

echo
echo "=== sanity: which sglang ==="
uv run python -c "import sglang; print(sglang.__file__)"

echo
echo "=== 1/2 kernel microbenchmark: merge equivalence + gather cost + attention ==="
uv run python ${WT}/tests/bench_hydragen_paged.py \
    --num-seqs 4 --topk 5 --step-offset 3 \
    --prefix-lens 4096,16384,50000 \
    --num-qo-heads 32 --num-kv-heads 8 \
    --num-layers 16 --num-draft-steps 3
echo "microbench exit: $?"

echo
echo "=== 2/2 E2E smoke: flat vs hydragen vs cascade (Llama-3.1-8B + 1B, narrativeqa 50k) ==="
uv run python ${WT}/tests/bench_dataset.py \
    --dataset-path /home/u4320956/fast-draft/data/narrativeqa_chat_50k.jsonl \
    --model-path meta-llama/Llama-3.1-8B \
    --draft-model-path meta-llama/Llama-3.2-1B \
    --speculative-algorithm STANDALONE \
    --speculative-num-steps 4 \
    --eagle-topk 5 \
    --max-new-tokens 128 \
    --num-samples 8 \
    --batch-size 2 \
    --context-length 64000 \
    --mem-fraction-static 0.60 \
    --tp 1 \
    --skip original \
    --skip flat_no_cg \
    --skip cascade_no_cg \
    --skip cascade_per_step \
    --skip cascade_per_step_no_cg \
    --skip fasttree \
    --result-dir ${WT}/results \
    --result-prefix hydragen_smoke_${SLURM_JOB_ID}_
echo "e2e exit: $?"

echo
echo "=== Done ==="
date
