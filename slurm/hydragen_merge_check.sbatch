#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=hydragen_merge_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:20:00
#SBATCH --output=/home/u4320956/fast-draft/.claude/worktrees/hydragen-compare/slurm/logs/%x_%j.out
#SBATCH --error=/home/u4320956/fast-draft/.claude/worktrees/hydragen-compare/slurm/logs/%x_%j.err

set -uo pipefail
date
export CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/cuda-12.6
export GCC_HOME=/work/HPC_software/LMOD/gcc/12.5.0
export PATH=${GCC_HOME}/bin:${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${GCC_HOME}/lib64:${LD_LIBRARY_PATH:-}
export CC=${GCC_HOME}/bin/gcc
export CXX=${GCC_HOME}/bin/g++
export CUDAHOSTCXX=${GCC_HOME}/bin/g++

WT=/home/u4320956/fast-draft/.claude/worktrees/hydragen-compare
export PYTHONPATH=${WT}/3rdparty/sglang/python
cd /home/u4320956/fast-draft

uv run python ${WT}/tests/bench_hydragen_paged.py \
    --num-seqs 4 --topk 5 --step-offset 3 \
    --prefix-lens 16384 --num-qo-heads 32 --num-kv-heads 8 \
    --num-layers 16 --num-draft-steps 3
echo "exit: $?"
date
