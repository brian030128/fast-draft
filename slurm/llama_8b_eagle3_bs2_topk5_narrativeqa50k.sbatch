#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=llama_8b_eagle3_bs2_topk5_narrativeqa50k
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
mkdir -p results

echo
echo "=== bench_dataset.py Llama-3.1-8B-Instruct + SpecForge EAGLE3 bs=2 topk=5 tree on narrativeqa 50k ==="
uv run python tests/bench_dataset.py \
    --dataset-path data/narrativeqa_chat_50k.jsonl \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-path lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge \
    --speculative-algorithm EAGLE3 \
    --speculative-num-steps 4 \
    --eagle-topk 5 \
    --max-new-tokens 256 \
    --batch-size 2 \
    --context-length 64000 \
    --mem-fraction-static 0.70 \
    --tp 1 \
    --skip cascade_no_cg \
    --skip flat_no_cg \
    --skip cascade_per_step \
    --skip cascade_per_step_no_cg \
    --skip fasttree \
    --result-prefix llama_8b_bs2_topk5_narrqa50k_${SLURM_JOB_ID}_

echo
echo "=== Done ==="
date
