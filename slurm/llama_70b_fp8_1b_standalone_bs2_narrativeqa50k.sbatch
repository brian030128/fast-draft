#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=llama_70b_fp8_1b_standalone_bs2_narrativeqa50k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-unset}"
date

export CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/cuda-12.6
export GCC_HOME=/work/HPC_software/LMOD/gcc/12.5.0
export PATH=${GCC_HOME}/bin:${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${GCC_HOME}/lib64:${LD_LIBRARY_PATH:-}
export CC=${GCC_HOME}/bin/gcc
export CXX=${GCC_HOME}/bin/g++
export CUDAHOSTCXX=${GCC_HOME}/bin/g++
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# Redirect HF cache to worker-local /tmp (28 TB SSD) so 70B download
# doesn't fill /home (24 GB free quota).
export HF_HOME=/tmp/${USER}/hf_cache
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
mkdir -p "$HF_HOME"
# Pull HF auth token from /home cache so gated 70B download works.
if [ -f /home/${USER}/.cache/huggingface/token ]; then
    export HF_TOKEN=$(cat /home/${USER}/.cache/huggingface/token)
fi
echo "HF_HOME=$HF_HOME, HF_TOKEN ${HF_TOKEN:+set}"
df -h /tmp | tail -1

cd /home/u4320956/fast-draft
mkdir -p results

echo
echo "=== bench_dataset.py Llama-3.3-70B-Instruct (fp8) + Llama-3.2-1B-Instruct STANDALONE tp=2 bs=2 topk=5 steps=6 dt=10 on narrativeqa 50k ==="
uv run python tests/bench_dataset.py \
    --dataset-path data/narrativeqa_chat_50k.jsonl \
    --model-path meta-llama/Llama-3.3-70B-Instruct \
    --draft-model-path meta-llama/Llama-3.2-1B-Instruct \
    --quantization fp8 \
    --speculative-algorithm STANDALONE \
    --speculative-num-steps 6 \
    --speculative-num-draft-tokens 10 \
    --eagle-topk 5 \
    --max-new-tokens 256 \
    --batch-size 2 \
    --context-length 64000 \
    --mem-fraction-static 0.80 \
    --tp 2 \
    --skip cascade_no_cg \
    --skip flat_no_cg \
    --skip cascade_per_step \
    --skip cascade_per_step_no_cg \
    --skip fasttree \
    --result-prefix llama_70b_fp8_1b_standalone_bs2_topk5_steps6_dt10_narrqa50k_${SLURM_JOB_ID}_

echo
echo "=== Done ==="
date
