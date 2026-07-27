#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=llama_70b_fp8_1b_bs4_topk10_narrqa200_cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail
echo "=== Job info ==="; echo "Job ID:  $SLURM_JOB_ID"; echo "Node:    $(hostname)"; echo "GPUs:    ${CUDA_VISIBLE_DEVICES:-unset}"; date

export CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/cuda-12.6
export GCC_HOME=/work/HPC_software/LMOD/gcc/12.5.0
export PATH=${GCC_HOME}/bin:${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${GCC_HOME}/lib64:${LD_LIBRARY_PATH:-}
export CC=${GCC_HOME}/bin/gcc; export CXX=${GCC_HOME}/bin/g++; export CUDAHOSTCXX=${GCC_HOME}/bin/g++
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export HF_HOME=/tmp/${USER}/hf_cache; export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub; mkdir -p "$HF_HOME"
[ -f /home/${USER}/.cache/huggingface/token ] && export HF_TOKEN=$(cat /home/${USER}/.cache/huggingface/token)
echo "HF_HOME=$HF_HOME, HF_TOKEN ${HF_TOKEN:+set}"; df -h /tmp | tail -1

cd /home/u4320956/fast-draft && mkdir -p results
echo
echo "=== bench_dataset.py --only cascade 200 narrativeqa 30k-100k bs=4 topk=10 tp=4 ==="
PYTHONUNBUFFERED=1 uv run python -u tests/bench_dataset.py \
    --dataset-path data/narrativeqa_chat_30k_100k.jsonl \
    --model-path meta-llama/Llama-3.3-70B-Instruct \
    --draft-model-path meta-llama/Llama-3.2-1B-Instruct \
    --quantization fp8 --kv-cache-dtype fp8_e5m2 \
    --speculative-algorithm STANDALONE \
    --speculative-num-steps 6 --speculative-num-draft-tokens 10 --eagle-topk 10 \
    --max-new-tokens 256 --batch-size 4 --context-length 110000 \
    --mem-fraction-static 0.80 --tp 4 \
    --time-spec --only cascade \
    --result-prefix llama_70b_fp8_1b_bs4_topk10_narrqa200_cascade_${SLURM_JOB_ID}_

echo; echo "=== Done ==="; date
