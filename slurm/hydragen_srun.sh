#!/bin/bash
# Hydragen vs Fast Draft comparison, launched with srun (see run_hydragen_srun.sh).
#
# HF_HOME points at /work rather than $HOME: the home quota is 100 GB and full,
# and /work already holds the complete Llama-3.1-8B-Instruct / 3.2-1B-Instruct
# snapshots. Weights are resolved (and downloaded, if anything is missing) on
# the compute node, not the login node.

set -uo pipefail

echo "=== Job info ==="
echo "Job ID:    ${SLURM_JOB_ID:-none}"
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

# Weights live on /work, which has room; $HOME does not.
export HF_HOME=/work/u4320956/hf
export HF_HUB_CACHE=/work/u4320956/hf/hub
unset HF_HUB_OFFLINE

WT=/home/u4320956/fast-draft/.claude/worktrees/hydragen-compare
export PYTHONPATH=${WT}/3rdparty/sglang/python
cd /home/u4320956/fast-draft

TARGET=meta-llama/Llama-3.1-8B-Instruct
DRAFT=meta-llama/Llama-3.2-1B-Instruct

echo
echo "=== HF cache ==="
echo "HF_HOME=$HF_HOME"
df -h /work/u4320956 | tail -1

echo
echo "=== 0/2 resolve weights on this node (downloads only what is missing) ==="
uv run python - <<PY
import os
from huggingface_hub import snapshot_download
for repo in ("${TARGET}", "${DRAFT}"):
    p = snapshot_download(
        repo,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"],
    )
    print(f"  {repo} -> {p}", flush=True)
PY
echo "download exit: $?"

echo
echo "=== 1/2 kernel microbenchmark ==="
uv run python ${WT}/tests/bench_hydragen_paged.py \
    --num-seqs 4 --topk 5 --step-offset 3 \
    --prefix-lens 4096,16384,50000 \
    --num-qo-heads 32 --num-kv-heads 8 \
    --num-layers 16 --num-draft-steps 3
echo "microbench exit: $?"

echo
echo "=== 2/2 E2E: paged vs hydragen vs cascade (--time-spec) ==="
uv run python ${WT}/tests/bench_dataset.py \
    --dataset-path /home/u4320956/fast-draft/data/narrativeqa_chat_50k.jsonl \
    --model-path ${TARGET} \
    --draft-model-path ${DRAFT} \
    --speculative-algorithm STANDALONE \
    --speculative-num-steps 4 \
    --eagle-topk 5 \
    --max-new-tokens 256 \
    --num-samples 16 \
    --batch-size 2 \
    --context-length 64000 \
    --mem-fraction-static 0.60 \
    --tp 1 \
    --time-spec \
    --skip original \
    --skip flat_no_cg \
    --skip cascade_no_cg \
    --skip cascade_per_step \
    --skip cascade_per_step_no_cg \
    --skip fasttree \
    --skip hydragen_no_cg \
    --result-dir ${WT}/results \
    --result-prefix hydragen_srun_${SLURM_JOB_ID:-manual}_
echo "e2e exit: $?"

echo
echo "=== Done ==="
date
