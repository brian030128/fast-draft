#!/bin/bash
# Hydragen vs Fast Draft comparison, launched with srun (see run_hydragen_srun.sh).
#
# Weights go on NODE-LOCAL storage. Neither shared filesystem is usable here:
#   $HOME  -- 100 GB quota, full, and its Llama-3.1-8B-Instruct snapshot is
#             missing 3 of 4 shards (this killed jobs 278387 and 278400).
#   /work  -- a weka mount that is not reliably visible from the workers; the
#             same path on hgpn01 showed 8.0K used in one allocation and 71 GB
#             in another, so reading weights from it works only by luck.
# /tmp on the workers is a real local block device (/dev/md131, 28 TB, ~1% used)
# and SLURM_TMPDIR is unset on this cluster, so /tmp/$USER is the path to use.
# Weights are resolved -- and downloaded if absent -- on the allocated GPU node.

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

export HF_HOME=/tmp/${USER}/hf
export HF_HUB_CACHE=${HF_HOME}/hub
mkdir -p "${HF_HUB_CACHE}"
unset HF_HUB_OFFLINE
# Triton/FlashInfer/SGLang JIT caches also land on node-local disk, so a full
# $HOME cannot stall a compile mid-job.
export TRITON_CACHE_DIR=/tmp/${USER}/triton
export FLASHINFER_WORKSPACE_BASE=/tmp/${USER}/flashinfer
mkdir -p "${TRITON_CACHE_DIR}" "${FLASHINFER_WORKSPACE_BASE}"

WT=/home/u4320956/fast-draft/.claude/worktrees/hydragen-compare
export PYTHONPATH=${WT}/3rdparty/sglang/python
cd /home/u4320956/fast-draft

TARGET=meta-llama/Llama-3.1-8B-Instruct
DRAFT=meta-llama/Llama-3.2-1B-Instruct

echo
echo "=== node-local cache ==="
echo "HF_HOME=$HF_HOME  (SLURM_TMPDIR=${SLURM_TMPDIR:-unset})"
df -h /tmp | tail -1

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
