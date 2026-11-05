# FastDraft

Code for *Efficient Tree Draft for Long-Context Speculative Decoding* (NeurIPS 2026).

FastDraft accelerates the draft phase of tree-based speculative decoding: a fused
two-level attention kernel reads the shared context once per draft step, and a static
plan lets the draft loop replay under CUDA Graphs. It is integrated into SGLang.

| Path | Contents |
|---|---|
| `3rdparty/flashinfer` | FlashInfer fork with the kernel (`CascadeBatchAttentionWrapper`) |
| `3rdparty/sglang` | SGLang fork with the FastDraft, Hydragen, DeFT and FastTree draft backends |
| `3rdparty/FastTree-Artifact` | FastTree and DeFT kernels, used as baselines |
| `3rdparty/hydragen` | Hydragen, used as a baseline |
| `tests/`, `scripts/` | Benchmarks and result processing |

## Setup

Requires NVIDIA GPUs, CUDA 12.8 and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/brian030128/fast-draft.git
cd fast-draft
uv sync
export CUDA_HOME=/usr/local/cuda-12.8   # your CUDA 12.8 install
```

The Hydragen baseline also needs flash-attn 2.8.3 built and unpacked to `.venv/fa2`
(or point `FLASH_ATTN2_PATH` at it).

Build the prompt files (`data/`):

```bash
uv run python tests/download_dataset.py --dataset gov_report  --min-tokens 20000
uv run python tests/download_dataset.py --dataset narrativeqa --min-tokens 10000 --max-tokens 100000
uv run python tests/download_dataset.py --dataset pg19        --min-tokens 10000 --max-tokens 100000
```

## Reproducing the results

Each run prints a summary and writes per-prompt CSVs. Speedup is
`mean(AR decode time) / mean(method decode time)`, with decode time = `e2e - ttft`.

**Table 1 (A6000 x 2), Table 2 (A6000), Figure 4:**

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_paper_e2e.sh all
uv run python scripts/aggregate_results.py
```

The other rows use `tests/bench_dataset.py` with these settings:

| Row | Target / draft | Dataset | Top-k / steps | Batch |
|---|---|---|---|---|
| H100 x 1 | Llama-3.1-8B / Llama-3.2-1B | Gov Report | 15 / 6 | 1 |
| H100 x 1 | Llama-3.1-8B / Llama-3.2-1B | NarrativeQA | 10 / 7 | 1 |
| H100 x 1 | Llama-3.1-8B / Llama-3.2-1B | PG-19 | 8 / 8 | 1 |
| RTX Pro 6000 x 4 | Llama-3.3-70B-Instruct / Llama-3.2-1B-Instruct | NarrativeQA | 15 / 10 | 1 |
| RTX Pro 6000 x 1 | Llama-3.1-8B / Llama-3.2-1B | NarrativeQA | 10 / 7 | 4, 8 |

**Figure 2 (two-level vs multi-level):**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python tests/test_cascade_2_level.py sweep --prompt_len 50000 --plot
```

**Figure 3 (kernel benchmark, H100):**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python tests/bench_tree_attn.py \
    --head-configs 32:8 --num-prefixes 1,8 --topk 8,16 --prefix-lens 2048,8192,16384,32768
```

## Notes

- The kernel's default tuning (`include/flashinfer/attention/cascade_tune.cuh`) was
  chosen on A6000. The H100 and RTX Pro 6000 results were measured with an earlier
  tuning.
- Suffix branch packing is on by default; set `SGLANG_CASCADE_PACK_SUFFIX=0` to turn it
  off. The reported end-to-end tables were measured with it off; it does not change
  accept lengths and is neutral at these context lengths.
