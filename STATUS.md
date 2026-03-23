# fast-draft: Project Status

*Last updated: 2026-03-21*

## Overview

Cascade draft attention for SGLang EAGLE speculative decoding. Modifies SGLang directly (in `3rdparty/sglang/`) to add `CascadeMultiStepDraftBackend` as an alternative to `FlashInferMultiStepDraftBackend`. The cascade backend reads the shared prefix KV cache once and combines it with each branch's unique suffix, eliminating redundant prefix reads across topk branches during EAGLE draft decode.

## Repository State

- **Branch:** `master` (no commits yet — all files staged or untracked)
- **Main branch:** `main`
- **Python:** 3.11, managed by `uv`
- **Submodules:** `3rdparty/sglang` and `3rdparty/flashinfer` (both have local modifications)

## Component Status

### Index Generation (`cascade_index_gen.py`) — Done
- Two Triton kernels (`build_shared_indices`, `build_unique_indices`) that split flat KV indices into shared-prefix (level 1) and unique-suffix (level 2)
- **234 lines**
- **21/21 unit tests passing** (`tests/test_cascade_indices.py`, 285 lines)

### Cascade Draft Backend (`flashinfer_cascade_backend.py`) — Done
- `CascadeMultiStepDraftBackend` and `CascadeDraftAttnBackend` classes
- **268 lines**
- Activated via `SGLANG_CASCADE_DRAFT=1` env var in `draft_utils.py` (259 lines)

### FlashInfer Fork (`3rdparty/flashinfer/`) — Done
- `CascadeBatchAttentionWrapper` added to `flashinfer/attention.py` (522 lines)
- Handles 2-level cascade attention (plan/run)
- Known issue: `torch.cuda.synchronize()` at line 292 causes overhead

### Cascade Attention Output Tests (`test_cascade_attention_output.py`) — Done
- **670 lines** of correctness tests verifying cascade attention matches flat attention output

### Kernel Microbenchmark (`bench_cascade_draft.py`) — Done
- **252 lines**, compares flat vs cascade at the kernel level
- Results: cascade is **1.13x faster** at EAGLE's operating point (topk=4, prefix~2K), up to **7.5x at topk=16/prefix=16K**

### Profiling Tool (`profile_cascade_draft.py`) — Done
- **519 lines**, kernel-level profiling utility

### E2E Benchmark (`bench_e2e.py`) — Done
- **340 lines**, runs flat vs cascade phases on real ShareGPT prompts via `sglang.Engine`
- Supports `--profile` flag for torch profiler traces
- Recent fix: replaced undefined `profiling_started` variable with `profile` parameter

### Server Launcher (`launch.py`) — Done
- Sets `SGLANG_CASCADE_DRAFT=1` and launches sglang server

## Benchmark Results

Existing traces in `traces/`:
- `flat-*.trace.json.gz` (DECODE + EXTEND)
- `cascade-*.trace.json.gz` (DECODE + EXTEND)

From `results.json` (10 requests per bucket):
| Prompt Tokens | Backend  | Avg Latency (s) | Avg TPS   | Avg Accept Length |
|---------------|----------|------------------|-----------|-------------------|
| ~445          | baseline | 1.039            | 123.2     | 4.92              |

## Performance Summary

- **Kernel-level:** Cascade is faster (1.13x–7.5x depending on topk/prefix length)
- **E2E:** Currently **0.86x (14% slower)** due to:
  - `torch.cuda.synchronize()` in `CascadeBatchAttentionWrapper.plan()` called 4x per EAGLE iteration
  - `seq_lens.cpu()` sync that wipes out the small kernel gain at topk=4

## Known Issues / Next Steps

1. **E2E overhead:** The synchronization cost in `plan()` dominates at EAGLE's typical operating point (topk=4). Eliminating or batching the `torch.cuda.synchronize()` calls is the critical path to E2E wins.
2. **No commits yet:** All work is unstaged/untracked on `master`. Needs initial commit and push.
3. **`results.json`** contains prior benchmark data — some entries show `avg_completion_tokens: 1` suggesting possible early stopping or short completions at certain prompt lengths.

## File Inventory

```
fast-draft/
  CLAUDE.md                          # Project instructions for Claude Code
  STATUS.md                          # This file
  launch.py                          # Server launcher
  pyproject.toml                     # Project config (uv)
  uv.lock                            # Dependency lock
  results.json                       # Prior benchmark results
  traces/                            # Profiler trace outputs
  tests/
    test_cascade_indices.py    (285) # Unit tests — 21/21 pass
    test_cascade_attention_output.py (670) # Correctness tests
    bench_cascade_draft.py     (252) # Kernel microbenchmark
    bench_e2e.py               (340) # E2E benchmark
    profile_cascade_draft.py   (519) # Profiling tool
  3rdparty/
    sglang/                          # Fork (editable install)
      python/sglang/srt/
        speculative/
          cascade_index_gen.py (234) # Triton index kernels
          draft_utils.py       (259) # Factory with env var toggle
        layers/attention/
          flashinfer_cascade_backend.py (268) # Cascade backend
          flashinfer_backend.py          # Flat baseline
    flashinfer/                      # Fork (editable install)
      flashinfer/
        attention.py           (522) # CascadeBatchAttentionWrapper
```
