# fast-draft

Research project and experiment codebase for the **Fast Draft** paper.

## Purpose

Existing LLM serving frameworks (e.g. SGLang) use page-size-1 paged attention for tree draft generation in speculative decoding. This approach is slow because it does not exploit the shared prefix IO across tree branches — every branch redundantly reads the same prefix KV cache. We propose replacing paged attention with FlashInfer's CascadeAttention (modified for tree drafting) to read the shared prefix once and combine it with each branch's unique suffix.

Since we modify CascadeAttention itself, we maintain a FlashInfer fork in `3rdparty/flashinfer/`.

### Experiments

1. **SGLang EAGLE tree draft kernel swap (done):** Replace SGLang's paged attention tree draft kernel with our batched cascade attention and measure speedup. Benchmark: `tests/bench_e2e.py`.
2. **MagicDec: sequential → tree draft with cascade attention (in progress):** MagicDec is a sequential long-context speculative decoding system. We convert it to tree draft using our cascade attention approach and compare against sequential draft speed, to show that our approach lowers tree draft time enough that even sequential methods should switch to tree drafting. Currently only the cascade attention path is integrated in MagicDec; the pure paged attention tree draft baseline is not yet added.

## What This Does

Modifies SGLang directly (in `3rdparty/sglang/`) to add `CascadeMultiStepDraftBackend` as an alternative to `FlashInferMultiStepDraftBackend`. The cascade backend reads the shared prefix KV cache once and combines it with each branch's unique suffix, eliminating redundant prefix reads across topk branches during EAGLE draft decode.

**How it works:** Setting `SGLANG_CASCADE_DRAFT=1` in the environment causes `DraftBackendFactory._create_flashinfer_decode_backend()` (in `draft_utils.py`) to import and use `CascadeMultiStepDraftBackend` from `flashinfer_cascade_backend.py`.

## Project Layout

```
fast-draft/
  3rdparty/
    sglang/                    # Fork of SGLang (editable install)
      python/sglang/srt/
        speculative/
          draft_utils.py       # Factory with SGLANG_CASCADE_DRAFT toggle
          cascade_index_gen.py # Triton kernels for 2-level cascade KV indices
        layers/attention/
          flashinfer_cascade_backend.py  # CascadeMultiStepDraftBackend
    MagicDec/                  # Fork of MagicDec with cascade attention for tree draft
    flashinfer/                # Fork of FlashInfer with CascadeBatchAttentionWrapper
  tests/
    test_cascade_indices.py    # Unit tests for index generation (21 tests)
    bench_cascade_draft.py     # Kernel-level microbenchmark (flat vs cascade)
    bench_e2e.py               # E2E benchmark: original vs flat vs cascade (STANDALONE spec decode)
  launch.py                    # Server launcher (sets SGLANG_CASCADE_DRAFT=1)
  pyproject.toml
```

## Environment

- **NEVER kill processes you don't own.** This is a shared multi-user system. Other users' processes may be running on GPUs. Only kill processes you started yourself.
- **Before running on GPU**, run `nvidia-smi` to check which GPUs are free (0 MiB used), then use `CUDA_VISIBLE_DEVICES` to select only free GPUs (e.g. `CUDA_VISIBLE_DEVICES=2,3 uv run python ...`). If a GPU has memory in use, assume another user is using it and pick a different one.
- **CUDA 12.8 required.** The default system nvcc is 11.8 which is too old (missing `cuda/functional` header for FlashInfer JIT). Set `CUDA_HOME=/usr/local/cuda-12.8` when running GPU code.
- **Always use `uv run python` to run scripts** (never bare `python`)
- Python venv at `.venv/` (managed by `uv`)
- Both `sglang` and `flashinfer-python` are editable installs from `3rdparty/`
- Source edits go directly to `3rdparty/sglang/python/sglang/...` and `3rdparty/flashinfer/...`
- Depends on: `torch>=2.4`, `triton>=3.0`, `sglang`, `flashinfer-python`

## Commands

```bash
# Run unit tests (index generation)
uv run pytest tests/test_cascade_indices.py -v

# Kernel microbenchmark (no full model needed)
uv run python tests/bench_cascade_draft.py
uv run python tests/bench_cascade_draft.py --topk 4 --num-seqs 1 --prefix-lens 512,1024,2048,4096

# E2E benchmark (original vs flat vs cascade, STANDALONE spec decode)
CUDA_VISIBLE_DEVICES=2,3 uv run python tests/bench_e2e.py \
    --model-path meta-llama/Llama-3.1-8B \
    --draft-model-path meta-llama/Llama-3.2-1B \
    --prompt-lengths 60000 --eagle-topk 10 --max-new-tokens 2000 \
    --num-requests 1 --context-length 65000 --mem-fraction-static 0.40 \
    --batch-size 1 --tp 2
# Useful flags: --skip-original, --only flat/cascade/original, --result-file out.json
```

## Key Design Decisions

- Direct modification of SGLang fork rather than external plugin — simpler, works correctly with `mp.spawn` workers
- Env var toggle: `SGLANG_CASCADE_DRAFT=1` enables cascade backend, `0` (default) uses standard flat backend
- Index generation in Triton: `cascade_index_gen.py` has two kernels that split flat KV indices into shared-prefix (level 1) and unique-suffix (level 2)
- `3rdparty/flashinfer/` contains `CascadeBatchAttentionWrapper` that does the actual 2-level cascade attention

## Critical Files

- `3rdparty/sglang/python/sglang/srt/layers/attention/flashinfer_cascade_backend.py` — cascade backend (`CascadeMultiStepDraftBackend`, `CascadeDraftAttnBackend`)
- `3rdparty/sglang/python/sglang/srt/speculative/cascade_index_gen.py` — Triton kernels for index generation (`build_shared_indices`, `build_unique_indices`)
- `3rdparty/sglang/python/sglang/srt/speculative/draft_utils.py` — `DraftBackendFactory` with `SGLANG_CASCADE_DRAFT` toggle
- `3rdparty/flashinfer/flashinfer/attention.py` — `CascadeBatchAttentionWrapper` (plan/run, `torch.cuda.synchronize()` at line 292)
- `3rdparty/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py` — flat baseline `FlashInferMultiStepDraftBackend` for comparison

## Status

- Index generation: implemented + tested (21/21 pass)
- Cascade draft backend: implemented
- Kernel microbenchmark: cascade is 1.13x faster at EAGLE's operating point (topk=4, prefix~2K) but up to 7.5x at topk=16/prefix=16K
- E2E: 0.86x (14% slower) — overhead from `torch.cuda.synchronize()` in `plan()` (4x per EAGLE iteration) and `seq_lens.cpu()` sync wipes out the small kernel gain at topk=4
