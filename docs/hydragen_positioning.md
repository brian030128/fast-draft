# Positioning Fast Draft against Hydragen

Reviewer concern (paraphrased): "Your cascade attention reads the shared prefix once
and combines it with each branch's unique suffix via log-sum-exp. Isn't that just
Hydragen (Juravsky et al., 2024) / Cascade Inference / RadixAttention?"

> **Revision note.** An earlier draft of this document claimed our level-2 attention
> needs a *tree causal mask* that Hydragen lacks. **That claim is false** and has been
> removed. `cascade_index_gen.py` shows level 2 materializes each branch's own ancestor
> path as its own contiguous KV range, so every branch has exactly one query attending
> to all of its own suffix tokens — plain full attention, no tree mask anywhere. A
> reviewer who opens the code would catch this immediately, so the rebuttal must not
> lean on it. The real gaps are Sections 2–4 below, which are stronger anyway because
> they are measurable.

## Short answer

Hydragen and Fast Draft share **one primitive**: attention decomposition into a
shared-prefix term and a unique-suffix term, recombined with the log-sum-exp
correction. We do not claim that primitive. Hydragen established it; FlashInfer's
cascade wrappers and Flash-Decoding's split-K are the same idea. We cite Hydragen as
its origin and state explicitly that our contribution is *not* the decomposition math.

We verify the primitive really is identical rather than merely similar.
`tests/bench_hydragen_paged.py` computes attention over prefix+suffix as a reference,
then merges the separately-computed prefix and suffix parts with each implementation:

| merge implementation                      | max rel error vs full attention |
|-------------------------------------------|--------------------------------:|
| FlashInfer `merge_state`                  | 7.911e-04 (exact, fp16 rounding) |
| Hydragen `combine_lse`, LSE × ln 2        | 7.911e-04 (identical residual)   |
| Hydragen `combine_lse_triton`, LSE × ln 2 | 7.911e-04 (identical residual)   |

Hydragen's own merge kernel, imported verbatim, reproduces FlashInfer's result to the
last digit; the only discrepancy is a log base convention (FlashInfer carries the LSE in
log2, Hydragen in natural log). So "same primitive" is a measured fact here, not a
rhetorical concession.

What is new is **the workload we apply it to and what that workload forces us to
build**: the tree-draft decode step of speculative decoding. Hydragen's implementation
does not run on this problem, and the reasons are structural rather than incidental.

## 1. The regime is inverted from the one Hydragen targets

Hydragen amortizes the shared prefix read across the **batch** — many *independent*
sequences (users, samples, few-shot completions) that happen to share a prompt. Its
wins come from large batch × long prefix; the paper's sweeps run batch sizes from 32
up to 2048–4096 with shared lengths of 1K–16K and unique suffixes of 128–256 generated
tokens (`scripts/microbenchmark.py`).

Fast Draft shares the prefix across the **topk tree branches of a single request's
draft step**. The shape is inverted:

| Axis                            | Hydragen operating point | Fast Draft (EAGLE draft decode) |
|---------------------------------|--------------------------|---------------------------------|
| "batch" that shares the prefix  | 32–4096 independent seqs | topk branches, ~4–16            |
| shared prefix length            | 1K–16K                   | the *whole* request context, up to ~60K |
| unique suffix length            | 128–256 generated tokens | tree depth, ~1–8 draft tokens   |
| sharing lifetime                | static (set once via `setup_caches`) | re-derived every draft step |

The prefix-to-suffix ratio is far more extreme (60K shared vs a handful of unique)
while the batch is tiny. Prefix-read amortization at batch = topk is exactly where a
naive two-pass decomposition struggles to pay for its own merge overhead.

Honest framing: the kernel-level speedup is real and grows with topk and prefix length
(up to 7.5× at topk=16 / prefix=16K in our microbenchmark), and the contribution is
making that regime reachable inside a real engine — not beating Hydragen at Hydragen's
own batch sizes.

## 2. Gap 1 — Hydragen requires contiguous caches; a serving engine has a paged pool

This is the load-bearing gap. `hydragen_attention` takes

```
shared_ks[i] : [sbatch, slen, kvheads, head_dim]     # or a flat varlen buffer
k            : [batch, kvlen, kvheads, head_dim]
```

i.e. *physically dense* tensors, pre-allocated by `setup_caches` and filled by
`append_shared`. The README lists "no paged KV cache / no continuous batching / no
model server / Llama-only" as limitations, and `SharedCache` is a single
`[max_batch_size * max_seq_length, num_heads, head_dim]` slab.

SGLang stores all KV in **one paged pool**; a request's prefix is a scattered list of
page ids, and prefix and draft-suffix pages are interleaved in that pool. A faithful
Hydragen port must therefore *gather* the prefix into a dense tensor — for every
layer, on every draft step. `tests/bench_hydragen_paged.py` measures that gather
against the attention it is meant to accelerate; the gather moves the entire prefix KV
of the layer, which is strictly more traffic than the shared-prefix attention pass
reads. That is not a tuning problem, it is the reason the port is impossible as
published rather than merely slow.

What we build instead: the shared/unique split is expressed as *two index arrays over
the existing page table*, produced on the fly by the Triton kernels in
`cascade_index_gen.py` (`build_shared_indices`, `build_unique_indices`). Nothing is
copied. Hydragen never touches paging; this integration is a substantive part of the
contribution, not a detail.

## 3. Gap 2 — the prefix/suffix split moves every draft step

In Hydragen the shared cache is established once and stays fixed for the whole
generation loop. In speculative decoding the boundary *moves every EAGLE iteration*:
what was "unique suffix" at step k becomes part of the shared context at step k+1, and
accepted draft tokens are absorbed into the prefix after every verify. Fast Draft
re-derives the two-level split on each of the (typically 4–6) draft steps within a
single verified token. Hydragen has no analog to this per-step re-derivation.

## 4. Gap 3 — Hydragen's CUDA graphs do not survive a changing prefix length

Hydragen *does* support CUDA graphs (`GraphedHydragenLlamaModel` in `hydragen/llama.py`),
so we must not claim CUDA graphs as such are novel. What is decisive is *when its
graphs stay valid*. The capture key includes the shared caches' batch sizes, varlen
flags, and **sliced sequence lengths**; `forward()` calls `invalidate()` and re-captures
whenever any of these change. Hydragen's own source comment says so directly
(`hydragen/llama.py`, `SharedCache.__init__`):

> "This involves slicing the varlen KV cache to extract the relevant part, which can
> lead to CUDA graph invalidations when varlen is off and the length of the shared
> prompt changes (see `GraphedHydragenLlamaModel`)."

Hydragen gets away with this because in its setting the shared prompt length is
**constant for the entire generation loop** — only the unique suffix grows, and that
growth is fed to a CUDA-graph-safe split-K kernel (this is exactly why `hydragen/flash.py`
carries a modified xformers kernel instead of calling flash-attn directly).

In speculative decoding the shared prefix length changes *every* accepted-token
iteration. Under Hydragen's design that means a full graph re-capture per decode step,
which is unusable. Making a two-level decomposition coexist with CUDA graphs when the
shared length changes every step is precisely the static-planning work in
`flashinfer_cascade_backend.py`: plan once at max draft depth, then patch
`kv_len` / `kv_end` / `kv_indptr` fields directly in the planner's workspace buffer at
replay instead of re-running the scheduler.

The reviewer is right that this static planning is the narrow core of the novelty. The
response is not to widen the claim but to show the gap is real and to measure it — see
the `hydragen` vs `cascade` columns in the E2E table, which differ *only* on this axis.

## What we added for the rebuttal

* `3rdparty/sglang/.../hydragen_draft_backend.py` — Hydragen's decomposition as a real
  SGLang draft backend (`SGLANG_HYDRAGEN_DRAFT=1`), built on FlashInfer's
  `MultiLevelCascadeAttentionWrapper`, which is exactly "two paged prefill passes +
  `merge_state_in_place`". It plans both levels every step and does no workspace
  patching — i.e. Hydragen's algorithm with none of our additions.
* `tests/bench_hydragen_paged.py` — merge-primitive equivalence, gather cost, and
  flat vs Hydragen-paged vs fused-cascade kernel timings.
* `hydragen` / `hydragen_no_cg` phases in `tests/bench_dataset.py`, so the E2E table
  reports SGLang-paged, Hydragen-paged, and Fast Draft side by side.

`hydragen` → `cascade` isolates our contribution with the decomposition math held
constant. `flat` → `hydragen` shows what Hydragen's idea alone buys in this regime.

## Measured: what the added baseline actually shows

Kernel level (H100, bs=4, topk=5, 32q/8kv heads, head_dim 128, fp16):

| prefix | SGLang flat | Hydragen-paged | Fast Draft | hy/flat | cascade/hy |
|--------|------------:|---------------:|-----------:|--------:|-----------:|
| 4 096  | 0.314 ms    | 0.100 ms       | 0.086 ms   | 3.14×   | 1.16×      |
| 16 384 | 1.250 ms    | 0.289 ms       | 0.135 ms   | 4.32×   | 2.13×      |
| 50 000 | 3.754 ms    | 0.810 ms       | 0.392 ms   | 4.63×   | 2.07×      |

End to end (Llama-3.1-8B + Llama-3.2-1B, STANDALONE, topk=5, depth=4, narrativeqa
50k, avg prompt 55 682 tokens, bs=2, H100):

| phase                     | decode (s) | tok/s | accept | vs SGLang paged |
|---------------------------|-----------:|------:|-------:|----------------:|
| `paged` (SGLang baseline) | 0.68       | 194.7 | 4.45   | —               |
| `hydragen_no_cg`          | 1.40       | 106.4 | 4.42   | 0.55×           |
| `hydragen` (CUDA graphs)  | 0.99       | 129.6 | 4.42   | 0.67×           |
| `cascade` (Fast Draft)    | 0.53       | 240.5 | 4.45   | **1.23×**       |

The honest reading, which is also the strongest one:

* **Hydragen's decomposition alone is a net loss end to end** (0.62–0.67× of the SGLang
  baseline) even though it reads 5× less KV than flat, exactly as promised.
* **The gap is the contribution**, and it is a kernel-efficiency gap, not a planning
  gap. At the draft model's real shape (32 q / 8 kv heads, head_dim 64, bs=2, topk=5,
  prefix 55 664) our fused kernel is **4.66× faster than the Hydragen port** —
  0.163 ms vs 0.760 ms — while both read the same 0.228 GB.
* **Measured, against the reviewer's own hypothesis:** per draft iteration Hydragen
  spends 4.69 ms on host work and 48.61 ms on device work. It is device-bound, planning
  is 10%, and CUDA graphs cannot help because there is little host time to remove. So
  "the gains are static planning, not prefix-sharing" is falsified on our own baseline.
* **The mechanism is occupancy.** With topk=5, pure deduplication caps the speedup over
  flat at 5×. Hydragen gets 2.91× (300 GB/s) — *below* the bound, because level 0 has
  only 10 query rows against a 55 664-token prefix and a stock paged-prefill kernel
  parallelizes over queries and heads. We get 13.58× (1399 GB/s) — *above* the bound,
  because we also split the shared-prefix read across CTAs. Ceiling on this GPU is
  2783 GB/s.
* Accept length is unchanged (3.93 vs 3.96; 4.42 vs 4.45 on the base models), so the
  Hydragen port is a correct implementation, not a strawman.

Earlier drafts of this file said the gap was per-step planning overhead. The host/device
split above refutes that; keep the occupancy argument, which is both true and stronger.

## Suggested related-work sentence

> Hydragen [Juravsky et al., 2024] introduced shared-prefix attention decomposition —
> computing attention separately over a shared prefix and unique suffixes and merging
> via log-sum-exp — for the large-batch shared-prompt setting (e.g. many samples from
> one prompt); FlashInfer's cascade wrappers provide the same decomposition over a
> paged KV cache. We adopt this primitive unchanged and apply it to a setting these
> works do not address: the tree-draft decode of speculative decoding, where the shared
> prefix is a single request's full context, the "batch" is the topk draft-tree
> branches, and the prefix/suffix boundary moves at every draft step. Our contribution
> is (i) deriving the two-level split on the fly over a paged pool rather than
> pre-sized contiguous caches, and (ii) a static plan that keeps the decomposition
> CUDA-graph-capturable even though the shared length changes every iteration — which
> Hydragen's own design explicitly cannot do.

## One-line rebuttal

> We agree the LSE-combine decomposition is Hydragen's, we cite it as such, and we
> verify our merge is numerically the same primitive. Our contribution is its
> application to speculative tree drafting: an on-the-fly two-level split over a paged
> KV cache with a per-step-moving prefix/suffix boundary, kept CUDA-graph-capturable by
> static plan patching — a combination Hydragen was neither designed for nor, by its
> own source comments, able to support.
