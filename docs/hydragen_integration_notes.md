# Integrating Hydragen into SGLang: what happened, what it cost, what to say

Working notes from actually doing the integration the reviewer asked for. Numbers are
H100 80GB HBM3, `tests/bench_hydragen_paged.py`, `num_seqs=4 topk=5 step_offset=3`,
32 query heads / 8 KV heads / head_dim 128, fp16, page_size 1.

## Summary

Hydragen's *algorithm* ports to SGLang cleanly and is a strong baseline — 3.2×–4.7×
over SGLang's own paged draft attention at the kernel level. Hydragen's
*implementation* does not port at all, for one concrete reason (contiguous caches) and
one structural reason (CUDA-graph invalidation on prefix-length change). Both are
quantified below and both are usable in the rebuttal.

The uncomfortable finding is in the other direction too: **one claim in the previous
positioning draft was simply wrong** (the "tree causal mask" gap). It has been removed.
See `hydragen_positioning.md`.

## Challenge 1 — Hydragen's reference code does not import

`hydragen/flash.py` does
`from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_varlen_forward`.
This environment has no flash-attn: `flash_attn` resolves to a namespace package
containing only `flash_attn/cute` (the CuTe DSL), with no `flash_attn_interface`.

*Impact:* `hydragen_attention` cannot be executed here at all, so "run Hydragen's code
as a baseline" is not available as-is.

*What we did:* stubbed `hydragen.flash` in `sys.modules` so `combine_lse` — which only
needs torch/triton/einops — can be imported verbatim from Hydragen's real source file.
That is enough to check the merge primitive, which is the only part of Hydragen we
actually need to compare against.

*Gotcha worth recording:* loading the file with `exec(compile(src, ...))` instead of a
real import breaks Triton. Triton's JIT resolves a kernel's source through
`inspect`/`linecache`; under `exec` its dependency finder walks the wrong function body
and dies with `RuntimeError: Unsupported function referenced: <built-in method stack>`
(it hits the `torch.stack` inside `combine_lse_torch`). Use a `sys.modules` stub, not
`exec`.

*Rebuttal value:* low on its own. Do not write "we could not run Hydragen" — that reads
as an excuse. Use it only to justify why the comparison is against a *port*.

## Challenge 2 — contiguous shared/unique caches vs a paged pool (the real blocker)

`hydragen_attention` takes `shared_ks[i]` as `[sbatch, slen, kvheads, head_dim]` and
`k` as `[batch, kvlen, kvheads, head_dim]` — dense tensors, pre-sized by `setup_caches`
and filled by `append_shared`. SGLang has one paged pool; a prefix is a scattered list
of page ids. A faithful port must gather.

Measured gather cost (K and V, `index_select` from the pool, per layer):

| prefix | gather / layer | achieved BW | per draft iteration (16 layers × 3 steps) |
|--------|----------------|-------------|--------------------------------------------|
| 4 096  | 0.053 ms       | 2517 GB/s   | 2.56 ms                                     |
| 16 384 | 0.189 ms       | 2847 GB/s   | 9.05 ms                                     |
| 50 000 | 0.558 ms       | 2936 GB/s   | 26.78 ms                                    |

Two things to note, and **be precise about both**:

1. The gather runs at ~2.9 TB/s, close to H100 HBM peak. It is memory-bound and cannot
   be engineered away — it moves the whole prefix KV.
2. It is a **tax, not a wall**. At prefix 50K a gathering Hydragen port would cost
   0.558 + 0.794 = 1.35 ms/layer against SGLang-flat's 3.71 ms — still 2.7× faster,
   down from 4.7×. Claiming Hydragen "cannot work" would overstate it; claiming it
   "gives up ~40% of its own advantage and needs ~0.8 GB of scratch per layer" is both
   true and sufficient.

The sharper framing: a page index is **4 bytes per token**, the KV it points at is
`2 × 8 × 128 × 2 = 4096` bytes per token — a **1024× ratio**. Expressing the two-level
split as index arrays (our `cascade_index_gen.py`, 2 Triton kernels per draft
iteration, ~microseconds) instead of gathering (26.8 ms per draft iteration at 50K
context) is the entire reason this is viable inside a serving engine. That contrast —
26.8 ms vs microseconds — is the number to put in the rebuttal.

## Challenge 3 — Hydragen's CUDA graphs break when the shared prefix grows

Hydragen does support CUDA graphs (`GraphedHydragenLlamaModel`), so we must not claim
CUDA graphs as such. But its capture key includes each shared cache's
`sliced_sequence_length`, and `forward()` calls `invalidate()` + re-captures whenever it
changes. Hydragen's own comment in `SharedCache.__init__`:

> "This involves slicing the varlen KV cache to extract the relevant part, which can
> lead to CUDA graph invalidations when varlen is off and the length of the shared
> prompt changes (see `GraphedHydragenLlamaModel`)."

In Hydragen's setting the shared prompt is fixed for the whole generation loop, so this
never fires. In speculative decoding the shared prefix grows every accepted-token
iteration, so it would fire *every* iteration — a full model re-capture per decode
step.

This is exactly the axis the reviewer identified as the true novelty, and it is worth
leaning into rather than deflecting: the static-planning machinery in
`flashinfer_cascade_backend.py` exists because a two-level decomposition whose shared
length changes every step is otherwise incompatible with CUDA graphs.

Supporting number — plan cost is *not* negligible at draft-decode scale:

| prefix | attention (hydragen-paged) | plan (hydragen-paged) | plan / attention |
|--------|----------------------------|-----------------------|------------------|
| 4 096  | 0.098 ms                   | 0.456 ms              | 4.7×             |
| 16 384 | 0.279 ms                   | 0.460 ms              | 1.6×             |
| 50 000 | 0.794 ms                   | 0.464 ms              | 0.6×             |

At short contexts the FlashInfer scheduler costs several times the attention itself.
Plan runs once per draft step (amortized over layers) while attention runs per layer,
so this is not a direct ratio — but it shows why "just call `plan()` every step",
which is what a straight Hydragen port does, is the wrong design at small prefix.

## Challenge 4 — FlashInfer pins a wrapper to one batch size under CUDA graphs

`BatchPrefillWithPagedKVCacheWrapper` records `_fixed_batch_size` at construction and
raises if `plan()` is later called with a different batch size. SGLang captures ~23
distinct batch sizes, so the Hydragen backend needs one wrapper set per captured batch
size: 23 × 3 steps × 2 levels = 138 wrappers, each of which would allocate its own 8 MB
int workspace **and** an 8 MB pinned host buffer — over 2 GB.

*Fix:* allocate int/pinned workspaces once per `(step, level)` and assign them onto each
wrapper directly, bypassing `reset_workspace_buffer` (which allocates a fresh pinned
buffer per call). 6 × 8 MB instead of 138 × 16 MB.

They may **not** be shared across *steps*: all draft steps are planned before the single
captured graph replays them, so step *i*'s schedule has to survive step *i+1*'s
`plan()`. They may be shared across batch sizes, because each captured graph is
replayed immediately after its own `plan()`.

Similarly, level-0 page indices are identical for every draft step within an iteration
(the shared prefix does not grow *within* a draft loop), so one level-0 index buffer is
enough; level-1 indices differ per step and need per-step buffers.

*Rebuttal value:* moderate. It is engineering, not science, but it is concrete evidence
that "just use Hydragen" is not a five-line change in a real engine.

## Kernel-level results

| prefix | SGLang flat | Hydragen-paged | Fast Draft cascade | hy/flat | cascade/hy |
|--------|-------------|----------------|--------------------|---------|------------|
| 4 096  | 0.316 ms    | 0.098 ms       | 0.086 ms           | 3.22×   | 1.14×      |
| 16 384 | 1.231 ms    | 0.279 ms       | 0.138 ms           | 4.42×   | 2.02×      |
| 50 000 | 3.713 ms    | 0.794 ms       | 0.419 ms           | 4.68×   | 1.90×      |

Read this honestly: **most of the kernel-level win is Hydragen's idea, not ours.**
Adding Hydragen-paged as a baseline drops our headline kernel speedup from ~4.7× (vs
SGLang) to ~1.9× (vs Hydragen). That is the number the reviewer is entitled to, and the
paper is stronger for reporting it than for having it discovered in review.

The remaining 1.9× is the fused two-level kernel (one pass, no separate
`merge_state_in_place` round-trip through HBM) plus static planning. Note it *grows*
with prefix length, which is the regime the paper targets.

## E2E results — the headline

Llama-3.1-8B target + Llama-3.2-1B draft, STANDALONE spec decode, topk=5, depth=4,
narrativeqa 50k (avg prompt 55 682 tokens), bs=2, H100, `--num-samples 8`.

| phase                     | decode (s) | tok/s | accept | vs SGLang paged |
|---------------------------|-----------:|------:|-------:|----------------:|
| `paged` (SGLang baseline) | 0.68       | 194.7 | 4.45   | —               |
| `hydragen_no_cg`          | 1.40       | 106.4 | 4.42   | 0.55×           |
| `hydragen` (CUDA graphs)  | 0.99       | 129.6 | 4.42   | 0.67×           |
| `cascade` (Fast Draft)    | 0.53       | 240.5 | 4.45   | **1.23×**       |

Three things to take from this.

**1. The Hydragen backend is numerically correct.** Accept length is 4.42 against the
baseline's 4.45 — the draft tree is the same tree. A broken attention implementation
would collapse acceptance. This validates the port, so the timing numbers mean
something.

**2. Hydragen's decomposition, ported faithfully, is a net *loss* end-to-end** — 0.67×
of SGLang's own paged draft attention, despite being 4.68× faster at the kernel level
(see the table above). The kernel win is real and it is Hydragen's; it just does not
survive contact with the draft loop, because the loop re-plans both cascade levels on
every one of the 3 draft steps, every decode iteration. CUDA graphs recover part of it
(0.55× → 0.67×) but cannot elide the planning, which happens on the host before replay.

**3. That gap is exactly what the paper's static planning closes.** Fast Draft is
1.23× the baseline and **1.85× the Hydragen-paged port** (240.5 / 129.6) — with the
decomposition math held constant between the two. The two backends differ only in
per-step planning vs plan-once-and-patch, and the fused vs separate merge.

This is a much better answer to the reviewer than the previous framing. It concedes
the primitive, adds the missing baseline, and shows the contribution is the part that
makes the primitive actually pay off inside a serving engine — rather than claiming
the primitive.

## Merge primitive — provably the same, up to the log base

Test (experiment 1 in `tests/bench_hydragen_paged.py`): run real attention over
prefix(2048)+suffix(4) as the reference, then run it over each half separately and merge
the two partial results. A merge implements the same primitive iff it reconstructs the
reference.

| merge implementation                    | max abs   | max rel   | verdict |
|-----------------------------------------|----------:|----------:|---------|
| FlashInfer `merge_state`                | 1.221e-04 | 7.911e-04 | EXACT   |
| Hydragen `combine_lse` (raw LSE)        | 2.028e-02 | 1.315e-01 | DIFFERS |
| Hydragen `combine_lse`, LSE × ln 2      | 1.221e-04 | 7.911e-04 | EXACT   |
| Hydragen `combine_lse_triton`, LSE × ln 2 | 1.221e-04 | 7.911e-04 | EXACT |

**FlashInfer carries the LSE in log2 space; Hydragen's `combine_lse` uses `exp()`, i.e.
natural log.** Rescale by ln 2 and the two agree to the last digit — identical residual
against the reference. That is the whole of the difference.

Two consequences:

* The "we concede the primitive" statement is now a *measured* fact, not a rhetorical
  concession: Hydragen's own merge kernel, imported verbatim, reproduces FlashInfer's
  result exactly. Say this plainly in related work.
* Practical footgun worth a footnote: feeding a FlashInfer LSE to a natural-log merge
  (or vice versa) silently produces wrong softmax weights — ~13% relative error here,
  with no error raised. My first version of this test made exactly that mistake and
  reported "DIFFERS" for both Hydragen variants.

## Draft/verify breakdown (`--time-spec`)

Same configuration, n=10:

| phase      | draft (s) | verify (s) | tok/s | accept |
|------------|----------:|-----------:|------:|-------:|
| `paged`    | 0.992     | 1.035      | 217.8 | 4.64   |
| `hydragen` | 2.379     | 1.005      | 139.4 | 4.63   |
| `cascade`  | 0.713     | 1.005      | 259.3 | 4.64   |

Verify time is constant (1.005–1.035 s) and accept length is constant (4.63–4.64), so the
entire effect is in the draft step, which is the only thing any of these backends touch.
Hydragen's decomposition *increases* draft time 0.992 → 2.379 s even though its kernel is
4.6× faster, and Fast Draft's draft step is **3.34× faster than the Hydragen port**.

Note for attribution (and to keep this consistent with `ablation_report.md`): against
*SGLang-default* the differential is mostly the cascade kernel (1.68×) with plan-once
contributing only 1.10×. Against a *Hydragen-style* baseline the weighting flips, because
Hydragen plans two wrappers per draft step and has no plan-once. Both are true; always say
which baseline an attribution is relative to.

## Open items

* E2E numbers for the `hydragen` phase (`SGLANG_HYDRAGEN_DRAFT=1`) — the kernel-level
  gap above says the E2E gap should be smaller, since draft attention is only part of a
  draft step.
* The gather baseline is measured, not implemented as a running backend. If a reviewer
  pushes, implementing `SGLANG_HYDRAGEN_GATHER_DRAFT=1` would make the "faithful
  Hydragen" column real rather than derived.
