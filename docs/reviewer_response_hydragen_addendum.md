# Addendum to `docs/reviewer_response.md` — the Hydragen baseline now exists

Two updates for `reviewer_response.md`, both from actually building the Hydragen
comparison rather than arguing around it. Kept in a separate file so it does not clobber
that document; fold in as you see fit.

Backing detail: `docs/hydragen_positioning.md`, `docs/hydragen_integration_notes.md`,
`tests/bench_hydragen_paged.py`,
`3rdparty/sglang/.../layers/attention/hydragen_draft_backend.py`.

---

## Correction to §1 — we do **not** add a tree mask to the suffix

`reviewer_response.md` §1 currently describes our FlashInfer modifications as "tree mask
on the suffix, plus the scheduler fixes…". **There is no tree mask.** This is the same
claim that was in `hydragen_positioning.md` and it has been retracted there too.

Verified in code:

* `CascadeBatchAttentionWrapper` only ever sets
  `self._mask_mode = MaskMode.CAUSAL if causal else MaskMode.NON_CAUSAL`
  (`3rdparty/flashinfer/flashinfer/attention.py`). No custom-mask path is used.
* `CascadeMultiStepDraftBackend` plans with `causal=False`
  (`flashinfer_cascade_backend.py`).
* `generate_cascade_unique_kv_indices` (`cascade_index_gen.py`) materializes **each
  branch's own ancestor path as its own contiguous KV range**
  (`prefix_len + topk_id * num_steps + [0, step_offset)`). Each branch contributes one
  query and attends to all `step_offset` tokens of its own range.

So level 2 is plain full attention over per-branch ranges. The tree structure is encoded
in the *index generation*, not in a mask — which is a cleaner story anyway, and it is the
honest one. A reviewer who opens the file will find no mask; the claim must come out
before submission.

The real modifications to cite are the scheduler fixes already documented in
`docs/cascade-vs-fasttree-analysis.md` (Task-0/1 threshold `>` → `>=`, 3-CTA/SM grid,
`kv_limit` cap, tile-boundary fix) plus `plan_for_draft` / `fast_cascade_plan` /
`update_draft_step`.

---

## Update to §4 — the Hydragen e2e baseline is now feasible, and we have the numbers

§4 currently says: *"Hydragen's own engine cannot run EAGLE tree spec-decode, so it is not
directly comparable end-to-end."* That is true of Hydragen's engine, but it lets the
reviewer's core complaint stand unanswered. We no longer have to decline.

`hydragen_draft_backend.py` implements **Hydragen's decomposition as a real SGLang draft
backend** (`SGLANG_HYDRAGEN_DRAFT=1`), on FlashInfer's
`MultiLevelCascadeAttentionWrapper` — literally two paged prefill passes plus
`merge_state_in_place`. It plans both levels every draft step and does no CUDA-graph
workspace patching: Hydragen's algorithm with none of our additions. So
`hydragen → cascade` isolates our contribution with the decomposition held constant.

**E2E**, STANDALONE, topk=5, depth=4, narrativeqa 50k, avg prompt 55 664 tokens,
bs=2, H100, `--time-spec`, n=10.

**At the paper's own configuration** — `eagle_topk=10`, `speculative_num_steps=7`,
`num_draft_tokens=15`, bs=1, Llama-3.1-8B + Llama-3.2-1B, narrativeqa, H100 (job 282236).
This is the config behind the H100/NarrativeQA row of Table~\ref{tab:speculative_decoding},
and it reproduces that row (published: ours 1.24 s, SGLang 2.59 s), so Hydragen slots in
as a fourth row directly:

| phase                     | draft (s) | verify (s) | accept |
|---------------------------|----------:|-----------:|-------:|
| `paged` (SGLang default)  | 2.317     | 0.928      | 7.35   |
| `hydragen` (CUDA graphs)  | **5.852** | 0.926      | 7.32   |
| `cascade` (Fast Draft)    | **1.388** | 0.926      | 7.38   |

Hydragen is **2.53× slower than SGLang** here — the slowest method in the table, below
FastTree's 3.13 s — and our margin over it is **4.22×**, wider than the 3.33× at top-k=5.
Verify time (0.926–0.928 s) and accept length (7.32–7.38) are constant, so the effect is
isolated to the draft step. (The `tput` / `vs paged` columns in this run's summary are
corrupted by an outlier sample and are not reported; draft/verify/accept are the metrics
Table 1 uses and are self-consistent.)

The runs below are at top-k=5 / depth=4, which does *not* match Table 1; they are kept
because the kernel-level analysis was done at that shape.

Llama-3.1-8B-Instruct + Llama-3.2-1B-Instruct (two independent runs, jobs 278505 and
the node-local rerun — reproducible to ~1%):

| phase                     | draft (s) | verify (s) | tok/s       | accept | vs paged |
|---------------------------|----------:|-----------:|------------:|-------:|---------:|
| `paged` (SGLang default)  | 1.127     | 1.184      | 183.7 / 182.2 | 3.96 | —        |
| `hydragen` (CUDA graphs)  | 2.837     | 1.178      | 113.1 / 113.9 | 3.93 | 0.62×    |
| `cascade` (Fast Draft)    | 0.852     | 1.189      | 211.8 / 211.2 | 3.94 | 1.15×    |

Llama-3.1-8B + Llama-3.2-1B (base models, same configuration):

| phase                     | draft (s) | verify (s) | tok/s | accept | vs paged |
|---------------------------|----------:|-----------:|------:|-------:|---------:|
| `paged` (SGLang default)  | 0.992     | 1.035      | 217.8 | 4.64   | —        |
| `hydragen` (CUDA graphs)  | 2.379     | 1.005      | 139.4 | 4.63   | 0.64×    |
| `cascade` (Fast Draft)    | 0.713     | 1.005      | 259.3 | 4.64   | 1.19×    |

**Kernel level** (bs=4, topk=5, 32q/8kv heads, head_dim 128, fp16):

| prefix | SGLang paged | Hydragen-paged | Fast Draft | hy/paged | cascade/hy |
|--------|-------------:|---------------:|-----------:|---------:|-----------:|
| 4 096  | 0.314 ms     | 0.100 ms       | 0.086 ms   | 3.14×    | 1.16×      |
| 16 384 | 1.250 ms     | 0.289 ms       | 0.135 ms   | 4.32×    | 2.13×      |
| 50 000 | 3.754 ms     | 0.810 ms       | 0.392 ms   | 4.63×    | 2.07×      |

What to take from this:

1. **Verify time is identical across all three** (1.005–1.035 s) and accept length is
   unchanged (4.63–4.64). The port is correct, not a strawman, and the effect is isolated
   to the draft step — the same control §5 already relies on.
2. **Hydragen's decomposition alone is a net loss end to end.** At the paper's config
   draft time goes *up*, 2.317 s → 5.852 s (2.53× slower than the baseline it is meant
   to improve).
3. **Fast Draft's draft step is 4.22× faster than the Hydragen port** at the paper's
   config (3.3× at top-k=5), decomposition held constant.

### Why — and it is the strongest answer we have to (d)(1)

The reviewer hypothesizes that our gains come from static planning rather than
prefix-sharing. Measured at the draft model's real shape (32 q / 8 kv heads, head_dim 64,
bs=2, topk=5, 16 layers × 4 steps), splitting one draft iteration into host work (which
CUDA-graph replay must re-execute) and device work (which it captures):

| prefix | backend  | host/iter | device/iter | host share |
|--------|----------|----------:|------------:|-----------:|
| 55 664 | hydragen | 4.69 ms   | 48.61 ms    | 10%        |
| 55 664 | cascade  | 1.48 ms   | 10.44 ms    | 14%        |

Hydragen is **device-bound**; planning is 10% of it. So static planning is *not* the
explanation — which is a point in our favour, because the reviewer's alternative
hypothesis is falsified on our own baseline rather than argued against.

The real mechanism is bandwidth utilization. Per-token KV = 8 × 64 × 2 × 2 B = 2048 B;
at prefix 55 664, bs=2:

| variant  | KV read  | time     | achieved BW | speedup vs flat |
|----------|---------:|---------:|------------:|----------------:|
| flat     | 1.140 GB | 2.213 ms | 515 GB/s    | —               |
| hydragen | 0.228 GB | 0.760 ms | 300 GB/s    | 2.91×           |
| cascade  | 0.228 GB | 0.163 ms | 1399 GB/s   | 13.58×          |

(Ceiling 2783 GB/s, measured by the gather microbenchmark on the same data, same GPU.)

With topk=5, **pure deduplication caps the speedup over flat at 5×**. Hydragen reaches
2.91× — *below* the bound: it reads exactly the reduced volume its decomposition promises,
then moves it at 300 GB/s. Fast Draft reaches 13.58×, *above* the bound, so it is not
merely reading less.

Two candidate explanations were tested and **both fail**, which is worth pre-empting
because a reviewer will raise them:

* **Tuning.** FlashInfer's paged prefill exposes `fixed_split_size` and
  `disable_split_kv`, with split-KV already on by default. Sweeping 256–16384, and
  disabling it outright, moves Hydragen's level 0 by under 3% (305 → 313 GB/s). Best
  tuned, it is still 4.59× slower than ours on identical KV volume.
* **Occupancy.** Sweeping query rows at *constant* level-0 KV volume, Hydragen is pinned
  at ~300–313 GB/s from 2 rows to 256 — 128× more queries changes nothing, so CTA count
  is not the constraint. (An earlier draft of this addendum claimed it was; retracted.)

What survives is architectural. Hydragen and stock Cascade Inference compose levels
*outside* the kernel: each level separately planned and launched, split along KV
internally, then combined by a third `merge_state_in_place`. Ours plans **one work queue
spanning both levels and KV chunks** — each item carries `cascade_num_kv_chunks` /
`cascade_kv_chunk_idx` beside its level's `kv_start`/`kv_end`, bucketed by shape rather
than by level — executed in a single persistent-kernel launch
(`cascade_persistent_attention`, `csrc/batch_attention.cu`), with one reduction merging
cross-level and cross-chunk partials together.

The regime scope, measured: our margin over Hydragen is 7.00× at top-k=1, 4.4–4.6× across
top-k 4–16 (where EAGLE runs), and 1.16× at top-k=128 — i.e. it converges exactly where
Hydragen's own target workload lives (large batches of independent completions, many query
rows). Stating this scope costs nothing and pre-empts the obvious probe.

**Do not claim causality.** We have shown the architecture differs and that we are faster;
we have *not* isolated that the fusion causes the speed rather than other memory-pipeline
differences. An nsys pass on reduction and launch overhead would settle it.

Consistent with §3, which attributes against *SGLang-default* (cascade kernel 1.68×,
plan-once 1.10×) — same conclusion that the kernel, not the planning, carries the win.
State which baseline each attribution is relative to.

Recommended §4 rewrite: keep vLLM-EAGLE as the independent-*engine* row, and add
Hydragen-in-SGLang as the independent-*method* row. The latter is the better controlled
experiment, since it changes one mechanism instead of the whole stack.

---

## Two further items worth adding

**Hydragen's CUDA graphs cannot survive a growing prefix — from its own source.**
`GraphedHydragenLlamaModel` re-captures whenever a shared cache's
`sliced_sequence_length` changes, and `SharedCache.__init__` says so directly:

> "This involves slicing the varlen KV cache to extract the relevant part, which can lead
> to CUDA graph invalidations when varlen is off and the length of the shared prompt
> changes (see `GraphedHydragenLlamaModel`)."

In Hydragen's setting the shared prompt is fixed for the whole generation loop. In
speculative decoding it grows every accepted-token iteration, so this would fire every
step. Useful for §5 (plan-rebuild triggers) as the contrast case.

**Why we port rather than gather.** `hydragen_attention` needs physically dense
`[sbatch, slen, kvheads, d]` caches; SGLang has page ids. Gathering the prefix costs
0.559 ms/layer at 50 K context (2.93 TB/s, i.e. HBM-bound and not optimizable) — 26.8 ms
per draft iteration over 16 layers × 3 steps. A page index is 4 B/token against
4096 B/token of KV, a 1024× ratio. That is the quantitative reason the two-level split is
expressed as index arrays. State it as a ~40% tax on Hydragen's own advantage, not as
"impossible" — it is a tax, and overstating it is checkable.
