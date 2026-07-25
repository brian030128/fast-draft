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

**E2E** (Llama-3.1-8B + Llama-3.2-1B, STANDALONE, topk=5, depth=4, narrativeqa 50k,
avg prompt 55 664 tokens, bs=2, H100, `--time-spec`, n=10):

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
2. **Hydragen's decomposition alone is a net loss end to end** (0.64×) *despite* being
   4.63× faster than SGLang's paged draft attention at the kernel level. Draft time goes
   *up*, 0.992 s → 2.379 s, because the draft loop re-plans both cascade levels on every
   step of every decode iteration.
3. **Fast Draft's draft step is 3.34× faster than the Hydragen port** (0.713 s vs
   2.379 s), decomposition held constant.

This strengthens §3 rather than contradicting it. §3's attribution is against
*SGLang-default*, where the differential is the cascade kernel (1.68×) and plan-once is
only 1.10×. Against a *Hydragen-style* baseline the weighting flips, because Hydragen
plans two wrappers per step and has no plan-once: there, static planning is the dominant
term. Both are true; state which baseline each attribution is relative to. The reviewer's
"gains are from static-planning, not prefix-sharing" hypothesis is answered by the pair:
vs SGLang-default it is the kernel; vs Hydragen it is the planning. Neither alone is the
whole story, and saying so is more credible than picking one.

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
