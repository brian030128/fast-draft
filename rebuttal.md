# Rebuttal

## Reviewer point (c) — positioning against Hydragen

### The paragraph

> **Response to (c).** We thank the reviewer — overlooking Hydragen was our oversight. It
> is the origin of the shared-prefix decomposition and its log-sum-exp recombination; we
> build on that primitive, do not claim it, and will cite it as prior art. Rather than
> argue the point, we implemented Hydragen's decomposition as a draft-decode backend
> inside our own harness and ran it at the configuration of
> Table~\ref{tab:speculative_decoding} (NarrativeQA, Llama-3.1-8B + Llama-3.2-1B, H100,
> top-$k{=}10$, $7$ steps). Its draft time is $5.85$s, against SGLang's $2.32$s,
> FastTree's $3.13$s, and our $1.39$s, at unchanged accept length
> ($7.32$/$7.35$/$7.38$) and constant verify time: Hydragen's decomposition alone is
> $2.53\times$ *slower* than the baseline it is meant to improve, and would be the
> slowest method in the table. Our margin over it is $4.22\times$. This is not a tuning
> artifact — sweeping FlashInfer's split-KV controls on Hydragen's shared-prefix pass
> (`fixed_split_size` $256$–$16384$, and `disable_split_kv`) moves it by under $3\%$, and
> our kernel stays $4.59\times$ faster on identical KV volume. The difference is
> architectural: Hydragen composes cascade levels *outside* the kernel, planning and
> launching each separately, splitting along KV internally, then combining them in a
> third merge kernel, so the two decompositions never inform one another. We instead
> decompose jointly — a single scheduler partitions draft-tree attention over cascade
> levels *and* KV chunks into one work queue, one persistent fused kernel executes it,
> and one reduction merges cross-level and cross-chunk partials together. The gain is
> specific to the regime tree drafting occupies, converging to $1.16\times$ at
> top-$k{=}128$ where Hydragen's own target workload (large batches of independent
> completions) lives. We will revise the manuscript to cite Hydragen in related work,
> report it as a fourth baseline row, and state this scope explicitly.

### Evidence behind each claim

**E2E, at the paper's own config** (`eagle_topk=10`, `speculative_num_steps=7`,
`num_draft_tokens=15`, bs=1, narrativeqa, H100 — job 282236). Reproduces the published
H100/NarrativeQA row (ours 1.24 s, SGLang 2.59 s), so Hydragen slots in as a fourth row:

| phase                    | draft (s) | verify (s) | accept |
|--------------------------|----------:|-----------:|-------:|
| `paged` (SGLang default) | 2.317     | 0.928      | 7.35   |
| `hydragen`               | **5.852** | 0.926      | 7.32   |
| `cascade` (ours)         | **1.388** | 0.926      | 7.38   |

Constant verify time and accept length confirm the port is correct and the effect is
isolated to the draft step. (`tput` / `vs paged` in that run's summary are corrupted by
an outlier sample and are not quoted; draft/verify/accept are Table 1's metrics.)

**The merge primitive is provably identical.** Reconstructing full attention from
separately computed prefix and suffix parts, Hydragen's own `combine_lse` (imported
verbatim) matches FlashInfer's `merge_state` to the same residual, 7.911e-04, once the
LSE is rescaled by ln 2 — FlashInfer carries it in log2, Hydragen in natural log. The
concession is measured, not rhetorical.

**Not a tuning artifact** (prefix 55 664, bs=2, topk=5):

| level-0 config               | time      | achieved BW |
|------------------------------|----------:|------------:|
| default (split-kv auto)      | 0.749 ms  | 305 GB/s    |
| `disable_split_kv=True`      | 0.741 ms  | 308 GB/s    |
| `fixed_split_size` 256–16384 | 0.729 ms  | 313 GB/s    |
| **ours (fused)**             | **0.163 ms** | **1397 GB/s** |

Under 3% across the whole knob range; disabling split-KV changes nothing.

**Regime scope** — query rows swept at *constant* level-0 KV volume:

| top-k | q rows | hydragen | hy BW    | ours     | ours BW   | ours/hy |
|------:|-------:|---------:|---------:|---------:|----------:|--------:|
| 1     | 2      | 0.747 ms | 305 GB/s | 0.107 ms | 2139 GB/s | 7.00×   |
| 5     | 10     | 0.746 ms | 306 GB/s | 0.161 ms | 1413 GB/s | 4.62×   |
| 10    | 20     | 0.729 ms | 313 GB/s | 0.163 ms | 1396 GB/s | 4.46×   |
| 16    | 32     | 0.752 ms | 303 GB/s | 0.168 ms | 1355 GB/s | 4.47×   |
| 128   | 256    | 0.760 ms | 300 GB/s | 0.655 ms |  348 GB/s | 1.16×   |

**The architectural difference**, verified in source rather than inferred:

* Hydragen / stock Cascade Inference compose levels *outside* the kernel — per-level plan
  and launch, split-KV internal to each, then a third `merge_state_in_place`.
* Ours plans one work queue spanning levels **and** KV chunks (each item carries
  `cascade_num_kv_chunks` / `cascade_kv_chunk_idx` beside its level's `kv_start`/`kv_end`,
  bucketed by shape — there is an explicit case where both levels land in Task 1),
  executed in one persistent-kernel launch (`cascade_persistent_attention`,
  `csrc/batch_attention.cu`), with a single reduction merging cross-level and cross-chunk
  partials.

### What not to claim

* **Do not claim a tree causal mask.** `_mask_mode` is only ever `CAUSAL`/`NON_CAUSAL`,
  the draft path plans `causal=False`, and the tree lives entirely in index generation.
  This wrong claim appears in `docs/reviewer_response.md` §1 and must come out.
* **Do not claim planning overhead is the cause.** Hydragen is device-bound: 4.69 ms host
  vs 48.61 ms device per draft iteration, host share 10%.
* **Do not claim occupancy is the cause.** Bandwidth is flat from 2 to 256 query rows.
* **Do not claim the fusion *causes* the speedup.** The architecture is verified and the
  speed is measured, but the causal link is not isolated from other memory-pipeline
  differences. An nsys pass on reduction and launch overhead would settle it. Until then,
  describe the architecture structurally and report the speedup empirically.

### Reproduce

```bash
srun ... bash slurm/hydragen_topk10.sh          # E2E fourth-row numbers
uv run python tests/bench_hydragen_paged.py     # merge equivalence, host/device split
uv run python tests/bench_hydragen_splitkv_tuning.py   # tuning is inert
uv run python tests/bench_hydragen_query_regime.py     # occupancy is not the cause
```
