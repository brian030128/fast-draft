# Why FlashInfer's Cascade Kernel Underperforms FastTree

## Setup

Benchmark: `bench_tree_attn.py` — EAGLE-style tree attention with shared prefix + per-branch suffix.
Both kernels receive identical data: `(total_tokens, num_kv_heads, head_dim)` KV tensors with per-token page indices.
Both use scattered index loads (page table lookup → KV data load). The access pattern is the same.

## Measured Gap

At p=4, prefix=4K, topk=4 (a representative EAGLE config):

| | Attention kernel | Reduction | Total |
|---|---|---|---|
| **FastTree** (Triton) | 31.8 us | 8.5 us | 40.3 us |
| **Cascade** (FlashInfer CUDA) | 62.8 us | 3.6 us | 66.4 us |

Both produce ~256-288 work items with similar KV chunk sizes (~128 tokens each).
The gap is entirely in the attention kernel — cascade's is **~2x slower** for the same logical work.

## Root Cause 1: CTA_TILE_Q Waste (75% of Q compute is useless)

FlashInfer's cascade kernel uses two "runners" fused into one kernel:

- **Runner1** (Task 0, "prefill"): `CTA_TILE_Q=64`, 4 warps along Q, 1 warp along KV
- **Runner2** (Task 1, "decode"): `CTA_TILE_Q=16`, 1 warp along Q, 4 warps along KV

For EAGLE topk=4 with GQA=4: `packed_qo = 4 × 4 = 16`. This goes to Task 0 (Runner1) after our `>= 16` threshold fix.

Runner1 allocates a 64-wide Q tile in shared memory and dispatches 4 warps along Q (each handling 16 rows). But only 16 out of 64 packed query slots are valid. **Warps 1-3 load garbage Q data, compute garbage attention scores, and discard the results.** 75% of all Q×K dot products are wasted.

FastTree uses `Q_TILE_SIZE=4` with `kv_group_num=4` → `Q_BLOCK_SIZE=16`. Exactly matches the actual query count. **Zero waste.**

## Root Cause 2: Fused Runner Smem/Register Tax

The cascade kernel fuses Runner1 and Runner2 into a single kernel binary:

```cpp
// persistent_template.cuh
CascadeAttentionKernelTemplate<Runner1, Runner2>(params_1, params_2) {
    Runner1::Run(params_1, &smem_storage_1);  // Task 0 work items
    Runner2::Run(params_2, &smem_storage_2);  // Task 1 work items
}
```

The launch configuration uses the **maximum** of both runners:

```cpp
NUM_THREADS = max(KTraits1::NUM_THREADS, KTraits2::NUM_THREADS);  // 128
smem = max(sizeof(KTraits1::SharedStorage), sizeof(KTraits2::SharedStorage));  // ~34KB
```

Measured via `cuobjdump`: **166 registers/thread**. With 128 threads/CTA:
- Registers per CTA: 166 × 128 = 21,248
- H100 has 65,536 regs/SM → max **3 CTAs/SM**
- With persistent grid launching 2 CTAs/SM → **12.5% occupancy** (256/2048 threads)

Even the tiny Task 1 work items (5 KV tokens each) pay this full register cost because they execute in the same fused kernel.

FastTree's Triton kernel is a single-purpose kernel with ~20KB smem and likely ~80-100 regs/thread, giving better occupancy.

## Root Cause 3: Persistent Kernel Fixed Grid

The cascade kernel always launches `grid = (1, 2 × num_SMs) = (1, 264)` CTAs — a persistent kernel that distributes work items via a per-CTA work list.

For small problems (e.g., p=1, topk=4: only 8+32=40 work items), most of the 264 CTAs have 0 work items. They still get scheduled by the GPU, check their empty work list, and exit. This is not catastrophic (the idle CTAs exit quickly) but adds launch overhead.

FastTree launches exactly as many blocks as needed: `grid = (num_vnodes, num_kv_heads)`. No wasted blocks.

More importantly, when we tried reducing KV splitting (`target_chunks=1`) to minimize partial-state reduction, the persistent grid became the bottleneck: 8 work items on 264 CTAs = 3% SM utilization. FastTree doesn't have this problem because its grid naturally scales with the work.

## Why It's Hard to Fix

### Can't change the warp layout

`get_num_warps_q()` in `prefill.cuh` is a hard binary switch:

```cpp
constexpr uint32_t get_num_warps_q(const uint32_t cta_tile_q) {
    if (cta_tile_q > 16) return 4;  // "prefill" layout
    else return 1;                   // "decode" layout
}
```

There's no middle ground. CTA_TILE_Q=32 still gives 4 warps (same waste as 64). CTA_TILE_Q=16 flips to 1 warp Q / 4 warps KV, which balloons CTA_TILE_KV from 32 to 64-128, **doubling smem** and hurting occupancy.

### Can't use CTA_TILE_Q=32

We tried this. Two problems:
1. `get_num_warps_q(32) = 4` — same 4 warps as CTA_Q=64, so the same wasted-warp problem (50% instead of 75%, but still bad)
2. For topk=16 (`packed_qo=64`): needs `ceil(64/32)=2` QO tiles. The kernel had a **correctness bug** — warps in tile 0 wrote beyond the tile boundary into tile 1's memory. We fixed this with a `qo_upperbound` patch, but the multi-tile overhead made it slower for multi-prefix configs.

### Can't use CTA_TILE_Q=16

We tried this too. The decode-like layout (1 warp Q, 4 warps KV) with `NUM_MMA_KV=2` gives `CTA_TILE_KV=128`, requiring 64KB for K+V smem. The `max(Runner1_smem, Runner2_smem)` fusion means **all work items** pay this 64KB cost. Occupancy drops from ~6 to ~3 CTAs/SM. Result: **worse across the board.**

With `NUM_MMA_KV=1`, `CTA_TILE_KV=64`, smem is reasonable (~36KB). But for topk=16: `ceil(64/16)=4` QO tiles, quadrupling the shared-prefix work items and partial states. The extra reduction overhead dominates.

### Can't eliminate KV splitting

Setting `target_chunks=1` would give each prefix a single work item (no partial states, write-through to final output). But the persistent kernel always launches 264 CTAs. With p=1/topk=4: only 8 work items → **3% SM utilization**. Result: 5-10x slower.

FastTree handles this naturally because it launches `grid=(num_vnodes, num_kv_heads)` — exactly the right number of blocks.

### Can't unfuse the runners

The two runners share one kernel binary. The nvcc compiler allocates `max(regs)` for the fused kernel. Splitting them into separate launches would require architectural changes to the persistent kernel framework.

## What We Tried (and Why It Didn't Work)

### CTA_TILE_Q=32 with 4 warps Q (get_num_warps_q(32)=4)
- 4 warps × 16 rows = 64, but tile is 32 → warps 2-3 overflow into adjacent tiles
- Found and fixed a **correctness bug**: tile boundary check in `qo_upperbound` (persistent.cuh:289)
- Even with the fix, multi-prefix configs regressed 14-16% due to idle warp overhead

### CTA_TILE_Q=16 (decode layout: 1 warp Q, 4 warps KV)
- CTA_TILE_KV balloons to 64-128 (4 warps × NUM_MMA_KV × 16)
- K+V smem doubles to 32-64KB, max(smem) penalty from fused runners
- topk=16 needs 4 QO tiles → 4x more work items
- **Result: worse across the board**

### CTA_TILE_Q=32 with 2 warps Q / 2 warps KV (new `get_num_warps_q` tier)
- Added `if (cta_tile_q > 32) return 4; if (cta_tile_q > 16) return 2; return 1;`
- Found and fixed a **race condition**: `threadblock_sync_mdo_states()` writes to `cta_sync_o_smem` (aliased with `q_smem` via union), then `write_o_()` reuses `q_smem` without barrier. Added `__syncthreads()` guard.
- After fix: 81/81 correctness tests pass across 3 runs
- **Performance**: helps p=1/16K (-5%) but hurts multi-prefix p=4/p=8 by 20-24% due to the extra `__syncthreads` barriers (2 per work item that were no-ops with NUM_WARPS_KV=1)

### target_chunks=1 (eliminate KV splitting)
- Forces each prefix into a single work item → enables write-through (skips reduction kernel)
- But persistent kernel always launches 264 CTAs. With p=1/topk=4: only 8 work items → 3% SM utilization
- **Result: 5-10x slower**

### NUM_MMA_KV=4 for Runner1
- Doubled KV throughput per CTA iteration
- Increased smem, reduced occupancy
- **Result: 3-5% worse**

## What We Did Improve

### 3 CTAs/SM grid (scheduler.cuh line 1425)

The persistent kernel was hardcoded to launch 2 CTAs/SM (264 total). With 128 threads per CTA and 166 regs/thread, this gave only 256 threads/SM = 12.5% occupancy, severely limiting HBM bandwidth utilization (measured: 0.13-0.25 TB/s out of 3.35 TB/s peak).

Bandwidth analysis revealed FastTree achieved ~2x better utilization by having more blocks per SM. The register budget allows 3 CTAs: 3 × 128 × 166 = 63,744 ≤ 65,536 regs/SM.

Changing from `num_sm *= 2` to `num_sm *= 3` gives 396 CTAs total, 384 threads/SM, 18.75% occupancy.

| Config | 2 CTA/SM | 3 CTA/SM | FastTree | Speedup |
|---|---|---|---|---|
| p=1, 4K, topk=8 | 0.039ms | **0.023ms** | 0.028ms | 1.7x, beats FT |
| p=4, 4K, topk=4 | 0.075ms | **0.048ms** | 0.050ms | 1.6x, beats FT |
| p=4, 16K, topk=16 | 0.279ms | **0.141ms** | 0.155ms | 2.0x, beats FT |
| p=8, 16K, topk=16 | 0.436ms | **0.227ms** | 0.253ms | 1.9x, beats FT |

Cascade now **wins in 19 of 27 benchmark configs**.

### kv_limit cap at 1024 (scheduler.cuh line ~1486)

For large-prefix multi-prefix configs (p=4+, prefix=16K), the scheduler created too few KV chunks per prefix. Example: p=8/16K/topk=16 had `target_chunks=4`, giving `kv_limit=4096` → only 256 Task 0 work items. FastTree creates 1088 blocks (KV_SPLIT=1024). The 4x parallelism gap starves HBM bandwidth.

Fix: enforce `target_chunks >= max_kv_len / 1024`, creating enough items to saturate the memory bus.

| Config | Before | After | FastTree | Cascade/FT ratio |
|---|---|---|---|---|
| p=8, 16K, topk=16 | 0.436ms | **0.293ms** | 0.252ms | 1.16x (was 1.73x) |
| p=4, 16K, topk=8 | 0.180ms | **0.145ms** | 0.148ms | **0.98x (wins!)** |
| p=8, 16K, topk=8 | 0.297ms | **0.249ms** | 0.232ms | 1.07x (was 1.28x) |
| p=4, 16K, topk=16 | 0.279ms | **0.184ms** | 0.154ms | 1.20x (was 1.82x) |

Small prefix configs unchanged (their kv_limit was already ≤ 1024).

### Threshold fix: `>` → `>=` (scheduler.cuh line 1416)

The scheduler classified work items into Task 0 (prefill) or Task 1 (decode) based on `packed_qo_len > 16`. For EAGLE topk=4/GQA=4, `packed_qo=16` failed this check and went to the **decode path** (Runner2), which has 1 warp Q / 4 warps KV — designed for single-query decode, not shared-prefix attention.

Changing to `>= 16` routes it to Task 0 (Runner1), which at least amortizes KV reads across the 16 packed queries using 4 warps along Q.

Impact at p=1, prefix=16K, topk=4: **0.092ms → 0.047ms (1.94x faster)**

### Tile boundary fix (persistent.cuh line 289)

When `CTA_TILE_Q < NUM_WARPS_Q × 16`, warps beyond the tile boundary could write into adjacent QO tiles' memory. We fixed `qo_upperbound` to use the tile boundary instead of the warp boundary:

```cpp
// Before (per-warp, allows overflow):
qo_upperbound = min(q_len, ceil_div(qo_packed_idx_base + CTA_TILE_Q, gqa_group_size));

// After (per-tile, prevents overflow):
qo_upperbound = min(q_len, ceil_div(packed_qo_start + (blockIdx.x + 1) * CTA_TILE_Q, gqa_group_size));
```

This is a correctness fix for the general case (currently not triggered with CTA_Q=64 for EAGLE configs, but necessary if CTA_Q is ever reduced).

## Remaining Gap (after all fixes)

After the threshold fix, tile boundary fix, and kv_limit cap, the gap has narrowed significantly but cascade still loses in most multi-prefix configs. Current state:

| Config | Cascade | FastTree | Ratio | Notes |
|---|---|---|---|---|
| p=1, 1K, topk=4 | 0.0176 | 0.0179 | **0.98x** | cascade wins |
| p=1, 16K, topk=4 | 0.0474 | 0.0622 | **0.76x** | cascade wins |
| p=1, 16K, topk=8 | 0.0516 | 0.0677 | **0.76x** | cascade wins |
| p=4, 16K, topk=8 | 0.1446 | 0.1476 | **0.98x** | cascade wins |
| p=4, 4K, topk=4 | 0.0750 | 0.0500 | 1.50x | cascade loses |
| p=8, 4K, topk=8 | 0.1175 | 0.0853 | 1.38x | cascade loses |
| p=8, 16K, topk=16 | 0.2933 | 0.2520 | 1.16x | cascade loses |

### Known architectural factors

| Factor | FastTree | Cascade | Gap |
|---|---|---|---|
| Q tile utilization | 100% (Q_TILE=16) | 25% (CTA_Q=64, packed_qo=16) | 4x wasted compute |
| Smem per CTA | ~20KB | ~34KB (max of 2 runners) | 1.7x |
| Registers/thread | ~80-100 (est.) | 166 (measured) | 1.7-2x |
| Grid sizing | Exact (num_vnodes × num_kv_heads) | Fixed 264 CTAs (persistent) | Inflexible |
| Kernel complexity | Single-purpose Triton | Fused 2-runner persistent CUDA | Higher overhead |

### Remaining losses (8 of 27 configs)

Cascade still loses at p=4/p=8 with prefix=1K (problem too small — kernel launch overhead dominates) and p=8/4K/topk=4-16 (moderate prefix with many prefixes — Q tile waste and per-item overhead accumulate).
