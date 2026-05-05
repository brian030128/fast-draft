# Cascade Attention Benchmark for Speculative Decoding Draft Trees

## Overview

This benchmark compares two cascade attention strategies for the **draft phase**
of speculative decoding, where only **leaf nodes** (the frontier of the draft
tree) are treated as queries:

| Method | Levels | Description |
|--------|--------|-------------|
| **Multi-Level** | N (varies by tree depth) | Each depth segment becomes a shared KV level |
| **Forced Two-Level** | 2 (always) | Level 0 = prompt, Level 1 = each leaf's full draft path |

Both use `MultiLevelCascadeAttentionWrapper` with CUDA Graph.

---

## Tree Topologies

### Best Case (for Two-Level): Chain-of-Stars

```
Root ─┬── L₁♦       depth 1: (W−1) leaves
      └── N₁ ─┬── L₂♦    depth 2: (W−1) leaves
              └── N₂ ─┬── L₃♦   ...
                      └── ... ─── Lₖ♦  depth D: W leaves
```

- At each depth, 1 parent spawns W children; only 1 continues deeper
- Total nodes = W × D
- Total leaves = W × D − (D − 1)

**Why Two-Level wins here:**

Two-Level redundancy is small because most leaves have very short paths:
- Leaf at depth d reads only d draft pages
- Average path length = Σ(d × count_at_d) / total_leaves ≈ D/2

Multi-Level has (D + 1) levels, each sharing only **1 page** per level.
The intermediate V+LSE writes between D+1 levels add overhead that exceeds
the tiny per-level KV sharing benefit.

**Redundancy analysis:**

```
R_two = Σ_{d=1}^{D} (count_at_d × d)           # total pages read (two-level)
R_multi = Σ_{d=1}^{D} 1 = D                     # unique pages (multi-level shares)
Redundancy = R_two − R_multi × total_leaves
```

For chain-of-stars: each level shares 1 page among only a subset of leaves.
Per-level savings = (leaves_at_or_below_d − 1) × 1 page.
This is modest, and the D+1 level overhead dominates.

### Worst Case (for Two-Level): Lollipop

```
Root ── S₁ ── S₂ ── ... ── Sₖ ─┬── L₁♦
                                ├── L₂♦
                                ├── ...
                                └── Lₙ♦
```

- Stem of length S = D − 1, burst of B = W × D − (D − 1) leaves at the tip
- All leaves at the same depth → 3 levels (prompt, stem, individual)

**Why Two-Level struggles here:**

Every leaf independently reads the **entire stem** of S pages:
- Two-Level total reads = B × (S + 1) pages
- Multi-Level total reads = S + B pages (stem shared, read once)
- **Redundancy = (B − 1) × S pages**

This is maximized when S ≈ budget/2, giving redundancy ≈ budget²/4.

### Random SD (simulates real speculative decoding)

```
Root ─┬── A ─┬── D♦
      │      └── E ── F♦
      ├── B ─── G♦
      └── C♦
```

- Probability-decaying expansion at each depth (Top-K style)
- Children randomly assigned to parents
- Represents realistic tree shapes from EAGLE, Medusa, etc.
- Monte Carlo averaged over N samples

---

## Proof: These Are the Extremes

### Claim

For a fixed budget B = W × D nodes, among all possible tree topologies:

1. **Lollipop maximizes Two-Level redundancy** (best case for Multi-Level)
2. **Chain-of-Stars minimizes Two-Level redundancy** (best case for Two-Level)

### Proof Sketch

**Definition:** Two-Level redundancy R = Σ over all leaves of (path_length − 1),
where path_length is the number of draft nodes from root to leaf.
Equivalently, R counts the total redundant page reads when each leaf
independently reads its full path instead of sharing common prefixes.

**Lollipop maximizes R:**

Given budget B and stem length S:
- Number of leaves = B − S
- Each leaf's path = S + 1 (stem + own page)
- R = (B − S) × S

This is a quadratic in S, maximized at S = B/2, giving R_max = B²/4.

No other topology with B nodes can achieve higher redundancy, because:
- Any branching before the tip creates shorter sub-paths for some leaves
- Shorter paths = less redundancy per leaf
- The lollipop concentrates ALL sharing into one maximal stem

**Chain-of-Stars minimizes R (among trees with depth D):**

- Branching at every depth means each "shared segment" is exactly 1 node
- Leaves at depth d have path length d, but Multi-Level only saves 1 page per level
- The "sharing width" decreases with depth (fewer leaves participate in deeper levels)
- Total redundancy per level = (remaining_leaves − 1) × 1 page
- This is the minimum possible for a tree that spans D depth levels

### Practical Impact

| Regime | Two-Level vs Multi-Level | Why |
|--------|-------------------------|-----|
| Small tree (D,W ≤ 16), short stems | **Two-Level wins** | Redundancy fits in L2 cache; Multi-Level pays intermediate overhead for many levels |
| Large tree, long stems (≥ 64 pages) | **Multi-Level wins** | Redundancy overflows L2 → DRAM bandwidth waste |

---

## Running

```bash
# Main sweep: Multi-Level vs Two-Level across topologies
python tests/test_cascade_2_level.py sweep --plot

# Stem sweep: vary stem length to find crossover point
python tests/test_cascade_2_level.py stem --plot

# Stress test: push to extreme sizes (includes Fused comparison)
python tests/test_cascade_2_level.py stress --plot

# Batch sweep: test L2 cache pollution
python tests/test_cascade_2_level.py batch --plot
```
