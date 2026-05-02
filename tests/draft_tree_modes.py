"""
Benchmark modes for draft tree attention comparison.

Contains:
  - sweep: depths × widths × topologies (Multi-Level vs Two-Level)
  - stem:  sweep stem length to find crossover
  - stress: push budget/page_size to extremes (3-way: Multi vs Two vs Fused)
  - batch:  sweep batch size for L2 cache pollution test
"""

from typing import List
import math

import torch

from draft_tree_lib import (
    allocate_kv_cache_and_pages,
    benchmark_fused_config,
    benchmark_one_config,
    build_min_sharing_tree,
    build_sampled_tree,
    build_forced_two_level_metadata,
    build_lollipop_tree,
    build_multi_level_metadata,
    build_max_sharing_tree,
    replicate_metadata,
)


# ---------------------------------------------------------------------------
# Sweep Mode — Multi-Level vs Two-Level (both unfused, w/ CUDA Graph)
# ---------------------------------------------------------------------------

def run_benchmark(
    depths: List[int],
    widths: List[int],
    prompt_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
    random_n: int,
    warmup: int,
    repeat: int,
    do_plot: bool,
):
    """Compare Multi-Level vs Two-Level cascade attention (both with CUDA Graph).

    Topologies:
      - Minimum Sharing: balanced fan-out, all `width` leaves at depth `depth`
                         (least prefix sharing across leaves).
      - Maximum Sharing: lollipop (single stem + leaf burst at tip — every
                         leaf shares the entire stem prefix).
      - Tree Sampled:    random SD-style tree (simulates real speculative
                         decoding), Monte-Carlo averaged over `random_n` samples.
    """
    torch.manual_seed(42)
    # results: (depth, width, topo, multi_ms, two_ms, speedup, levels)
    results = []

    total_configs = len(depths) * len(widths) * 3
    done = 0

    for width in widths:
        for depth in depths:
            for topo_name, builder_fn in [
                ("Minimum Sharing", build_min_sharing_tree),
                ("Maximum Sharing", build_max_sharing_tree),
            ]:
                edges = builder_fn(depth, width)
                kv_data, prompt_pages, node_page_map = allocate_kv_cache_and_pages(
                    edges, prompt_len, num_kv_heads, head_dim, page_size, dtype,
                )

                multi_meta = build_multi_level_metadata(
                    edges, prompt_pages, node_page_map, page_size, prompt_len,
                )
                two_meta = build_forced_two_level_metadata(
                    edges, prompt_pages, node_page_map, page_size, prompt_len,
                )

                if multi_meta is None or two_meta is None:
                    continue

                total_q = multi_meta["total_q"]
                q = torch.randn(total_q, num_qo_heads, head_dim, device="cuda", dtype=dtype)

                multi_ms = benchmark_one_config(
                    multi_meta, q, kv_data,
                    num_qo_heads, num_kv_heads, head_dim, page_size, dtype,
                    warmup, repeat,
                )
                two_ms = benchmark_one_config(
                    two_meta, q, kv_data,
                    num_qo_heads, num_kv_heads, head_dim, page_size, dtype,
                    warmup, repeat,
                )

                speedup = multi_ms / two_ms if two_ms > 0 else float("inf")
                results.append((depth, width, topo_name, multi_ms, two_ms, speedup,
                                multi_meta["num_levels"]))
                done += 1
                print(f"  [{done}/{total_configs}]  W={width:>2d}  D={depth:>3d}  "
                      f"topo={topo_name:<16s}  multi={multi_ms:.4f}ms  "
                      f"two={two_ms:.4f}ms  M/T={speedup:.2f}x  "
                      f"(levels={multi_meta['num_levels']}, leaves={total_q})")

            # Tree-Sampled topology: Monte Carlo with N random SD-style samples
            rand_multi_total = 0.0
            rand_two_total = 0.0
            for sample_i in range(random_n):
                edges = build_sampled_tree(depth, width, seed=sample_i * 1000 + depth)
                if not edges:
                    continue
                kv_data, prompt_pages, node_page_map = allocate_kv_cache_and_pages(
                    edges, prompt_len, num_kv_heads, head_dim, page_size, dtype,
                )
                multi_meta = build_multi_level_metadata(
                    edges, prompt_pages, node_page_map, page_size, prompt_len,
                )
                two_meta = build_forced_two_level_metadata(
                    edges, prompt_pages, node_page_map, page_size, prompt_len,
                )
                if multi_meta is None or two_meta is None:
                    continue

                total_q = multi_meta["total_q"]
                q = torch.randn(total_q, num_qo_heads, head_dim, device="cuda", dtype=dtype)

                rand_multi_total += benchmark_one_config(
                    multi_meta, q, kv_data,
                    num_qo_heads, num_kv_heads, head_dim, page_size, dtype,
                    warmup, repeat,
                )
                rand_two_total += benchmark_one_config(
                    two_meta, q, kv_data,
                    num_qo_heads, num_kv_heads, head_dim, page_size, dtype,
                    warmup, repeat,
                )

            rand_multi = rand_multi_total / random_n
            rand_two = rand_two_total / random_n
            rand_spd = rand_multi / rand_two if rand_two > 0 else float("inf")
            results.append((depth, width, "Tree Sampled", rand_multi, rand_two, rand_spd, 0))
            done += 1
            print(f"  [{done}/{total_configs}]  W={width:>2d}  D={depth:>3d}  "
                  f"topo={'Tree Sampled':<16s}  multi={rand_multi:.4f}ms  "
                  f"two={rand_two:.4f}ms  M/T={rand_spd:.2f}x  "
                  f"(N={random_n})")

    # -----------------------------------------------------------------------
    # Print Results Table
    # -----------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("Cascade Multi-Level vs Forced Two-Level (both w/ CUDA Graph)")
    print(f"  prompt_len={prompt_len}, heads={num_qo_heads}/{num_kv_heads}, "
          f"head_dim={head_dim}, page_size={page_size}")
    print(f"  warmup={warmup}, repeat={repeat}, random_N={random_n}")
    print(f"{'='*100}")
    print(f"  {'W':>3}  {'D':>3}  {'Topo':>16}  "
          f"{'Multi':>10}  {'Two':>10}  {'M/T':>6}")
    print(f"  {'-'*3}  {'-'*3}  {'-'*16}  "
          f"{'-'*10}  {'-'*10}  {'-'*6}")
    for depth, width, topo, multi, two, spd, lvls in results:
        print(f"  {width:>3}  {depth:>3}  {topo:>16}  "
              f"{multi:>10.4f}  {two:>10.4f}  {spd:>5.2f}x")
    print(f"{'='*100}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    if do_plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.colors import TwoSlopeNorm

            topo_names = ["Minimum Sharing", "Maximum Sharing", "Tree Sampled"]
            sorted_depths = sorted(depths)
            sorted_widths = sorted(widths)

            # Per-topology grids[depth_idx, width_idx] = M/T
            grids = []
            for topo_name in topo_names:
                grid = np.full((len(sorted_depths), len(sorted_widths)), np.nan)
                for d, w, tp, _, _, spd, _ in results:
                    if tp != topo_name:
                        continue
                    di = sorted_depths.index(d)
                    wi = sorted_widths.index(w)
                    grid[di, wi] = spd
                grids.append(grid)

            # Symmetric diverging norm centered at 1.0 (blue = Two wins).
            finite = np.concatenate([g[np.isfinite(g)] for g in grids])
            half = float(np.nanmax(np.abs(finite - 1.0))) if finite.size else 0.5
            half = max(half, 0.05)
            norm = TwoSlopeNorm(vcenter=1.0, vmin=1.0 - half, vmax=1.0 + half)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            im = None
            for col, (topo_name, grid) in enumerate(zip(topo_names, grids)):
                ax = axes[col]
                im = ax.imshow(grid, cmap="RdBu", norm=norm,
                               aspect="equal", origin="lower")
                ax.set_xticks(range(len(sorted_widths)))
                ax.set_xticklabels(sorted_widths)
                ax.set_yticks(range(len(sorted_depths)))
                ax.set_yticklabels(sorted_depths)
                ax.set_xlabel("Width")
                if col == 0:
                    ax.set_ylabel("Depth")
                ax.set_title(f"{topo_name}\nblue = Two wins, red = Multi wins")

                # Annotate cells with M/T values
                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        v = grid[i, j]
                        if np.isnan(v):
                            continue
                        # Pick text color for contrast against the cell background.
                        rgba = plt.get_cmap("RdBu")(norm(v))
                        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=8,
                                color="white" if luminance < 0.5 else "black")

            fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)

            plt.suptitle(
                "Multi-Level vs Forced Two-Level Attention (w/ CUDA Graph)",
                fontsize=13, fontweight="bold",
            )
            out_path = "draft_tree_benchmark.svg"
            plt.savefig(out_path, bbox_inches="tight")
            print(f"\nPlot saved to {out_path}")
        except ImportError:
            print("\nmatplotlib not available, skipping plot.")


# ---------------------------------------------------------------------------
# Stem Sweep Mode — sweep stem length to find crossover
# ---------------------------------------------------------------------------

def sweep_stem_length(
    budgets: List[int],
    prompt_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
    do_plot: bool,
):
    """Sweep stem length for fixed budgets: Multi vs Two vs Fused.

    For each budget, vary stem_length from 1 to budget-1.
    Finds where fused/two-level start losing to multi-level.
    """
    torch.manual_seed(42)
    all_results = []

    for budget in budgets:
        stem_lengths = sorted(set(
            [1, 2, 3] +
            list(range(4, budget - 1, max(1, (budget - 1) // 20))) +
            [budget // 4, budget // 3, budget // 2,
             2 * budget // 3, 3 * budget // 4] +
            [budget - 3, budget - 2]
        ))
        stem_lengths = [s for s in stem_lengths if 1 <= s < budget]

        print(f"\nBudget={budget}: sweeping {len(stem_lengths)} stem lengths...")

        for stem_len in stem_lengths:
            burst_count = budget - stem_len

            edges = build_lollipop_tree(stem_len, budget)
            kv_data, prompt_pages, node_page_map = allocate_kv_cache_and_pages(
                edges, prompt_len, num_kv_heads, head_dim, page_size, dtype,
            )
            multi_meta = build_multi_level_metadata(
                edges, prompt_pages, node_page_map, page_size, prompt_len,
            )
            two_meta = build_forced_two_level_metadata(
                edges, prompt_pages, node_page_map, page_size, prompt_len,
            )
            if multi_meta is None or two_meta is None:
                continue

            total_q = multi_meta["total_q"]
            q = torch.randn(total_q, num_qo_heads, head_dim, device="cuda", dtype=dtype)

            multi_ms = benchmark_one_config(
                multi_meta, q, kv_data,
                num_qo_heads, num_kv_heads, head_dim, page_size, dtype,
                warmup, repeat,
            )
            two_ms = benchmark_one_config(
                two_meta, q, kv_data,
                num_qo_heads, num_kv_heads, head_dim, page_size, dtype,
                warmup, repeat,
            )
            fused_ms = benchmark_fused_config(
                multi_meta, q, kv_data,
                num_qo_heads, num_kv_heads, head_dim, page_size, dtype,
                warmup, repeat,
            )

            all_results.append((budget, stem_len, burst_count, total_q,
                                multi_ms, two_ms, fused_ms))

            best = min(multi_ms, two_ms, fused_ms)
            if best == fused_ms:
                marker = "🔥"
            elif best == multi_ms:
                marker = "✅"
            else:
                marker = "⚠️ "
            print(f"  {marker} stem={stem_len:>4d}  leaves={total_q:>4d}  "
                  f"multi={multi_ms:.4f}  two={two_ms:.4f}  fused={fused_ms:.4f}  "
                  f"F/M={fused_ms/multi_ms:.2f}x  "
                  f"(levels={multi_meta['num_levels']})")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("Stem Sweep: Multi-Level vs Two-Level vs Fused (SD draft, leaf-only)")
    print(f"  prompt_len={prompt_len}, heads={num_qo_heads}/{num_kv_heads}, "
          f"head_dim={head_dim}, page_size={page_size}")
    print(f"{'='*120}")
    print(f"  {'Budget':>6}  {'Stem':>5}  {'Leaves':>6}  "
          f"{'Multi':>8}  {'Two':>8}  {'Fused':>8}  "
          f"{'M/T':>6}  {'F/M':>6}  {'Best':>8}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*6}  "
          f"{'-'*8}  {'-'*8}  {'-'*8}  "
          f"{'-'*6}  {'-'*6}  {'-'*8}")
    for budget, stem, burst, leaves, multi, two, fused in all_results:
        mt = multi / two if two > 0 else 0
        fm = fused / multi if multi > 0 else 0
        best = min(multi, two, fused)
        if best == fused:
            bname = "Fused"
        elif best == multi:
            bname = "Multi"
        else:
            bname = "Two"
        print(f"  {budget:>6}  {stem:>5}  {leaves:>6}  "
              f"{multi:>8.4f}  {two:>8.4f}  {fused:>8.4f}  "
              f"{mt:>5.2f}x  {fm:>5.2f}x  {bname:>8}")
    print(f"{'='*120}")

    # Find crossover
    for budget in budgets:
        b_data = [(s, multi, fused)
                  for b, s, _, _, multi, _, fused in all_results if b == budget]
        b_data.sort()
        crossovers = []
        for i in range(len(b_data) - 1):
            s1, m1, f1 = b_data[i]
            s2, m2, f2 = b_data[i + 1]
            ratio1 = f1 / m1 if m1 > 0 else 0
            ratio2 = f2 / m2 if m2 > 0 else 0
            if (ratio1 < 1.0 and ratio2 >= 1.0) or (ratio1 >= 1.0 and ratio2 < 1.0):
                crossovers.append((s1 + s2) // 2)
        if crossovers:
            print(f"  Budget={budget}: Fused↔Multi crossover at stem ≈ {crossovers}")
        else:
            all_fused_win = all(f / m < 1.0 for _, m, f in b_data if m > 0)
            if all_fused_win:
                print(f"  Budget={budget}: Fused wins at ALL stem lengths 🔥")
            else:
                print(f"  Budget={budget}: Multi-Level wins at ALL stem lengths ✅")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    if do_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            cmap = plt.cm.viridis
            colors = [cmap(i / max(1, len(budgets) - 1)) for i in range(len(budgets))]

            ax1 = axes[0]
            for bi, budget in enumerate(budgets):
                b_data = [(s, fused / multi if multi > 0 else 0)
                          for b, s, _, _, multi, _, fused in all_results if b == budget]
                b_data.sort()
                ax1.plot([x[0] for x in b_data], [x[1] for x in b_data],
                         "-o", color=colors[bi], label=f"B={budget}",
                         linewidth=1.5, markersize=3)
            ax1.axhline(y=1.0, color="red", linestyle="--", linewidth=2, alpha=0.8)
            ax1.set_xlabel("Stem Length")
            ax1.set_ylabel("Fused / Multi-Level")
            ax1.set_title("Fused vs Multi-Level\nBelow red = Fused wins")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            ax2 = axes[1]
            mid_budget = budgets[len(budgets) // 2]
            b_data = [(s, multi, two, fused)
                      for b, s, _, _, multi, two, fused in all_results if b == mid_budget]
            b_data.sort()
            stems = [x[0] for x in b_data]
            ax2.plot(stems, [x[1] for x in b_data], "-o", color="blue",
                     label="Multi-Level", linewidth=2, markersize=4)
            ax2.plot(stems, [x[2] for x in b_data], "-s", color="red",
                     label="Two-Level", linewidth=2, markersize=4)
            ax2.plot(stems, [x[3] for x in b_data], "-^", color="green",
                     label="Fused", linewidth=2, markersize=4)
            ax2.set_xlabel("Stem Length")
            ax2.set_ylabel("Latency (ms)")
            ax2.set_title(f"Absolute Latency (Budget={mid_budget})")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.suptitle(
                "Stem Sweep: Multi vs Two vs Fused\n"
                f"prompt_len={prompt_len}, heads={num_qo_heads}/{num_kv_heads}, "
                f"head_dim={head_dim}",
                fontsize=14, fontweight="bold",
            )
            plt.tight_layout()
            out_path = "stem_sweep_benchmark.svg"
            plt.savefig(out_path, bbox_inches="tight")
            print(f"\nStem sweep plot saved to {out_path}")
        except ImportError:
            print("\nmatplotlib not available, skipping plot.")


# ---------------------------------------------------------------------------
# Stress Test — 3-way: Multi-Level vs Two-Level vs Fused
# ---------------------------------------------------------------------------

def stress_test(
    budgets: List[int],
    page_sizes: List[int],
    prompt_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
    do_plot: bool,
):
    """Stress test: fix stem=budget/2, sweep budgets × page_sizes."""
    torch.manual_seed(42)
    per_token_kv_bytes = num_kv_heads * head_dim * 2 * 2
    l2_cache_bytes = 6 * 1024 * 1024

    all_results = []

    for ps in page_sizes:
        per_page_kv_bytes = ps * per_token_kv_bytes
        print(f"\n{'='*100}")
        print(f"Page Size = {ps}, per-page KV = {per_page_kv_bytes/1024:.1f} KB")
        l2_overflow_stem = l2_cache_bytes // per_page_kv_bytes
        print(f"L2 overflow at stem_length > {l2_overflow_stem} pages")
        print(f"{'='*100}")

        for budget in budgets:
            stem_len = budget // 2
            stem_kv_bytes = stem_len * per_page_kv_bytes
            stem_kv_mb = stem_kv_bytes / (1024 * 1024)
            fits_l2 = stem_kv_bytes <= l2_cache_bytes

            edges = build_lollipop_tree(stem_len, budget)
            kv_data, prompt_pages, node_page_map = allocate_kv_cache_and_pages(
                edges, prompt_len, num_kv_heads, head_dim, ps, dtype,
            )
            multi_meta = build_multi_level_metadata(
                edges, prompt_pages, node_page_map, ps, prompt_len,
            )
            two_meta = build_forced_two_level_metadata(
                edges, prompt_pages, node_page_map, ps, prompt_len,
            )
            if multi_meta is None or two_meta is None:
                continue

            total_q = multi_meta["total_q"]
            q = torch.randn(total_q, num_qo_heads, head_dim, device="cuda", dtype=dtype)

            multi_ms = benchmark_one_config(
                multi_meta, q, kv_data,
                num_qo_heads, num_kv_heads, head_dim, ps, dtype,
                warmup, repeat,
            )
            two_ms = benchmark_one_config(
                two_meta, q, kv_data,
                num_qo_heads, num_kv_heads, head_dim, ps, dtype,
                warmup, repeat,
            )
            fused_ms = benchmark_fused_config(
                multi_meta, q, kv_data,
                num_qo_heads, num_kv_heads, head_dim, ps, dtype,
                warmup, repeat,
            )

            spd_m_vs_t = multi_ms / two_ms if two_ms > 0 else float("inf")
            spd_f_vs_t = fused_ms / two_ms if two_ms > 0 else float("nan")

            all_results.append((budget, ps, stem_len, total_q,
                                stem_kv_mb, fits_l2,
                                multi_ms, two_ms, fused_ms,
                                spd_m_vs_t, spd_f_vs_t))

            l2_tag = "✅ L2" if fits_l2 else "💥 DRAM"
            print(f"  B={budget:>5d}  stem={stem_len:>4d}  leaves={total_q:>4d}  "
                  f"stemKV={stem_kv_mb:>6.1f}MB  [{l2_tag}]  "
                  f"multi={multi_ms:.4f}ms  two={two_ms:.4f}ms  fused={fused_ms:.4f}ms  "
                  f"M/T={spd_m_vs_t:.2f}x  F/T={spd_f_vs_t:.2f}x")

    # Summary
    print(f"\n{'='*130}")
    print("STRESS TEST SUMMARY: Multi-Level vs Two-Level vs Fused")
    print(f"{'='*130}")
    print(f"  {'PageSz':>6}  {'Budget':>6}  {'Stem':>5}  {'Leaves':>6}  "
          f"{'StemKV':>8}  {'L2?':>4}  "
          f"{'Multi':>8}  {'Two':>8}  {'Fused':>8}  "
          f"{'M/T':>6}  {'F/T':>6}  {'Best':>10}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*6}  "
          f"{'-'*8}  {'-'*4}  "
          f"{'-'*8}  {'-'*8}  {'-'*8}  "
          f"{'-'*6}  {'-'*6}  {'-'*10}")
    for (budget, ps, stem, leaves, kv_mb, l2,
         multi, two, fused, spd_mt, spd_ft) in all_results:
        l2_str = "Y" if l2 else "N"
        best = min(multi, two, fused)
        if best == fused:
            best_name = "Fused"
        elif best == two:
            best_name = "Two-Level"
        else:
            best_name = "Multi-Lvl"
        print(f"  {ps:>6}  {budget:>6}  {stem:>5}  {leaves:>6}  "
              f"{kv_mb:>7.1f}M  {l2_str:>4}  "
              f"{multi:>8.4f}  {two:>8.4f}  {fused:>8.4f}  "
              f"{spd_mt:>5.2f}x  {spd_ft:>5.2f}x  {best_name:>10}")
    print(f"{'='*130}")

    if do_plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            cmap = plt.cm.tab10
            styles = ["-o", "-s", "-^"]

            ax1 = axes[0]
            for pi, ps in enumerate(page_sizes):
                ps_data = [(b, s) for b, p, _, _, _, _, _, _, _, s, _ in all_results if p == ps]
                ps_data.sort()
                ax1.plot([x[0] for x in ps_data], [x[1] for x in ps_data],
                         styles[pi % 3], color=cmap(pi), label=f"ps={ps}", linewidth=2)
            ax1.axhline(y=1.0, color="red", linestyle="--", linewidth=2, alpha=0.8)
            ax1.set_xlabel("Budget"); ax1.set_ylabel("Multi / Two")
            ax1.set_title("Multi vs Two\n>1 = Two wins"); ax1.legend(); ax1.grid(True, alpha=0.3)
            ax1.set_xscale("log", base=2)

            ax2 = axes[1]
            for pi, ps in enumerate(page_sizes):
                ps_data = [(b, s) for b, p, _, _, _, _, _, _, _, _, s in all_results if p == ps]
                ps_data.sort()
                ax2.plot([x[0] for x in ps_data], [x[1] for x in ps_data],
                         styles[pi % 3], color=cmap(pi), label=f"ps={ps}", linewidth=2)
            ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=2, alpha=0.8)
            ax2.set_xlabel("Budget"); ax2.set_ylabel("Fused / Two")
            ax2.set_title("Fused vs Two\n>1 = Two wins"); ax2.legend(); ax2.grid(True, alpha=0.3)
            ax2.set_xscale("log", base=2)

            plt.suptitle(f"Stress Test: stem=budget/2\nprompt_len={prompt_len}, "
                         f"heads={num_qo_heads}/{num_kv_heads}, head_dim={head_dim}",
                         fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig("stress_test_benchmark.svg", bbox_inches="tight")
            print("\nStress test plot saved to stress_test_benchmark.svg")
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Batch Sweep — test L2 cache pollution with increasing batch size
# ---------------------------------------------------------------------------

def batch_sweep_test(
    batch_sizes: List[int],
    stem_length: int,
    budget: int,
    prompt_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
    do_plot: bool,
):
    """Sweep batch size for fixed lollipop tree to test L2 cache effects."""
    torch.manual_seed(42)
    per_token_kv_bytes = num_kv_heads * head_dim * 2 * 2
    per_page_kv_bytes = page_size * per_token_kv_bytes
    stem_kv_bytes = stem_length * per_page_kv_bytes
    l2_cache_bytes = 6 * 1024 * 1024

    burst_count = budget - stem_length
    print(f"Batch Sweep: stem={stem_length}, burst={burst_count}, budget={budget}")
    print(f"  Single-request stem KV = {stem_kv_bytes/1024:.1f} KB")
    print(f"  L2 cache ≈ 6 MB → fits ~{l2_cache_bytes // stem_kv_bytes} stems concurrently\n")

    edges = build_lollipop_tree(stem_length, budget)
    single_kv, prompt_pages, node_page_map = allocate_kv_cache_and_pages(
        edges, prompt_len, num_kv_heads, head_dim, page_size, dtype,
    )
    single_total_pages = single_kv.shape[0]
    single_multi = build_multi_level_metadata(edges, prompt_pages, node_page_map, page_size, prompt_len)
    single_two = build_forced_two_level_metadata(edges, prompt_pages, node_page_map, page_size, prompt_len)
    if single_multi is None or single_two is None:
        print("ERROR: metadata generation failed"); return

    single_total_q = single_multi["total_q"]
    all_results = []

    for bs in batch_sizes:
        total_pages = single_total_pages * bs
        try:
            batch_kv = torch.randn(total_pages, 2, page_size, num_kv_heads, head_dim,
                                   device="cuda", dtype=dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠️  batch={bs}: OOM, skipping"); continue

        batch_multi = replicate_metadata(single_multi, bs, single_total_q, single_total_pages)
        batch_two = replicate_metadata(single_two, bs, single_total_q, single_total_pages)
        q = torch.randn(single_total_q * bs, num_qo_heads, head_dim, device="cuda", dtype=dtype)

        multi_ms = benchmark_one_config(batch_multi, q, batch_kv,
                                        num_qo_heads, num_kv_heads, head_dim, page_size, dtype, warmup, repeat)
        two_ms = benchmark_one_config(batch_two, q, batch_kv,
                                      num_qo_heads, num_kv_heads, head_dim, page_size, dtype, warmup, repeat)
        speedup = multi_ms / two_ms if two_ms > 0 else float("inf")
        total_stem_kv_mb = bs * stem_kv_bytes / (1024 * 1024)

        all_results.append((bs, multi_ms, two_ms, speedup, total_stem_kv_mb))
        winner = "Two-Level" if speedup > 1.0 else "⚠️  Multi-Level"
        print(f"  batch={bs:>5d}  multi={multi_ms:.4f}ms  two={two_ms:.4f}ms  "
              f"speedup={speedup:.2f}x  allStemKV={total_stem_kv_mb:.1f}MB  → {winner}")
        del batch_kv, q; torch.cuda.empty_cache()

    print(f"\n{'='*90}")
    print(f"Batch Sweep Summary: stem={stem_length}, burst={burst_count}")
    print(f"{'='*90}")
    for bs, multi, two, spd, kv_mb in all_results:
        print(f"  batch={bs:>5}  multi={multi:.4f}  two={two:.4f}  M/T={spd:.2f}x  stemKV={kv_mb:.1f}MB")
    print(f"{'='*90}")
