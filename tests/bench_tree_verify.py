"""Kernel-level microbenchmark: paged attention (with tree mask) vs cascade for EAGLE tree verify.

Simulates the tree verify attention pattern:
  - A prefix of `prefix_len` KV tokens (shared context)
  - A draft tree of width `topk` and depth `depth` (random parent assignment)

Three approaches compared:
  1. paged+mask: single request, all verify tokens as queries, custom_mask encodes tree
  2. paged_nomask: same single request, NO mask (all-to-all attention) — isolates mask overhead
  3. cascade: 2-level — shared prefix (level 0) + per-token ancestor chain (level 1)

Uses page_size=1 so each KV token occupies exactly one page.

Default parameters match Llama-3.1-8B (no TP):
  num_qo_heads=32, num_kv_heads=8, head_dim=128, dtype=bfloat16.

Usage:
    python tests/bench_tree_verify.py
    python tests/bench_tree_verify.py --prefix-lens 1024,4096,16384
    python tests/bench_tree_verify.py --topks 4,8,16 --depths 6,10
    python tests/bench_tree_verify.py --num-qo-heads 8 --num-kv-heads 2  # TP=4
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "3rdparty", "flashinfer"),
)

import flashinfer
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.cascade import MultiLevelCascadeAttentionWrapper


def generate_random_tree(topk, depth):
    """Generate a random draft tree and compute ancestor sets.

    Returns:
        num_tokens: total verify tokens (1 root + depth * topk)
        ancestors: list of sets, ancestors[i] = set of ancestor token indices (including i)
        depth_of: list of ints, depth_of[i] = depth of token i in tree
    """
    num_tokens = 1 + depth * topk
    parent = [None] * num_tokens
    depth_of = [0] * num_tokens

    for d in range(1, depth + 1):
        layer_start = 1 + (d - 1) * topk
        prev_layer_start = 1 + (d - 2) * topk if d >= 2 else 0
        prev_layer_end = 1 + (d - 1) * topk if d >= 2 else 1
        for k in range(topk):
            token_idx = layer_start + k
            parent_idx = torch.randint(prev_layer_start, prev_layer_end, (1,)).item()
            parent[token_idx] = parent_idx
            depth_of[token_idx] = d

    ancestors = [set() for _ in range(num_tokens)]
    for i in range(num_tokens):
        cur = i
        while cur is not None:
            ancestors[i].add(cur)
            cur = parent[cur]

    return num_tokens, ancestors, depth_of


def time_kernel(fn, warmup, repeat):
    """Time a GPU kernel with CUDA events. Returns median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_evts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_evts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        start_evts[i].record()
        fn()
        end_evts[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_evts, end_evts)]
    return sorted(times)[len(times) // 2]


def bench_tree_verify(
    prefix_lens=(1024, 4096, 16384, 65536),
    topks=(4, 16),
    depths=(6, 10),
    num_qo_heads=32,
    num_kv_heads=8,
    head_dim=128,
    dtype=torch.bfloat16,
    warmup=10,
    repeat=50,
):
    page_size = 1
    results = []
    device = "cuda"

    for topk in topks:
        for depth in depths:
            num_tokens, ancestors, depth_of = generate_random_tree(topk, depth)
            print(
                f"\n{'='*90}\n"
                f"Tree: topk={topk}, depth={depth}, verify_tokens={num_tokens}\n"
                f"{'='*90}"
            )

            for prefix_len in prefix_lens:
                total_pages = prefix_len + num_tokens

                k_cache = torch.randn(
                    total_pages, page_size, num_kv_heads, head_dim,
                    device=device, dtype=dtype,
                )
                v_cache = torch.randn(
                    total_pages, page_size, num_kv_heads, head_dim,
                    device=device, dtype=dtype,
                )
                kv_data = (k_cache, v_cache)

                q = torch.randn(
                    num_tokens, num_qo_heads, head_dim,
                    device=device, dtype=dtype,
                )

                # Shared paged attention setup (1 request, all pages)
                paged_qo_indptr = torch.tensor(
                    [0, num_tokens], device=device, dtype=torch.int32
                )
                all_kv_pages = torch.arange(
                    total_pages, device=device, dtype=torch.int32
                )
                paged_kv_indptr = torch.tensor(
                    [0, total_pages], device=device, dtype=torch.int32
                )
                paged_kv_last_page_len = torch.tensor(
                    [1], device=device, dtype=torch.int32
                )
                kv_len = total_pages

                # ============================================================
                # 1. Paged Attention WITH custom_mask (tree mask baseline)
                # ============================================================
                custom_mask = torch.zeros(
                    num_tokens * kv_len, dtype=torch.bool, device=device
                )
                for i in range(num_tokens):
                    row_start = i * kv_len
                    custom_mask[row_start : row_start + prefix_len] = True
                    for anc in ancestors[i]:
                        custom_mask[row_start + prefix_len + anc] = True

                paged_masked = BatchPrefillWithPagedKVCacheWrapper(
                    torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device),
                    kv_layout="NHD",
                )
                paged_masked.plan(
                    paged_qo_indptr, paged_kv_indptr, all_kv_pages,
                    paged_kv_last_page_len,
                    num_qo_heads, num_kv_heads, head_dim, page_size,
                    custom_mask=custom_mask,
                    q_data_type=dtype, kv_data_type=dtype,
                )
                masked_ms = time_kernel(
                    lambda: paged_masked.run(q, kv_data), warmup, repeat
                )

                # ============================================================
                # 2. Paged Attention WITHOUT mask (all-to-all, isolates mask cost)
                # ============================================================
                paged_nomask = BatchPrefillWithPagedKVCacheWrapper(
                    torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device),
                    kv_layout="NHD",
                )
                paged_nomask.plan(
                    paged_qo_indptr, paged_kv_indptr, all_kv_pages,
                    paged_kv_last_page_len,
                    num_qo_heads, num_kv_heads, head_dim, page_size,
                    causal=False,
                    q_data_type=dtype, kv_data_type=dtype,
                )
                nomask_ms = time_kernel(
                    lambda: paged_nomask.run(q, kv_data), warmup, repeat
                )

                # ============================================================
                # 3. Paged Attention prefix-only (no tree tokens in KV)
                #    Isolates the prefix computation cost
                # ============================================================
                prefix_only_pages = torch.arange(
                    prefix_len, device=device, dtype=torch.int32
                )
                prefix_kv_indptr = torch.tensor(
                    [0, prefix_len], device=device, dtype=torch.int32
                )
                prefix_kv_last_page_len = torch.tensor(
                    [1], device=device, dtype=torch.int32
                )
                paged_prefix = BatchPrefillWithPagedKVCacheWrapper(
                    torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device),
                    kv_layout="NHD",
                )
                paged_prefix.plan(
                    paged_qo_indptr, prefix_kv_indptr, prefix_only_pages,
                    prefix_kv_last_page_len,
                    num_qo_heads, num_kv_heads, head_dim, page_size,
                    causal=False,
                    q_data_type=dtype, kv_data_type=dtype,
                )
                prefix_only_ms = time_kernel(
                    lambda: paged_prefix.run(q, kv_data), warmup, repeat
                )

                # ============================================================
                # 4. 2-Level Cascade: shared prefix + per-token ancestor chain
                # ============================================================
                casc_qo_indptr_l0 = torch.tensor(
                    [0, num_tokens], device=device, dtype=torch.int32
                )
                casc_kv_indptr_l0 = torch.tensor(
                    [0, prefix_len], device=device, dtype=torch.int32
                )
                casc_kv_indices_l0 = torch.arange(
                    prefix_len, device=device, dtype=torch.int32
                )
                casc_kv_last_page_len_l0 = torch.tensor(
                    [1], device=device, dtype=torch.int32
                )

                casc_qo_indptr_l1 = torch.arange(
                    num_tokens + 1, device=device, dtype=torch.int32
                )
                ancestor_page_indices = []
                kv_indptr_l1_list = [0]
                kv_last_page_len_l1_list = []
                for i in range(num_tokens):
                    anc_sorted = sorted(ancestors[i])
                    pages = [prefix_len + a for a in anc_sorted]
                    ancestor_page_indices.extend(pages)
                    kv_indptr_l1_list.append(kv_indptr_l1_list[-1] + len(pages))
                    kv_last_page_len_l1_list.append(1)

                casc_kv_indptr_l1 = torch.tensor(
                    kv_indptr_l1_list, device=device, dtype=torch.int32
                )
                casc_kv_indices_l1 = torch.tensor(
                    ancestor_page_indices, device=device, dtype=torch.int32
                )
                casc_kv_last_page_len_l1 = torch.tensor(
                    kv_last_page_len_l1_list, device=device, dtype=torch.int32
                )

                cascade_wrapper = MultiLevelCascadeAttentionWrapper(
                    num_levels=2,
                    float_workspace_buffer=torch.zeros(
                        128 * 1024 * 1024, dtype=torch.uint8, device=device
                    ),
                    kv_layout="NHD",
                )
                cascade_wrapper.plan(
                    qo_indptr_arr=[casc_qo_indptr_l0, casc_qo_indptr_l1],
                    paged_kv_indptr_arr=[casc_kv_indptr_l0, casc_kv_indptr_l1],
                    paged_kv_indices_arr=[casc_kv_indices_l0, casc_kv_indices_l1],
                    paged_kv_last_page_len=[casc_kv_last_page_len_l0, casc_kv_last_page_len_l1],
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    page_size=page_size,
                    causal=False,
                    q_data_type=dtype, kv_data_type=dtype,
                )
                cascade_ms = time_kernel(
                    lambda: cascade_wrapper.run(q, kv_data), warmup, repeat
                )

                # Correctness: paged+mask vs cascade
                paged_out = paged_masked.run(q, kv_data)
                cascade_out = cascade_wrapper.run(q, kv_data)
                max_diff = (paged_out - cascade_out).abs().max().item()

                mask_overhead = masked_ms - nomask_ms
                cascade_vs_nomask = nomask_ms / cascade_ms if cascade_ms > 0 else float("inf")
                speedup = masked_ms / cascade_ms if cascade_ms > 0 else float("inf")

                results.append({
                    "topk": topk,
                    "depth": depth,
                    "vtoks": num_tokens,
                    "prefix": prefix_len,
                    "masked_ms": masked_ms,
                    "nomask_ms": nomask_ms,
                    "prefix_only_ms": prefix_only_ms,
                    "cascade_ms": cascade_ms,
                    "mask_overhead_ms": mask_overhead,
                    "speedup": speedup,
                    "cascade_vs_nomask": cascade_vs_nomask,
                    "max_diff": max_diff,
                })

                print(
                    f"  prefix={prefix_len:>6d}  "
                    f"masked={masked_ms:.3f}  "
                    f"nomask={nomask_ms:.3f}  "
                    f"prefix_only={prefix_only_ms:.3f}  "
                    f"cascade={cascade_ms:.3f}  "
                    f"mask_oh={mask_overhead:+.3f}ms  "
                    f"speedup={speedup:.2f}x  "
                    f"diff={max_diff:.1e}"
                )

    # Summary table
    hdr = (
        f"{'topk':>4s} {'dep':>3s} {'vtk':>3s} {'prefix':>6s} "
        f"{'masked':>7s} {'nomask':>7s} {'pfx_only':>8s} {'cascade':>7s} "
        f"{'mask_oh':>7s} {'spdup':>5s} {'c/nm':>5s} {'diff':>8s}"
    )
    print(f"\n{'='*len(hdr)}")
    print(hdr)
    print(f"{'-'*len(hdr)}")
    for r in results:
        print(
            f"{r['topk']:>4d} {r['depth']:>3d} {r['vtoks']:>3d} "
            f"{r['prefix']:>6d} "
            f"{r['masked_ms']:>7.3f} {r['nomask_ms']:>7.3f} "
            f"{r['prefix_only_ms']:>8.3f} {r['cascade_ms']:>7.3f} "
            f"{r['mask_overhead_ms']:>+7.3f} {r['speedup']:>5.2f} "
            f"{r['cascade_vs_nomask']:>5.2f} {r['max_diff']:>8.1e}"
        )
    print(
        f"\nColumns: masked=paged+tree_mask, nomask=paged no mask, "
        f"pfx_only=paged prefix KV only,\n"
        f"         cascade=2-level, mask_oh=masked-nomask, "
        f"spdup=masked/cascade, c/nm=nomask/cascade"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: paged attention (tree mask) vs cascade for EAGLE tree verify"
    )
    parser.add_argument(
        "--prefix-lens", type=str, default="1024,4096,16384,65536",
        help="Comma-separated prefix lengths in tokens",
    )
    parser.add_argument(
        "--topks", type=str, default="4,16",
        help="Comma-separated tree widths (topk values)",
    )
    parser.add_argument(
        "--depths", type=str, default="6,10",
        help="Comma-separated tree depths",
    )
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    prefix_lens = [int(x) for x in args.prefix_lens.split(",")]
    topks = [int(x) for x in args.topks.split(",")]
    depths = [int(x) for x in args.depths.split(",")]

    print(f"Config: heads={args.num_qo_heads}/{args.num_kv_heads}, "
          f"head_dim={args.head_dim}, page_size=1, dtype=bfloat16")

    bench_tree_verify(
        prefix_lens=prefix_lens,
        topks=topks,
        depths=depths,
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        warmup=args.warmup,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    main()
