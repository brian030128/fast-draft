"""Kernel-level microbenchmark: flat decode vs cascade for EAGLE draft decode.

Simulates the draft decode attention pattern:
  - num_seqs requests, each with a prefix of `prefix_len` KV tokens
  - topk branches per request, each with `step_offset` draft suffix tokens
  - Flat approach: topk * num_seqs independent decode calls, each reading full prefix + suffix
  - Cascade approach: read prefix once, combine with per-branch suffix

Default parameters match sglang with TP=4 on Llama-3.1-8B:
  num_qo_heads=8, num_kv_heads=2, head_dim=128, dtype=float16, page_size=1,
  separate K/V buffers (not interleaved).

Usage:
    python tests/bench_cascade_draft.py
    python tests/bench_cascade_draft.py --num-seqs 8 --topk 6
    python tests/bench_cascade_draft.py --prefix-lens 256,1024,4096,16384
    python tests/bench_cascade_draft.py --num-qo-heads 32 --num-kv-heads 8  # full model (no TP)
"""

import argparse
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "3rdparty", "flashinfer"),
)

import flashinfer
from flashinfer.attention import CascadeBatchAttentionWrapper as CascadeBatchAttention


def ceil_div(a, b):
    return (a + b - 1) // b


def bench_eagle_draft_decode(
    num_seqs_list=(1, 4, 8),
    topk=6,
    num_steps=5,
    prefix_lens=(256, 512, 1024, 2048, 4096, 8192, 16384),
    num_qo_heads=8,
    num_kv_heads=2,
    head_dim=128,
    page_size=1,
    dtype=torch.float16,
    warmup=50,
    repeat=200,
):
    results = []

    for num_seqs in num_seqs_list:
        for prefix_len in prefix_lens:
            for step_offset in [1, num_steps - 1]:
                total_branches = num_seqs * topk
                suffix_len = step_offset
                total_kv_per_branch = prefix_len + suffix_len

                # Allocate paged KV cache as separate K and V buffers
                # (matches sglang's memory pool layout)
                total_pages = num_seqs * (prefix_len + topk * num_steps)
                k_data = torch.randn(
                    total_pages, num_kv_heads, head_dim,
                    device="cuda", dtype=dtype,
                )
                v_data = torch.randn(
                    total_pages, num_kv_heads, head_dim,
                    device="cuda", dtype=dtype,
                )
                kv_data = (k_data, v_data)
                q = torch.randn(
                    total_branches, num_qo_heads, head_dim,
                    device="cuda", dtype=dtype,
                )

                # Build page indices
                # For each request r, prefix pages are [r*stride, r*stride+prefix_len)
                # Draft pages for branch k at step s: r*stride + prefix_len + k*num_steps + s
                stride = prefix_len + topk * num_steps

                # --- Flat decode: each branch gets prefix + suffix ---
                flat_indices_list = []
                for r in range(num_seqs):
                    base = r * stride
                    prefix_pages = torch.arange(
                        base, base + prefix_len, device="cuda", dtype=torch.int32
                    )
                    for k in range(topk):
                        suffix_pages = torch.arange(
                            base + prefix_len + k * num_steps,
                            base + prefix_len + k * num_steps + suffix_len,
                            device="cuda", dtype=torch.int32,
                        )
                        flat_indices_list.append(torch.cat([prefix_pages, suffix_pages]))

                flat_kv_indices = torch.cat(flat_indices_list)
                flat_kv_indptr = torch.arange(
                    0, (total_branches + 1) * total_kv_per_branch,
                    total_kv_per_branch,
                    device="cuda", dtype=torch.int32,
                )
                flat_last_page_len = torch.ones(
                    total_branches, device="cuda", dtype=torch.int32
                )
                flat_kv_len = torch.full(
                    (total_branches,), total_kv_per_branch,
                    device="cuda", dtype=torch.int32,
                )

                flat_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
                    "NHD",
                )
                flat_wrapper.plan(
                    flat_kv_indptr, flat_kv_indices, flat_last_page_len,
                    num_qo_heads, num_kv_heads, head_dim, page_size,
                    q_data_type=dtype,
                )

                for _ in range(warmup):
                    flat_wrapper.run(q, kv_data)
                torch.cuda.synchronize()

                start_evts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
                end_evts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
                for i in range(repeat):
                    start_evts[i].record()
                    flat_wrapper.run(q, kv_data)
                    end_evts[i].record()
                torch.cuda.synchronize()
                flat_times = [s.elapsed_time(e) for s, e in zip(start_evts, end_evts)]
                flat_median = sorted(flat_times)[len(flat_times) // 2]

                # --- Cascade: shared prefix + unique suffix ---
                # Shared level: num_seqs entries, each grouping topk Q tokens
                shared_indices_list = []
                for r in range(num_seqs):
                    base = r * stride
                    shared_indices_list.append(
                        torch.arange(base, base + prefix_len, device="cuda", dtype=torch.int32)
                    )
                shared_kv_indices = torch.cat(shared_indices_list)
                shared_kv_indptr = torch.arange(
                    0, (num_seqs + 1) * prefix_len, prefix_len,
                    device="cuda", dtype=torch.int32,
                )
                shared_kv_len = torch.full(
                    (num_seqs,), prefix_len, device="cuda", dtype=torch.int32
                )
                qo_indptr_shared = torch.arange(
                    0, (num_seqs + 1) * topk, topk,
                    device="cuda", dtype=torch.int32,
                )

                # Unique level: total_branches entries, each with suffix_len pages
                unique_indices_list = []
                for r in range(num_seqs):
                    base = r * stride
                    for k in range(topk):
                        unique_indices_list.append(
                            torch.arange(
                                base + prefix_len + k * num_steps,
                                base + prefix_len + k * num_steps + suffix_len,
                                device="cuda", dtype=torch.int32,
                            )
                        )
                unique_kv_indices = torch.cat(unique_indices_list)
                unique_kv_indptr = torch.arange(
                    0, (total_branches + 1) * suffix_len, suffix_len,
                    device="cuda", dtype=torch.int32,
                )
                unique_kv_len = torch.full(
                    (total_branches,), suffix_len, device="cuda", dtype=torch.int32
                )
                qo_indptr_unique = torch.arange(
                    0, total_branches + 1, device="cuda", dtype=torch.int32
                )

                cascade = CascadeBatchAttention(
                    num_levels=2, kv_layout="NHD", device="cuda",
                )
                cascade.plan(
                    qo_indptr_arr=[qo_indptr_shared, qo_indptr_unique],
                    kv_indptr_arr=[shared_kv_indptr, unique_kv_indptr],
                    kv_indices_arr=[shared_kv_indices, unique_kv_indices],
                    kv_len_arr=[shared_kv_len, unique_kv_len],
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim,
                    head_dim_vo=head_dim,
                    page_size=page_size,
                    causal=False,
                    q_data_type=dtype,
                    kv_data_type=dtype,
                )

                for _ in range(warmup):
                    cascade.run(q, kv_data)
                torch.cuda.synchronize()

                start_evts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
                end_evts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
                for i in range(repeat):
                    start_evts[i].record()
                    cascade.run(q, kv_data)
                    end_evts[i].record()
                torch.cuda.synchronize()
                cascade_times = [s.elapsed_time(e) for s, e in zip(start_evts, end_evts)]
                cascade_median = sorted(cascade_times)[len(cascade_times) // 2]

                # Correctness check
                flat_out = flat_wrapper.run(q, kv_data)
                cascade_out, _ = cascade.run(q, kv_data)
                max_diff = (flat_out - cascade_out).abs().max().item()

                speedup = flat_median / cascade_median if cascade_median > 0 else float("inf")
                results.append({
                    "num_seqs": num_seqs,
                    "prefix_len": prefix_len,
                    "step": step_offset,
                    "branches": total_branches,
                    "flat_ms": flat_median,
                    "cascade_ms": cascade_median,
                    "speedup": speedup,
                    "max_diff": max_diff,
                })

    # Print results
    print(f"\n{'='*100}")
    print(f"EAGLE Draft Decode Attention: Flat vs Cascade")
    print(f"  topk={topk}, num_steps={num_steps}, heads={num_qo_heads}/{num_kv_heads}, "
          f"head_dim={head_dim}, page_size={page_size}, dtype={dtype}")
    print(f"{'='*100}")
    print(f"  {'seqs':>4}  {'prefix':>7}  {'step':>4}  {'branches':>8}  "
          f"{'flat(ms)':>9}  {'cascade(ms)':>11}  {'speedup':>8}  {'max_diff':>10}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*11}  {'-'*8}  {'-'*10}")
    for r in results:
        print(f"  {r['num_seqs']:>4}  {r['prefix_len']:>7}  {r['step']:>4}  {r['branches']:>8}  "
              f"{r['flat_ms']:>9.4f}  {r['cascade_ms']:>11.4f}  {r['speedup']:>7.2f}x  {r['max_diff']:>10.6f}")
    print(f"{'='*100}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EAGLE draft decode kernel benchmark")
    parser.add_argument("--num-seqs", default="1,4,8", help="Comma-separated num_seqs values")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--prefix-lens", default="256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--num-qo-heads", type=int, default=8, help="Number of query/output heads (default: 8 for TP=4)")
    parser.add_argument("--num-kv-heads", type=int, default=2, help="Number of KV heads (default: 2 for TP=4)")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    bench_eagle_draft_decode(
        num_seqs_list=[int(x) for x in args.num_seqs.split(",")],
        topk=args.topk,
        num_steps=args.num_steps,
        prefix_lens=[int(x) for x in args.prefix_lens.split(",")],
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        warmup=args.warmup,
        repeat=args.repeat,
    )
