"""Reproduce SGLang's exact cascade vs flat kernel dispatch pattern.

Unlike bench_cascade_draft.py which benchmarks a single run() call with
pre-planned wrappers, this script simulates the FULL per-step cycle:
  plan() or fast_cascade_plan() + run()
to match what SGLang actually does during EAGLE draft decode.

This helps isolate whether the kernel itself is slower, or if
plan/sync overhead is the bottleneck.

Usage:
    uv run python tests/bench_cascade_sglang_shapes.py
    uv run python tests/bench_cascade_sglang_shapes.py --topk 10 --prefix-len 8192
    uv run python tests/bench_cascade_sglang_shapes.py --topk 10 --prefix-len 8192 --kernel-only
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


def bench_sglang_pattern(
    num_seqs=1,
    topk=10,
    num_steps=5,
    prefix_len=8192,
    num_qo_heads=32,
    num_kv_heads=8,
    head_dim=128,
    page_size=1,
    dtype=torch.bfloat16,
    warmup=20,
    repeat=100,
    kernel_only=False,
):
    """Simulate SGLang's per-step plan+run cycle for both flat and cascade."""

    total_branches = num_seqs * topk
    stride = prefix_len + topk * num_steps

    # Allocate paged KV cache
    total_pages = num_seqs * stride
    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        device="cuda", dtype=dtype,
    )
    q = torch.randn(
        total_branches, num_qo_heads, head_dim,
        device="cuda", dtype=dtype,
    )

    print(f"\nConfig: num_seqs={num_seqs}, topk={topk}, prefix_len={prefix_len}, "
          f"num_steps={num_steps}, branches={total_branches}")
    print(f"  heads={num_qo_heads}/{num_kv_heads}, head_dim={head_dim}, "
          f"page_size={page_size}, dtype={dtype}")
    print(f"  kernel_only={kernel_only}\n")

    results = []

    for step_offset in range(1, num_steps):
        total_kv_per_branch = prefix_len + step_offset

        # === FLAT: BatchDecodeWithPagedKVCacheWrapper ===
        flat_indices_list = []
        for r in range(num_seqs):
            base = r * stride
            prefix_pages = torch.arange(base, base + prefix_len, device="cuda", dtype=torch.int32)
            for k in range(topk):
                suffix_pages = torch.arange(
                    base + prefix_len + k * num_steps,
                    base + prefix_len + k * num_steps + step_offset,
                    device="cuda", dtype=torch.int32,
                )
                flat_indices_list.append(torch.cat([prefix_pages, suffix_pages]))

        flat_kv_indices = torch.cat(flat_indices_list)
        flat_kv_indptr = torch.arange(
            0, (total_branches + 1) * total_kv_per_branch, total_kv_per_branch,
            device="cuda", dtype=torch.int32,
        )
        flat_last_page_len = torch.ones(total_branches, device="cuda", dtype=torch.int32)
        flat_kv_len = torch.full((total_branches,), total_kv_per_branch, device="cuda", dtype=torch.int32)

        flat_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD",
        )

        # Pre-compute CPU indptr (like SGLang's global_override_indptr_cpu)
        flat_kv_indptr_cpu = flat_kv_indptr.cpu()
        flat_kv_len_cpu = flat_kv_len.cpu()

        def flat_plan_and_run():
            flat_wrapper.plan(
                flat_kv_indptr, flat_kv_indices, flat_last_page_len,
                num_qo_heads, num_kv_heads, head_dim, page_size,
                q_data_type=dtype,
            )
            return flat_wrapper.run(q, kv_data)

        def flat_run_only():
            return flat_wrapper.run(q, kv_data)

        # Warm up flat (plan once for kernel_only mode)
        flat_wrapper.plan(
            flat_kv_indptr, flat_kv_indices, flat_last_page_len,
            num_qo_heads, num_kv_heads, head_dim, page_size,
            q_data_type=dtype,
        )
        flat_fn = flat_run_only if kernel_only else flat_plan_and_run
        for _ in range(warmup):
            flat_fn()
        torch.cuda.synchronize()

        start_evts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        end_evts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        for i in range(repeat):
            start_evts[i].record()
            flat_fn()
            end_evts[i].record()
        torch.cuda.synchronize()
        flat_times = [s.elapsed_time(e) for s, e in zip(start_evts, end_evts)]
        flat_median = sorted(flat_times)[len(flat_times) // 2]

        # === CASCADE: CascadeBatchAttentionWrapper ===
        # Shared level (prefix)
        shared_indices_list = []
        for r in range(num_seqs):
            base = r * stride
            shared_indices_list.append(
                torch.arange(base, base + prefix_len, device="cuda", dtype=torch.int32)
            )
        shared_kv_indices = torch.cat(shared_indices_list)
        shared_kv_indptr = torch.arange(
            0, (num_seqs + 1) * prefix_len, prefix_len, device="cuda", dtype=torch.int32,
        )
        shared_kv_len = torch.full((num_seqs,), prefix_len, device="cuda", dtype=torch.int32)
        qo_indptr_shared = torch.arange(
            0, (num_seqs + 1) * topk, topk, device="cuda", dtype=torch.int32,
        )

        # Unique level (suffix)
        unique_indices_list = []
        for r in range(num_seqs):
            base = r * stride
            for k in range(topk):
                unique_indices_list.append(
                    torch.arange(
                        base + prefix_len + k * num_steps,
                        base + prefix_len + k * num_steps + step_offset,
                        device="cuda", dtype=torch.int32,
                    )
                )
        unique_kv_indices = torch.cat(unique_indices_list)
        unique_kv_indptr = torch.arange(
            0, (total_branches + 1) * step_offset, step_offset,
            device="cuda", dtype=torch.int32,
        )
        unique_kv_len = torch.full((total_branches,), step_offset, device="cuda", dtype=torch.int32)
        qo_indptr_unique = torch.arange(0, total_branches + 1, device="cuda", dtype=torch.int32)

        # CPU tensors (like SGLang's cascade backend builds)
        qo_indptr_shared_cpu = qo_indptr_shared.cpu()
        qo_indptr_unique_cpu = qo_indptr_unique.cpu()
        kv_indptr_shared_cpu = shared_kv_indptr.cpu()
        kv_indptr_unique_cpu = unique_kv_indptr.cpu()
        kv_len_shared_cpu = shared_kv_len.to(torch.int32).cpu()
        kv_len_unique_cpu = unique_kv_len.cpu()

        cascade = CascadeBatchAttention(num_levels=2, kv_layout="NHD", device="cuda")

        plan_kwargs = dict(
            num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim, head_dim_vo=head_dim,
            page_size=page_size, causal=False,
            q_data_type=dtype, kv_data_type=dtype,
        )

        # First plan (compiles JIT module)
        cascade.plan(
            qo_indptr_arr=[qo_indptr_shared_cpu, qo_indptr_unique_cpu],
            kv_indptr_arr=[kv_indptr_shared_cpu, kv_indptr_unique_cpu],
            kv_indices_arr=[shared_kv_indices, unique_kv_indices],
            kv_len_arr=[kv_len_shared_cpu, kv_len_unique_cpu],
            **plan_kwargs,
        )

        def cascade_fast_plan_and_run():
            cascade.fast_cascade_plan(
                qo_indptr_host_arr=[qo_indptr_shared_cpu, qo_indptr_unique_cpu],
                kv_indptr_host_arr=[kv_indptr_shared_cpu, kv_indptr_unique_cpu],
                kv_indices_arr=[shared_kv_indices, unique_kv_indices],
                kv_len_host_arr=[kv_len_shared_cpu, kv_len_unique_cpu],
                **plan_kwargs,
            )
            return cascade.run(q, kv_data)

        def cascade_run_only():
            return cascade.run(q, kv_data)

        cascade_fn = cascade_run_only if kernel_only else cascade_fast_plan_and_run
        for _ in range(warmup):
            cascade_fn()
        torch.cuda.synchronize()

        start_evts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        end_evts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        for i in range(repeat):
            start_evts[i].record()
            cascade_fn()
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
            "step": step_offset,
            "flat_ms": flat_median,
            "cascade_ms": cascade_median,
            "speedup": speedup,
            "max_diff": max_diff,
        })

    mode = "kernel-only" if kernel_only else "fast_plan+run"
    print(f"{'='*80}")
    print(f"SGLang-pattern benchmark ({mode})")
    print(f"  num_seqs={num_seqs}, topk={topk}, prefix_len={prefix_len}, "
          f"heads={num_qo_heads}/{num_kv_heads}")
    print(f"{'='*80}")
    print(f"  {'step':>4}  {'flat(ms)':>9}  {'cascade(ms)':>11}  {'speedup':>8}  {'max_diff':>10}")
    print(f"  {'-'*4}  {'-'*9}  {'-'*11}  {'-'*8}  {'-'*10}")
    for r in results:
        print(f"  {r['step']:>4}  {r['flat_ms']:>9.4f}  {r['cascade_ms']:>11.4f}  "
              f"{r['speedup']:>7.2f}x  {r['max_diff']:>10.6f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGLang-pattern cascade vs flat benchmark")
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--prefix-len", type=int, default=8192)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--kernel-only", action="store_true",
                        help="Time only run(), not plan+run")
    args = parser.parse_args()

    bench_sglang_pattern(
        num_seqs=args.num_seqs,
        topk=args.topk,
        num_steps=args.num_steps,
        prefix_len=args.prefix_len,
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        warmup=args.warmup,
        repeat=args.repeat,
        kernel_only=args.kernel_only,
    )
