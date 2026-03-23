"""Profile cascade draft attention to find remaining sync-point overhead.

Runs the full init_forward_metadata + forward_decode path (same as E2E)
with CUDA graphs disabled, using torch.profiler to capture a Chrome trace.

Usage:
    # Generate trace (opens in chrome://tracing)
    uv run python tests/profile_cascade_draft.py

    # Custom parameters
    uv run python tests/profile_cascade_draft.py --topk 4 --num-seqs 4 --prefix-len 2048

    # Compare flat vs cascade overhead
    uv run python tests/profile_cascade_draft.py --mode both
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock

import torch
import torch.profiler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "3rdparty", "flashinfer"),
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "3rdparty", "sglang", "python"),
)

from sglang.srt.speculative.cascade_index_gen import (
    build_shared_indices,
    build_unique_indices,
)
from flashinfer.attention import CascadeBatchAttentionWrapper as CascadeBatchAttention
import flashinfer


@dataclass
class ProfileConfig:
    num_seqs: int = 4
    topk: int = 4
    num_steps: int = 5
    prefix_len: int = 2048
    num_qo_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    page_size: int = 1
    dtype: torch.dtype = torch.bfloat16
    warmup: int = 10
    active: int = 5
    trace_dir: str = "traces"


def make_fake_data(cfg: ProfileConfig):
    """Create fake paged KV cache and request pool matching EAGLE layout."""
    stride = cfg.prefix_len + cfg.topk * cfg.num_steps
    pool_len = stride
    total_pages = cfg.num_seqs * stride

    # req_to_token: [num_seqs, pool_len] mapping request slots to page indices
    req_to_token = torch.zeros(
        cfg.num_seqs, pool_len, dtype=torch.int32, device="cuda"
    )
    for r in range(cfg.num_seqs):
        base = r * stride
        # Prefix pages
        req_to_token[r, :cfg.prefix_len] = torch.arange(
            base, base + cfg.prefix_len, dtype=torch.int32
        )
        # Draft pages: topk branches * num_steps
        for k in range(cfg.topk):
            start_slot = cfg.prefix_len + k * cfg.num_steps
            start_page = base + cfg.prefix_len + k * cfg.num_steps
            req_to_token[r, start_slot : start_slot + cfg.num_steps] = torch.arange(
                start_page, start_page + cfg.num_steps, dtype=torch.int32
            )

    req_pool_indices = torch.arange(cfg.num_seqs, dtype=torch.int32, device="cuda")
    seq_lens = torch.full(
        (cfg.num_seqs,), cfg.prefix_len, dtype=torch.int32, device="cuda"
    )

    # Paged KV cache: [total_pages, 2, page_size, num_kv_heads, head_dim]
    kv_data = torch.randn(
        total_pages, 2, cfg.page_size, cfg.num_kv_heads, cfg.head_dim,
        device="cuda", dtype=cfg.dtype,
    )

    total_branches = cfg.num_seqs * cfg.topk
    q = torch.randn(
        total_branches, cfg.num_qo_heads, cfg.head_dim,
        device="cuda", dtype=cfg.dtype,
    )

    return req_to_token, req_pool_indices, seq_lens, kv_data, q, pool_len


def profile_cascade(cfg: ProfileConfig):
    """Profile the cascade path: index gen + plan + run."""
    req_to_token, req_pool_indices, seq_lens, kv_data, q, pool_len = make_fake_data(cfg)
    num_seqs = cfg.num_seqs
    total_branches = num_seqs * cfg.topk

    # Pre-build cascade wrappers (one per step, like CascadeMultiStepDraftBackend)
    wrappers: List[CascadeBatchAttention] = []
    for _ in range(cfg.num_steps - 1):
        wrappers.append(
            CascadeBatchAttention(num_levels=2, kv_layout="NHD", device="cuda")
        )

    # Pre-compute deterministic qo_indptr tensors on CPU
    qo_indptr_shared_cpu = torch.arange(
        0, (num_seqs + 1) * cfg.topk, cfg.topk,
        dtype=torch.int32, device="cpu",
    )
    qo_indptr_unique_cpu = torch.arange(
        0, total_branches + 1, dtype=torch.int32, device="cpu",
    )

    max_total_prefix = num_seqs * pool_len

    def one_iteration():
        """Simulate one EAGLE draft iteration (index gen + plan + run for all steps)."""
        # --- Index generation (GPU) ---
        with torch.profiler.record_function("cascade::build_shared_indices"):
            kv_indptr_shared, kv_indices_shared, kv_len_shared = build_shared_indices(
                req_pool_indices, req_to_token, seq_lens,
                torch.device("cuda"), pool_len,
                max_total_prefix=max_total_prefix,
            )

        all_kv_indices_unique = []
        all_kv_indptr_unique = []
        all_kv_len_unique = []
        for i in range(cfg.num_steps - 1):
            step_offset = i + 1
            with torch.profiler.record_function(f"cascade::build_unique_indices[{i}]"):
                kv_indptr_u, kv_indices_u, kv_len_u = build_unique_indices(
                    req_pool_indices, req_to_token, seq_lens,
                    cfg.topk, step_offset, cfg.num_steps,
                    cfg.page_size, torch.device("cuda"), pool_len,
                )
            all_kv_indices_unique.append(kv_indices_u)
            all_kv_indptr_unique.append(kv_indptr_u)
            all_kv_len_unique.append(kv_len_u)

        # --- Batch GPU->CPU transfer ---
        with torch.profiler.record_function("cascade::batch_gpu_to_cpu"):
            kv_indptr_shared_cpu = kv_indptr_shared.to("cpu", non_blocking=True)
            kv_len_shared_cpu = kv_len_shared.to("cpu", non_blocking=True)
            all_kv_indptr_unique_cpu = [t.to("cpu", non_blocking=True) for t in all_kv_indptr_unique]
            all_kv_len_unique_cpu = [t.to("cpu", non_blocking=True) for t in all_kv_len_unique]

        with torch.profiler.record_function("cascade::sync_after_transfer"):
            torch.cuda.synchronize()

        # Slice to actual size
        actual_total_prefix = int(kv_indptr_shared_cpu[-1].item())
        kv_indices_shared = kv_indices_shared[:actual_total_prefix]

        # --- Plan + Run per step ---
        for i in range(cfg.num_steps - 1):
            wrapper = wrappers[i]
            with torch.profiler.record_function(f"cascade::plan[{i}]"):
                wrapper.fast_cascade_plan(
                    qo_indptr_host_arr=[qo_indptr_shared_cpu, qo_indptr_unique_cpu],
                    kv_indptr_host_arr=[kv_indptr_shared_cpu, all_kv_indptr_unique_cpu[i]],
                    kv_indices_arr=[kv_indices_shared, all_kv_indices_unique[i]],
                    kv_len_host_arr=[kv_len_shared_cpu, all_kv_len_unique_cpu[i]],
                    num_qo_heads=cfg.num_qo_heads,
                    num_kv_heads=cfg.num_kv_heads,
                    head_dim_qk=cfg.head_dim,
                    head_dim_vo=cfg.head_dim,
                    page_size=cfg.page_size,
                    causal=False,
                    q_data_type=cfg.dtype,
                    kv_data_type=cfg.dtype,
                )
            with torch.profiler.record_function(f"cascade::run[{i}]"):
                wrapper.run(q, kv_data)

    # Initial plan() call to initialize self.module in each wrapper
    print("Initializing wrappers with plan()...")
    kv_indptr_shared, kv_indices_shared, kv_len_shared = build_shared_indices(
        req_pool_indices, req_to_token, seq_lens,
        torch.device("cuda"), pool_len,
    )
    for i in range(cfg.num_steps - 1):
        step_offset = i + 1
        kv_indptr_u, kv_indices_u, kv_len_u = build_unique_indices(
            req_pool_indices, req_to_token, seq_lens,
            cfg.topk, step_offset, cfg.num_steps,
            cfg.page_size, torch.device("cuda"), pool_len,
        )
        wrappers[i].plan(
            qo_indptr_arr=[
                torch.arange(0, (num_seqs + 1) * cfg.topk, cfg.topk, dtype=torch.int32, device="cuda"),
                torch.arange(0, total_branches + 1, dtype=torch.int32, device="cuda"),
            ],
            kv_indptr_arr=[kv_indptr_shared, kv_indptr_u],
            kv_indices_arr=[kv_indices_shared, kv_indices_u],
            kv_len_arr=[kv_len_shared, kv_len_u],
            num_qo_heads=cfg.num_qo_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim_qk=cfg.head_dim,
            head_dim_vo=cfg.head_dim,
            page_size=cfg.page_size,
            causal=False,
            q_data_type=cfg.dtype,
            kv_data_type=cfg.dtype,
        )

    # Warmup
    print(f"Warming up ({cfg.warmup} iters)...")
    for _ in range(cfg.warmup):
        one_iteration()
    torch.cuda.synchronize()

    # Profile
    trace_path = os.path.join(cfg.trace_dir, "cascade_draft")
    os.makedirs(cfg.trace_dir, exist_ok=True)
    print(f"Profiling ({cfg.active} iters)...")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0, warmup=1, active=cfg.active, repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
        record_shapes=True,
        with_stack=True,
        profile_memory=False,
    ) as prof:
        for _ in range(1 + cfg.active):
            one_iteration()
            prof.step()

    # Print summary
    print(f"\nTrace saved to: {trace_path}/")
    print("Open with: tensorboard --logdir traces/  OR  chrome://tracing\n")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    return prof


def profile_flat(cfg: ProfileConfig):
    """Profile the flat (baseline) path for comparison."""
    req_to_token, req_pool_indices, seq_lens, kv_data, q, pool_len = make_fake_data(cfg)
    num_seqs = cfg.num_seqs
    total_branches = num_seqs * cfg.topk

    # Flat: one wrapper handling all branches with full prefix+suffix
    flat_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
    )

    stride = cfg.prefix_len + cfg.topk * cfg.num_steps

    def build_flat_indices(step_offset):
        total_kv = cfg.prefix_len + step_offset
        indices_list = []
        for r in range(num_seqs):
            base = r * stride
            prefix = torch.arange(base, base + cfg.prefix_len, device="cuda", dtype=torch.int32)
            for k in range(cfg.topk):
                suffix = torch.arange(
                    base + cfg.prefix_len + k * cfg.num_steps,
                    base + cfg.prefix_len + k * cfg.num_steps + step_offset,
                    device="cuda", dtype=torch.int32,
                )
                indices_list.append(torch.cat([prefix, suffix]))
        return torch.cat(indices_list), total_kv

    def one_iteration_flat():
        for i in range(cfg.num_steps - 1):
            step_offset = i + 1
            with torch.profiler.record_function(f"flat::build_indices[{i}]"):
                flat_kv_indices, total_kv = build_flat_indices(step_offset)
                flat_kv_indptr = torch.arange(
                    0, (total_branches + 1) * total_kv, total_kv,
                    device="cuda", dtype=torch.int32,
                )
                flat_kv_len = torch.full(
                    (total_branches,), total_kv, device="cuda", dtype=torch.int32,
                )
                flat_last_page = torch.ones(total_branches, device="cuda", dtype=torch.int32)

            with torch.profiler.record_function(f"flat::plan[{i}]"):
                flat_wrapper.plan(
                    flat_kv_indptr, flat_kv_indices, flat_last_page,
                    cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim, cfg.page_size,
                    q_data_type=cfg.dtype,
                )
            with torch.profiler.record_function(f"flat::run[{i}]"):
                flat_wrapper.run(q, kv_data)

    print(f"Warming up flat ({cfg.warmup} iters)...")
    for _ in range(cfg.warmup):
        one_iteration_flat()
    torch.cuda.synchronize()

    trace_path = os.path.join(cfg.trace_dir, "flat_draft")
    os.makedirs(cfg.trace_dir, exist_ok=True)
    print(f"Profiling flat ({cfg.active} iters)...")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0, warmup=1, active=cfg.active, repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
        record_shapes=True,
        with_stack=True,
        profile_memory=False,
    ) as prof:
        for _ in range(1 + cfg.active):
            one_iteration_flat()
            prof.step()

    print(f"\nFlat trace saved to: {trace_path}/")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    return prof


def compare_timing(cfg: ProfileConfig):
    """Quick CUDA event timing comparison (no profiler overhead)."""
    req_to_token, req_pool_indices, seq_lens, kv_data, q, pool_len = make_fake_data(cfg)
    num_seqs = cfg.num_seqs
    total_branches = num_seqs * cfg.topk

    # --- Setup cascade ---
    wrappers = []
    for _ in range(cfg.num_steps - 1):
        wrappers.append(CascadeBatchAttention(num_levels=2, kv_layout="NHD", device="cuda"))

    qo_indptr_shared_cpu = torch.arange(
        0, (num_seqs + 1) * cfg.topk, cfg.topk, dtype=torch.int32, device="cpu",
    )
    qo_indptr_unique_cpu = torch.arange(
        0, total_branches + 1, dtype=torch.int32, device="cpu",
    )
    max_total_prefix = num_seqs * pool_len

    # Init wrappers
    kv_indptr_s, kv_indices_s, kv_len_s = build_shared_indices(
        req_pool_indices, req_to_token, seq_lens, torch.device("cuda"), pool_len,
    )
    for i in range(cfg.num_steps - 1):
        kv_indptr_u, kv_indices_u, kv_len_u = build_unique_indices(
            req_pool_indices, req_to_token, seq_lens,
            cfg.topk, i + 1, cfg.num_steps, cfg.page_size, torch.device("cuda"), pool_len,
        )
        wrappers[i].plan(
            qo_indptr_arr=[
                torch.arange(0, (num_seqs + 1) * cfg.topk, cfg.topk, dtype=torch.int32, device="cuda"),
                torch.arange(0, total_branches + 1, dtype=torch.int32, device="cuda"),
            ],
            kv_indptr_arr=[kv_indptr_s, kv_indptr_u],
            kv_indices_arr=[kv_indices_s, kv_indices_u],
            kv_len_arr=[kv_len_s, kv_len_u],
            num_qo_heads=cfg.num_qo_heads, num_kv_heads=cfg.num_kv_heads,
            head_dim_qk=cfg.head_dim, head_dim_vo=cfg.head_dim,
            page_size=cfg.page_size, causal=False,
            q_data_type=cfg.dtype, kv_data_type=cfg.dtype,
        )

    def cascade_iter():
        kv_indptr_shared, kv_indices_shared, kv_len_shared = build_shared_indices(
            req_pool_indices, req_to_token, seq_lens,
            torch.device("cuda"), pool_len, max_total_prefix=max_total_prefix,
        )
        all_u = []
        for i in range(cfg.num_steps - 1):
            all_u.append(build_unique_indices(
                req_pool_indices, req_to_token, seq_lens,
                cfg.topk, i + 1, cfg.num_steps, cfg.page_size, torch.device("cuda"), pool_len,
            ))
        kv_indptr_shared_cpu = kv_indptr_shared.to("cpu", non_blocking=True)
        kv_len_shared_cpu = kv_len_shared.to("cpu", non_blocking=True)
        u_indptr_cpu = [t[0].to("cpu", non_blocking=True) for t in all_u]
        u_len_cpu = [t[2].to("cpu", non_blocking=True) for t in all_u]
        torch.cuda.synchronize()
        actual = int(kv_indptr_shared_cpu[-1].item())
        kv_indices_shared_sliced = kv_indices_shared[:actual]
        for i in range(cfg.num_steps - 1):
            wrappers[i].fast_cascade_plan(
                qo_indptr_host_arr=[qo_indptr_shared_cpu, qo_indptr_unique_cpu],
                kv_indptr_host_arr=[kv_indptr_shared_cpu, u_indptr_cpu[i]],
                kv_indices_arr=[kv_indices_shared_sliced, all_u[i][1]],
                kv_len_host_arr=[kv_len_shared_cpu, u_len_cpu[i]],
                num_qo_heads=cfg.num_qo_heads, num_kv_heads=cfg.num_kv_heads,
                head_dim_qk=cfg.head_dim, head_dim_vo=cfg.head_dim,
                page_size=cfg.page_size, causal=False,
                q_data_type=cfg.dtype, kv_data_type=cfg.dtype,
            )
            wrappers[i].run(q, kv_data)

    # --- Setup flat ---
    stride = cfg.prefix_len + cfg.topk * cfg.num_steps
    flat_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD",
    )

    def flat_iter():
        for i in range(cfg.num_steps - 1):
            step_offset = i + 1
            total_kv = cfg.prefix_len + step_offset
            idx = []
            for r in range(num_seqs):
                base = r * stride
                pre = torch.arange(base, base + cfg.prefix_len, device="cuda", dtype=torch.int32)
                for k in range(cfg.topk):
                    suf = torch.arange(
                        base + cfg.prefix_len + k * cfg.num_steps,
                        base + cfg.prefix_len + k * cfg.num_steps + step_offset,
                        device="cuda", dtype=torch.int32,
                    )
                    idx.append(torch.cat([pre, suf]))
            flat_kv_indices = torch.cat(idx)
            flat_kv_indptr = torch.arange(0, (total_branches + 1) * total_kv, total_kv, device="cuda", dtype=torch.int32)
            flat_last = torch.ones(total_branches, device="cuda", dtype=torch.int32)
            flat_wrapper.plan(
                flat_kv_indptr, flat_kv_indices, flat_last,
                cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim, cfg.page_size,
                q_data_type=cfg.dtype,
            )
            flat_wrapper.run(q, kv_data)

    # Warmup
    for _ in range(cfg.warmup):
        cascade_iter()
        flat_iter()
    torch.cuda.synchronize()

    # Time
    repeat = 50
    for name, fn in [("cascade", cascade_iter), ("flat", flat_iter)]:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        for i in range(repeat):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times = sorted([s.elapsed_time(e) for s, e in zip(starts, ends)])
        median = times[len(times) // 2]
        p10 = times[len(times) // 10]
        p90 = times[9 * len(times) // 10]
        print(f"{name:>8}: median={median:.3f}ms  p10={p10:.3f}ms  p90={p90:.3f}ms")


def main():
    parser = argparse.ArgumentParser(description="Profile cascade draft attention overhead")
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--prefix-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--active", type=int, default=5)
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument(
        "--mode", choices=["cascade", "flat", "both", "compare"],
        default="cascade",
        help="cascade: profile cascade path; flat: profile flat path; "
             "both: profile both; compare: quick timing comparison",
    )
    args = parser.parse_args()

    cfg = ProfileConfig(
        num_seqs=args.num_seqs,
        topk=args.topk,
        num_steps=args.num_steps,
        prefix_len=args.prefix_len,
        warmup=args.warmup,
        active=args.active,
        trace_dir=args.trace_dir,
    )

    print(f"Config: num_seqs={cfg.num_seqs}, topk={cfg.topk}, num_steps={cfg.num_steps}, "
          f"prefix_len={cfg.prefix_len}")
    print(f"  branches={cfg.num_seqs * cfg.topk}, "
          f"heads={cfg.num_qo_heads}/{cfg.num_kv_heads}, head_dim={cfg.head_dim}\n")

    if args.mode in ("cascade", "both"):
        print("=" * 60)
        print("PROFILING CASCADE PATH")
        print("=" * 60)
        profile_cascade(cfg)

    if args.mode in ("flat", "both"):
        print("=" * 60)
        print("PROFILING FLAT PATH")
        print("=" * 60)
        profile_flat(cfg)

    if args.mode == "compare":
        print("=" * 60)
        print("TIMING COMPARISON (no profiler overhead)")
        print("=" * 60)
        compare_timing(cfg)


if __name__ == "__main__":
    main()
