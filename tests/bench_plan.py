"""Benchmark plan overhead: cascade_plan vs fast_decode_plan vs plan-once-copy.

Measures the raw planning cost without any model forward pass, to isolate
the scheduling overhead that contributes to the draft time gap.

Run:
    CUDA_VISIBLE_DEVICES=1 uv run python tests/bench_plan.py
    CUDA_VISIBLE_DEVICES=1 uv run python tests/bench_plan.py --prefix-len 50000 --topk 2 --num-steps 5
"""

import argparse
import time

import torch
from flashinfer.attention import CascadeBatchAttentionWrapper
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper


def ceil_div(a, b):
    return (a + b - 1) // b


def bench_fn(fn, warmup=10, iters=200):
    """Time a function using cuda synchronize barriers."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / iters * 1000  # ms


def build_cascade_tensors(num_seqs, topk, prefix_len, suffix_len, page_size=1):
    """Build CPU/GPU tensors for cascade plan calls."""
    total_branches = num_seqs * topk
    num_prefix_pages = ceil_div(prefix_len, page_size)
    num_suffix_pages = ceil_div(suffix_len, page_size)

    # Level 0 (shared prefix)
    shared_kv_indices = torch.cat([
        torch.arange(s * num_prefix_pages, (s + 1) * num_prefix_pages, dtype=torch.int32, device="cuda")
        for s in range(num_seqs)
    ])
    shared_kv_indptr_cpu = torch.arange(
        0, (num_seqs + 1) * num_prefix_pages, num_prefix_pages, dtype=torch.int32
    )[:num_seqs + 1]
    shared_kv_len_cpu = torch.full((num_seqs,), prefix_len, dtype=torch.int32)
    shared_qo_indptr_cpu = torch.arange(
        0, (num_seqs + 1) * topk, topk, dtype=torch.int32
    )[:num_seqs + 1]

    # Level 1 (unique suffix)
    base_offset = num_seqs * num_prefix_pages
    unique_kv_indices = torch.cat([
        torch.arange(base_offset + b * num_suffix_pages,
                      base_offset + (b + 1) * num_suffix_pages,
                      dtype=torch.int32, device="cuda")
        for b in range(total_branches)
    ]) if num_suffix_pages > 0 else torch.empty(0, dtype=torch.int32, device="cuda")
    unique_kv_indptr_cpu = torch.arange(
        0, (total_branches + 1) * num_suffix_pages, num_suffix_pages, dtype=torch.int32
    )[:total_branches + 1]
    unique_kv_len_cpu = torch.full((total_branches,), suffix_len, dtype=torch.int32)
    unique_qo_indptr_cpu = torch.arange(total_branches + 1, dtype=torch.int32)

    return {
        "shared_kv_indices": shared_kv_indices,
        "unique_kv_indices": unique_kv_indices,
        "shared_kv_indptr_cpu": shared_kv_indptr_cpu,
        "unique_kv_indptr_cpu": unique_kv_indptr_cpu,
        "shared_kv_len_cpu": shared_kv_len_cpu,
        "unique_kv_len_cpu": unique_kv_len_cpu,
        "shared_qo_indptr_cpu": shared_qo_indptr_cpu,
        "unique_qo_indptr_cpu": unique_qo_indptr_cpu,
    }


def build_flat_tensors(num_seqs, topk, prefix_len, suffix_len, page_size=1):
    """Build tensors for flat (BatchDecode) plan calls."""
    total_branches = num_seqs * topk
    total_kv_len = prefix_len + suffix_len
    num_pages = ceil_div(total_kv_len, page_size)

    kv_indptr = torch.arange(
        0, (total_branches + 1) * num_pages, num_pages, dtype=torch.int32, device="cuda"
    )[:total_branches + 1]
    kv_indptr_cpu = kv_indptr.cpu()
    kv_indices = torch.arange(total_branches * num_pages, dtype=torch.int32, device="cuda")
    last_page_len = torch.ones(total_branches, dtype=torch.int32, device="cuda")

    return {
        "kv_indptr": kv_indptr,
        "kv_indptr_cpu": kv_indptr_cpu,
        "kv_indices": kv_indices,
        "last_page_len": last_page_len,
    }


def bench_cascade_plan(num_seqs, topk, prefix_len, num_steps, num_qo_heads, num_kv_heads, head_dim, iters):
    """Benchmark: N-1 cascade_plan calls (old per-step approach)."""
    max_depth = num_steps - 1
    results = {}

    # --- Per-step cascade_plan (old approach) ---
    wrappers = []
    for _ in range(max_depth):
        w = CascadeBatchAttentionWrapper(num_levels=2, kv_layout="NHD", device="cuda")
        wrappers.append(w)

    # First call to JIT compile
    tensors_max = build_cascade_tensors(num_seqs, topk, prefix_len, max_depth)
    wrappers[0].plan(
        qo_indptr_arr=[tensors_max["shared_qo_indptr_cpu"], tensors_max["unique_qo_indptr_cpu"]],
        kv_indptr_arr=[tensors_max["shared_kv_indptr_cpu"], tensors_max["unique_kv_indptr_cpu"]],
        kv_indices_arr=[tensors_max["shared_kv_indices"], tensors_max["unique_kv_indices"]],
        kv_len_arr=[tensors_max["shared_kv_len_cpu"], tensors_max["unique_kv_len_cpu"]],
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim, head_dim_vo=head_dim,
        page_size=1, causal=False,
    )
    # Share module
    for w in wrappers[1:]:
        w.module = wrappers[0].module

    per_step_tensors = []
    for i in range(max_depth):
        per_step_tensors.append(build_cascade_tensors(num_seqs, topk, prefix_len, i + 1))

    def per_step_plan():
        for i in range(max_depth):
            t = per_step_tensors[i]
            wrappers[i].fast_cascade_plan(
                qo_indptr_host_arr=[t["shared_qo_indptr_cpu"], t["unique_qo_indptr_cpu"]],
                kv_indptr_host_arr=[t["shared_kv_indptr_cpu"], t["unique_kv_indptr_cpu"]],
                kv_indices_arr=[t["shared_kv_indices"], t["unique_kv_indices"]],
                kv_len_host_arr=[t["shared_kv_len_cpu"], t["unique_kv_len_cpu"]],
                num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim, head_dim_vo=head_dim,
                page_size=1, causal=False,
            )

    results["cascade_per_step"] = bench_fn(per_step_plan, iters=iters)

    # --- Single cascade_plan (plan-once-copy approach) ---
    def single_plan():
        wrappers[-1].fast_cascade_plan(
            qo_indptr_host_arr=[tensors_max["shared_qo_indptr_cpu"], tensors_max["unique_qo_indptr_cpu"]],
            kv_indptr_host_arr=[tensors_max["shared_kv_indptr_cpu"], tensors_max["unique_kv_indptr_cpu"]],
            kv_indices_arr=[tensors_max["shared_kv_indices"], tensors_max["unique_kv_indices"]],
            kv_len_host_arr=[tensors_max["shared_kv_len_cpu"], tensors_max["unique_kv_len_cpu"]],
            num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim, head_dim_vo=head_dim,
            page_size=1, causal=False,
        )

    results["cascade_single"] = bench_fn(single_plan, iters=iters)

    # --- Single cascade_plan + copy + patch (full plan-once-copy) ---
    # Extract level2 filter once
    TASK1_BASE = 14
    kv_len_byte_offset = wrappers[-1]._plan_info[TASK1_BASE + 4]
    kv_end_byte_offset = wrappers[-1]._plan_info[TASK1_BASE + 7]
    kv_indptr_byte_offset = wrappers[-1]._plan_info[TASK1_BASE + 1]
    work_indptr_byte_offset = wrappers[-1]._plan_info[TASK1_BASE + 9]

    kv_len_start = kv_len_byte_offset // 4
    kv_end_start = kv_end_byte_offset // 4

    page_locked_buf = wrappers[-1].page_locked_int_workspace_buffer.view(torch.int32)
    num_clusters = wrappers[-1]._plan_info[1]
    work_indptr_start = work_indptr_byte_offset // 4
    total_works = int(page_locked_buf[work_indptr_start + num_clusters])

    kv_indptr_start = kv_indptr_byte_offset // 4
    work_kv_indptrs = page_locked_buf[kv_indptr_start : kv_indptr_start + total_works]
    level2_offset = tensors_max["shared_kv_indices"].shape[0]
    level2_mask = work_kv_indptrs >= level2_offset
    level2_indices = torch.where(level2_mask)[0].to(device="cuda")

    def plan_once_copy():
        last = wrappers[-1]
        last.fast_cascade_plan(
            qo_indptr_host_arr=[tensors_max["shared_qo_indptr_cpu"], tensors_max["unique_qo_indptr_cpu"]],
            kv_indptr_host_arr=[tensors_max["shared_kv_indptr_cpu"], tensors_max["unique_kv_indptr_cpu"]],
            kv_indices_arr=[tensors_max["shared_kv_indices"], tensors_max["unique_kv_indices"]],
            kv_len_host_arr=[tensors_max["shared_kv_len_cpu"], tensors_max["unique_kv_len_cpu"]],
            num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim, head_dim_vo=head_dim,
            page_size=1, causal=False,
        )
        for i in range(max_depth - 1):
            wrappers[i].int_workspace_buffer.copy_(last.int_workspace_buffer)
            wrappers[i]._plan_info = list(last._plan_info)
        for i in range(max_depth):
            step_offset = i + 1
            buf = wrappers[i].int_workspace_buffer.view(torch.int32)
            buf[kv_len_start + level2_indices] = step_offset + 1
            buf[kv_end_start + level2_indices] = step_offset

    results["cascade_plan_once_copy"] = bench_fn(plan_once_copy, iters=iters)

    # --- Just the copy + patch part (no plan call) ---
    # Pre-run plan so workspace is populated
    wrappers[-1].fast_cascade_plan(
        qo_indptr_host_arr=[tensors_max["shared_qo_indptr_cpu"], tensors_max["unique_qo_indptr_cpu"]],
        kv_indptr_host_arr=[tensors_max["shared_kv_indptr_cpu"], tensors_max["unique_kv_indptr_cpu"]],
        kv_indices_arr=[tensors_max["shared_kv_indices"], tensors_max["unique_kv_indices"]],
        kv_len_host_arr=[tensors_max["shared_kv_len_cpu"], tensors_max["unique_kv_len_cpu"]],
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim, head_dim_vo=head_dim,
        page_size=1, causal=False,
    )

    def copy_patch_only():
        last = wrappers[-1]
        for i in range(max_depth - 1):
            wrappers[i].int_workspace_buffer.copy_(last.int_workspace_buffer)
            wrappers[i]._plan_info = list(last._plan_info)
        for i in range(max_depth):
            step_offset = i + 1
            buf = wrappers[i].int_workspace_buffer.view(torch.int32)
            buf[kv_len_start + level2_indices] = step_offset + 1
            buf[kv_end_start + level2_indices] = step_offset

    results["cascade_copy_patch_only"] = bench_fn(copy_patch_only, iters=iters)

    # --- Phase 3: fast replay (no plan, no workspace copy, direct patches) ---
    # Task-agnostic: scan both tasks for Level 0 and Level 1 items.
    # The C++ scheduler can assign Level 0 items to either task.
    SHARED_BASE = 2 + 12 + 12  # = 26
    len_kv_chunk_byte_offset = wrappers[-1]._plan_info[SHARED_BASE]
    kv_limit_t0 = int(page_locked_buf[len_kv_chunk_byte_offset // 4])
    kv_limit_t1 = int(page_locked_buf[len_kv_chunk_byte_offset // 4 + 1])
    kv_limit = max(kv_limit_t0, kv_limit_t1)

    shared_offset = tensors_max["shared_kv_indices"].shape[0]

    l0_patches = []  # (kv_len_s, kv_end_s, l0_idx, last_chunk_idx)
    l1_patches = []  # (kv_indptr_s, l1_idx, kv_indptr_base)
    total_l0 = 0
    total_l1 = 0
    total_last_chunk = 0

    for task in range(2):
        task_base = 2 + task * 12
        t_work_indptr_s = wrappers[-1]._plan_info[task_base + 9] // 4
        t_total = int(page_locked_buf[t_work_indptr_s + num_clusters])
        if t_total == 0:
            continue

        t_kv_indptr_s = wrappers[-1]._plan_info[task_base + 1] // 4
        t_kv_indptrs = page_locked_buf[t_kv_indptr_s : t_kv_indptr_s + t_total]

        l0_mask = t_kv_indptrs < shared_offset
        if l0_mask.any():
            t_kv_len_s = wrappers[-1]._plan_info[task_base + 4] // 4
            t_kv_end_s = wrappers[-1]._plan_info[task_base + 7] // 4
            t_kv_start_s = wrappers[-1]._plan_info[task_base + 6] // 4
            l0_idx = torch.where(l0_mask)[0]
            l0_kv_end_arr = page_locked_buf[t_kv_end_s + l0_idx]
            l0_kv_start_arr = page_locked_buf[t_kv_start_s + l0_idx]
            chunk_sizes = l0_kv_end_arr - l0_kv_start_arr
            last_mask = chunk_sizes < kv_limit
            last_chunk = l0_idx[last_mask] if last_mask.any() else l0_idx[:0]
            l0_patches.append((
                t_kv_len_s, t_kv_end_s,
                l0_idx.to(device="cuda"), last_chunk.to(device="cuda"),
            ))
            total_l0 += l0_idx.numel()
            total_last_chunk += last_chunk.numel()

        l1_mask = t_kv_indptrs >= shared_offset
        if l1_mask.any():
            l1_idx = torch.where(l1_mask)[0]
            l1_base = (t_kv_indptrs[l1_idx] - shared_offset).to(device="cuda", dtype=torch.int32)
            l1_patches.append((t_kv_indptr_s, l1_idx.to(device="cuda"), l1_base))
            total_l1 += l1_idx.numel()

    # Shared kv_indices buffer
    total_kv_buf_len = shared_offset + tensors_max["unique_kv_indices"].shape[0]
    kv_indices_buf = torch.empty(total_kv_buf_len + 1024, dtype=torch.int32, device="cuda")

    # Simulate slightly grown prefix for replay
    new_prefix_val = prefix_len + 5
    new_prefix_gpu = torch.tensor(new_prefix_val, dtype=torch.int32, device="cuda")
    new_kv_len_l0_val = new_prefix_val + topk
    new_shared_len = new_prefix_val * num_seqs

    shared_indices = tensors_max["shared_kv_indices"]
    unique_indices = tensors_max["unique_kv_indices"]
    sl = shared_indices.shape[0]
    ul = unique_indices.shape[0]

    # Build combined index tensor for single-scatter per wrapper
    all_idx_parts = []
    all_val_sizes = []  # (count, type) for building values at replay time
    for kv_len_s, kv_end_s, l0_idx, last_chunk in l0_patches:
        all_idx_parts.append(kv_len_s + l0_idx)
        all_val_sizes.append(("l0_kv_len", l0_idx.numel()))
        if last_chunk.numel() > 0:
            all_idx_parts.append(kv_end_s + last_chunk)
            all_val_sizes.append(("l0_kv_end", last_chunk.numel()))
    l1_kv_indptr_bases = []
    for kv_indptr_s, l1_idx, base in l1_patches:
        all_idx_parts.append(kv_indptr_s + l1_idx)
        all_val_sizes.append(("l1_kv_indptr", l1_idx.numel()))
        l1_kv_indptr_bases.append(base)

    all_indices = torch.cat(all_idx_parts) if all_idx_parts else torch.empty(0, dtype=torch.int64, device="cuda")
    total_patches = all_indices.numel()

    # Pre-build values tensor (structure is fixed; only scalar prefix changes per iter)
    all_values = torch.empty(total_patches, dtype=torch.int32, device="cuda")
    offset = 0
    for label, count in all_val_sizes:
        if label == "l0_kv_len":
            all_values[offset : offset + count] = new_kv_len_l0_val
        elif label == "l0_kv_end":
            all_values[offset : offset + count] = new_prefix_val
        elif label == "l1_kv_indptr":
            all_values[offset : offset + count] = l1_kv_indptr_bases[0] + new_shared_len
        offset += count

    def fast_replay():
        kv_indices_buf[:sl].copy_(shared_indices, non_blocking=True)
        kv_indices_buf[sl : sl + ul].copy_(unique_indices, non_blocking=True)
        for w in wrappers:
            buf = w.int_workspace_buffer.view(torch.int32)
            buf[all_indices] = all_values

    results["cascade_fast_replay"] = bench_fn(fast_replay, iters=iters)

    print(f"  Phase 3 info: kv_limit={kv_limit}, L0 works={total_l0}, "
          f"L1 works={total_l1}, last_chunk={total_last_chunk}")

    return results


def bench_flat_plan(num_seqs, topk, prefix_len, num_steps, num_qo_heads, num_kv_heads, head_dim, iters):
    """Benchmark: N-1 fast_decode_plan calls (flat baseline)."""
    from flashinfer.decode import fast_decode_plan
    import functools

    max_depth = num_steps - 1
    total_branches = num_seqs * topk
    results = {}

    wrappers = []
    for i in range(max_depth):
        w = BatchDecodeWithPagedKVCacheWrapper(
            torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
            kv_layout="NHD",
            use_cuda_graph=True,
            paged_kv_indices_buffer=torch.empty(total_branches * (prefix_len + num_steps), dtype=torch.int32, device="cuda"),
            paged_kv_indptr_buffer=torch.empty(total_branches + 1, dtype=torch.int32, device="cuda"),
            paged_kv_last_page_len_buffer=torch.ones(total_branches, dtype=torch.int32, device="cuda"),
        )
        wrappers.append(w)

    flat_tensors = []
    for i in range(max_depth):
        suffix_len = i + 1
        t = build_flat_tensors(num_seqs, topk, prefix_len, suffix_len)
        flat_tensors.append(t)

    # plan() on each wrapper to JIT compile + populate _cached_module
    for i, w in enumerate(wrappers):
        w.plan(
            flat_tensors[i]["kv_indptr"],
            flat_tensors[i]["kv_indices"],
            flat_tensors[i]["last_page_len"],
            num_qo_heads, num_kv_heads, head_dim, 1,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
        )
    # Replace begin_forward with fast_decode_plan for all
    for w in wrappers:
        w.begin_forward = functools.partial(fast_decode_plan, w)

    # Precompute CPU indptrs
    indptrs_cpu = [t["kv_indptr_cpu"] for t in flat_tensors]

    def per_step_flat():
        for i in range(max_depth):
            t = flat_tensors[i]
            wrappers[i].begin_forward(
                t["kv_indptr"],
                t["kv_indices"],
                t["last_page_len"],
                num_qo_heads, num_kv_heads, head_dim, 1,
                q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                non_blocking=True,
                global_override_indptr_cpu=indptrs_cpu[i],
            )

    results["flat_per_step"] = bench_fn(per_step_flat, iters=iters)

    # Single flat plan (for reference)
    def single_flat():
        t = flat_tensors[-1]
        wrappers[-1].begin_forward(
            t["kv_indptr"],
            t["kv_indices"],
            t["last_page_len"],
            num_qo_heads, num_kv_heads, head_dim, 1,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
            non_blocking=True,
            global_override_indptr_cpu=indptrs_cpu[-1],
        )

    results["flat_single"] = bench_fn(single_flat, iters=iters)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark plan overhead")
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--prefix-len", type=int, default=50000)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    print(f"Config: num_seqs={args.num_seqs}, topk={args.topk}, prefix_len={args.prefix_len}, "
          f"num_steps={args.num_steps}, heads={args.num_qo_heads}/{args.num_kv_heads}, "
          f"head_dim={args.head_dim}, iters={args.iters}")
    print()

    # Cascade benchmarks
    cascade = bench_cascade_plan(
        args.num_seqs, args.topk, args.prefix_len, args.num_steps,
        args.num_qo_heads, args.num_kv_heads, args.head_dim, args.iters,
    )

    # Flat benchmarks
    flat = bench_flat_plan(
        args.num_seqs, args.topk, args.prefix_len, args.num_steps,
        args.num_qo_heads, args.num_kv_heads, args.head_dim, args.iters,
    )

    max_depth = args.num_steps - 1
    print(f"{'Method':<40} {'ms/iter':>10} {'calls':>8} {'ms/call':>10}")
    print("-" * 75)
    print(f"{'cascade per-step (old)':<40} {cascade['cascade_per_step']:>10.3f} {max_depth:>8} {cascade['cascade_per_step']/max_depth:>10.3f}")
    print(f"{'cascade single plan':<40} {cascade['cascade_single']:>10.3f} {1:>8} {cascade['cascade_single']:>10.3f}")
    print(f"{'cascade plan-once-copy (Phase 2)':<40} {cascade['cascade_plan_once_copy']:>10.3f} {'1+copy':>8} {'':>10}")
    print(f"{'cascade copy+patch only':<40} {cascade['cascade_copy_patch_only']:>10.3f} {0:>8} {'':>10}")
    print(f"{'cascade fast_replay (Phase 3)':<40} {cascade['cascade_fast_replay']:>10.3f} {0:>8} {'':>10}")
    print(f"{'flat per-step':<40} {flat['flat_per_step']:>10.3f} {max_depth:>8} {flat['flat_per_step']/max_depth:>10.3f}")
    print(f"{'flat single plan':<40} {flat['flat_single']:>10.3f} {1:>8} {flat['flat_single']:>10.3f}")

    print()
    est_iters = 90
    print(f"Estimated over {est_iters} decode iterations:")
    print(f"  cascade per-step (old):   {cascade['cascade_per_step'] * est_iters:>8.1f} ms")
    print(f"  cascade plan-once-copy:   {cascade['cascade_plan_once_copy'] * est_iters:>8.1f} ms")
    print(f"  cascade copy+patch only:  {cascade['cascade_copy_patch_only'] * est_iters:>8.1f} ms")
    print(f"  cascade fast_replay:      {cascade['cascade_fast_replay'] * est_iters:>8.1f} ms")
    print(f"  flat per-step:            {flat['flat_per_step'] * est_iters:>8.1f} ms")


if __name__ == "__main__":
    main()
