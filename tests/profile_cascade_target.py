"""Profile harness for the cascade attention -32% loss row.

Target row (from tests/bench_tree_attn.py at hq=64/hkv=8):
    num_prefixes=8, prefix_len=2048, topk=16, suffix_len=8 (mid of 1..16 sweep)
    cascade ~ 0.144 ms vs fasttree ~ 0.110 ms (cascade -32%)

This harness:
  1. Builds the workload identically to bench_tree_attn.py.
  2. Plans the cascade wrapper once.
  3. Times cascade.run() and fasttree_decode() with bench_gpu_time using
     CUDA graphs for low-noise wall-clock measurement.
  4. Reports per-call timing breakdown.

Run with nsys to get per-kernel-launch latency:
    nsys profile --stats=true --force-overwrite=true --gpu-metrics-device=all \
        -o /tmp/cascade_target uv run python tests/profile_cascade_target.py
"""

import math
import os
import sys
import statistics

import torch

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "3rdparty", "flashinfer"))
sys.path.insert(0, os.path.join(_root, "3rdparty", "FastTree-Artifact", "kernel_bench"))
sys.path.insert(0, os.path.join(_root, "tests"))

import flashinfer  # noqa: E402
from flashinfer.attention import CascadeBatchAttentionWrapper  # noqa: E402
from flashinfer.testing import bench_gpu_time  # noqa: E402
from fasttree import FastTreeParams, fasttree_preparation, fasttree_decode  # noqa: E402

from bench_tree_attn import (  # noqa: E402
    build_tree_info,
    prepare_data,
    build_flashinfer_cascade_indices,
)


TARGET = dict(
    num_prefixes=8,
    prefix_len=2048,
    topk=16,
    suffix_len=8,           # mid of the 1..16 sweep
    num_qo_heads=64,
    num_kv_heads=8,
    head_dim=128,
    dtype=torch.float16,
)


def _plan_cascade_with(casc_idx, cfg):
    cascade = CascadeBatchAttentionWrapper(num_levels=2, kv_layout="NHD", device="cuda")
    cascade.plan(
        qo_indptr_arr=casc_idx["qo_indptr_arr"],
        kv_indptr_arr=casc_idx["kv_indptr_arr"],
        kv_indices_arr=casc_idx["kv_indices_arr"],
        kv_len_arr=casc_idx["kv_len_arr"],
        num_qo_heads=cfg["num_qo_heads"],
        num_kv_heads=cfg["num_kv_heads"],
        head_dim_qk=cfg["head_dim"],
        head_dim_vo=cfg["head_dim"],
        page_size=1,
        causal=False,
        q_data_type=cfg["dtype"],
        kv_data_type=cfg["dtype"],
    )
    return cascade


def setup_cascade(cfg):
    tree_info = build_tree_info(cfg["num_prefixes"], cfg["prefix_len"], cfg["topk"], cfg["suffix_len"])
    data = prepare_data(tree_info, cfg["num_qo_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["dtype"])
    casc_idx = build_flashinfer_cascade_indices(tree_info, data, cfg["num_prefixes"], cfg["topk"])
    q_casc = casc_idx["q_reorder"]
    cascade = _plan_cascade_with(casc_idx, cfg)
    kv_data = (data["k_tree"], data["v_tree"])
    out_buf = torch.empty_like(q_casc)
    lse_buf = torch.empty(q_casc.shape[0], q_casc.shape[1], device="cuda", dtype=torch.float32)
    # warmup
    cascade.run(q_casc, kv_data, out=out_buf, lse=lse_buf)
    torch.cuda.synchronize()

    def run_cascade():
        cascade.run(q_casc, kv_data, out=out_buf, lse=lse_buf)

    return run_cascade, data, tree_info, casc_idx, q_casc, kv_data, out_buf, lse_buf


def setup_cascade_level_only(level, cfg, tree_info, data, q_casc):
    """Plan a cascade kernel with only one of the two levels populated.

    level=0: only the shared prefix level (drop level 1 by zeroing its qo lengths
             so no level-1 work items are emitted but the level still exists)
    level=1: only the unique suffix level

    Returns a callable that runs cascade.run() over the level-only plan.
    """
    casc_idx = build_flashinfer_cascade_indices(tree_info, data, cfg["num_prefixes"], cfg["topk"])
    qo0, qo1 = casc_idx["qo_indptr_arr"]
    kvip0, kvip1 = casc_idx["kv_indptr_arr"]
    kvix0, kvix1 = casc_idx["kv_indices_arr"]
    kvl0, kvl1 = casc_idx["kv_len_arr"]

    # Empty-level placeholders: 1-request batch with qo_len=0 and kv_len=0.
    # We use single-element sentinels rather than zero-sized tensors because the
    # planner dereferences data_ptr() unconditionally.
    def _empty(other):
        # qo_indptr_arr: 2 entries [0, 0]; kv_indptr_arr: 2 entries [0, 0];
        # kv_indices_arr: 1 entry [0]; kv_len_arr: 1 entry [0]
        empty_qo = torch.zeros(2, dtype=qo0.dtype, device=qo0.device)
        empty_kvip = torch.zeros(2, dtype=kvip0.dtype, device=kvip0.device)
        empty_kvix = torch.zeros(1, dtype=kvix0.dtype, device=kvix0.device)
        empty_kvl = torch.zeros(1, dtype=kvl0.dtype, device=kvl0.device)
        return empty_qo, empty_kvip, empty_kvix, empty_kvl

    if level == 0:
        empty_qo, empty_kvip, empty_kvix, empty_kvl = _empty(qo1)
        casc_idx_new = {
            **casc_idx,
            "qo_indptr_arr": [qo0, empty_qo],
            "kv_indptr_arr": [kvip0, empty_kvip],
            "kv_indices_arr": [kvix0, empty_kvix],
            "kv_len_arr": [kvl0, empty_kvl],
        }
    else:
        empty_qo, empty_kvip, empty_kvix, empty_kvl = _empty(qo0)
        casc_idx_new = {
            **casc_idx,
            "qo_indptr_arr": [empty_qo, qo1],
            "kv_indptr_arr": [empty_kvip, kvip1],
            "kv_indices_arr": [empty_kvix, kvix1],
            "kv_len_arr": [empty_kvl, kvl1],
        }
    cascade = _plan_cascade_with(casc_idx_new, cfg)
    kv_data = (data["k_tree"], data["v_tree"])
    out_buf = torch.empty_like(q_casc)
    lse_buf = torch.empty(q_casc.shape[0], q_casc.shape[1], device="cuda", dtype=torch.float32)
    # warmup
    cascade.run(q_casc, kv_data, out=out_buf, lse=lse_buf)
    torch.cuda.synchronize()

    def run_level_only():
        cascade.run(q_casc, kv_data, out=out_buf, lse=lse_buf)

    return run_level_only


def setup_fasttree(cfg, data, q_orig):
    tree_info_ft = build_tree_info(cfg["num_prefixes"], cfg["prefix_len"], cfg["topk"], cfg["suffix_len"])
    data_ft = prepare_data(tree_info_ft, cfg["num_qo_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["dtype"])
    data_ft["q"].copy_(q_orig)
    data_ft["k_tree"].copy_(data["k_tree"])
    data_ft["v_tree"].copy_(data["v_tree"])

    params = FastTreeParams()
    params.set_values(0.66, 0.33, 0.1)
    params.set_q_tile_sizes([16, 4])
    params.set_kv_tile_sizes([32, 32])
    params.set_kv_group_num(cfg["num_qo_heads"] // cfg["num_kv_heads"])

    ft_aux, _ = fasttree_preparation(
        tree_info_ft,
        data_ft["kv_ptrs"],
        data_ft["batch_size"],
        cfg["num_qo_heads"],
        cfg["num_kv_heads"],
        cfg["head_dim"],
        [1024, 128],
        [132, 528],
        [132, 132],
        params,
    )

    sm_scale = 1.0 / math.sqrt(cfg["head_dim"])
    ft_out = torch.empty(
        data_ft["batch_size"], cfg["num_qo_heads"], cfg["head_dim"],
        device="cuda", dtype=cfg["dtype"],
    )

    def run_fasttree():
        fasttree_decode(
            data_ft["q"], data_ft["k_tree"], data_ft["v_tree"], ft_out,
            *ft_aux, [16, 4], [32, 32], sm_scale,
        )

    run_fasttree()  # warmup
    torch.cuda.synchronize()
    return run_fasttree


def stats(times_ms):
    return {
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "n": len(times_ms),
    }


def fmt_stats(label, s):
    return (
        f"{label:>20}: median={s['median_ms']*1000:7.2f} us  "
        f"mean={s['mean_ms']*1000:7.2f} us  "
        f"stdev={s['stdev_ms']*1000:6.2f} us  "
        f"min={s['min_ms']*1000:7.2f} us  "
        f"max={s['max_ms']*1000:7.2f} us  "
        f"n={s['n']}"
    )


def main():
    cfg = TARGET
    print(f"Profiling target row: {cfg}")
    print()

    run_cascade, data, tree_info, _casc_idx, q_casc, _kv, _out, _lse = setup_cascade(cfg)
    run_fasttree = setup_fasttree(cfg, data, data["q"])

    # CUDA graphs give the most accurate launch-overhead-amortized wall clock
    # available without CUPTI. We also report a non-graph measurement so we
    # can isolate launch overhead.
    print("Backend: CUDA graphs (10 calls per replay, l2-flush via rotating buffers)")
    cas_g = bench_gpu_time(run_cascade, use_cuda_graph=True, num_iters_within_graph=10,
                           dry_run_time_ms=50, repeat_time_ms=200)
    ft_g  = bench_gpu_time(run_fasttree, use_cuda_graph=True, num_iters_within_graph=10,
                           dry_run_time_ms=50, repeat_time_ms=200)
    print(fmt_stats("cascade (graph)", stats(cas_g)))
    print(fmt_stats("fasttree (graph)", stats(ft_g)))
    cas_med = statistics.median(cas_g)
    ft_med = statistics.median(ft_g)
    pct = (cas_med - ft_med) / ft_med * 100
    print(f"\n  cascade vs fasttree (graph median): {pct:+.1f}%  "
          f"(cascade {cas_med*1000:.2f} us, fasttree {ft_med*1000:.2f} us)")
    print()

    print("Backend: CUDA events (default), individual launches")
    cas_e = bench_gpu_time(run_cascade, dry_run_time_ms=50, repeat_time_ms=200)
    ft_e  = bench_gpu_time(run_fasttree, dry_run_time_ms=50, repeat_time_ms=200)
    print(fmt_stats("cascade (events)", stats(cas_e)))
    print(fmt_stats("fasttree (events)", stats(ft_e)))
    cas_med_e = statistics.median(cas_e)
    ft_med_e = statistics.median(ft_e)
    pct_e = (cas_med_e - ft_med_e) / ft_med_e * 100
    print(f"\n  cascade vs fasttree (events median): {pct_e:+.1f}%  "
          f"(cascade {cas_med_e*1000:.2f} us, fasttree {ft_med_e*1000:.2f} us)")
    print()

    cas_launch_overhead = (cas_med_e - cas_med) * 1000
    ft_launch_overhead = (ft_med_e - ft_med) * 1000
    print(f"  Estimated launch overhead per call (events - graph):")
    print(f"    cascade:  {cas_launch_overhead:6.2f} us")
    print(f"    fasttree: {ft_launch_overhead:6.2f} us")
    print()
    print("If cascade's launch overhead is much larger than fasttree's, the")
    print("two-launch (attn + reduction) cost is a meaningful fraction of the gap.")


if __name__ == "__main__":
    main()
