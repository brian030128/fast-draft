"""Benchmark cascade vs flat across the named tree archetypes.

Per cell: build the archetype's KVTreeNode tree, then run flat /
cascade / ml_cascade FlashInfer paths and report median latency. The
table flags `cascade < flat * 0.95` so the boundary of "is sharing
beneficial?" is visible at a glance.

Reuses the prep + run helpers from tests/bench_tree_attn.py so what we
exercise here is exactly the same kernel path the existing 5-way bench
uses; the difference is the tree shape distribution.

Usage:
    python tests/bench_tree_categories.py
    python tests/bench_tree_categories.py --csv out.csv
    python tests/bench_tree_categories.py --only single_q1,single_q4
    python tests/bench_tree_categories.py --skip flat,cascade  # only ml_cascade
"""

import argparse
import csv
import os
import sys
from typing import List, Optional

import torch

# Path setup mirrors bench_tree_attn.py.
_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(__file__))  # tests/ for sibling imports
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "3rdparty", "flashinfer"))
sys.path.insert(0, os.path.join(_root, "3rdparty", "FastTree-Artifact", "kernel_bench"))

import flashinfer  # noqa: F401  (forces ext load)
try:
    from flashinfer import (
        FusedMultiLevelCascadeAttentionWrapper as CascadeBatchAttentionWrapper,
    )
except ImportError:
    from flashinfer.attention import CascadeBatchAttentionWrapper
from flashinfer.cascade import MultiLevelCascadeAttentionWrapper

from bench_tree_attn import (
    prepare_data,
    build_flashinfer_flat_indices,
    build_flashinfer_cascade_indices,
    bench_ms,
)
from tree_categories import ARCHETYPES, all_cells


def run_cell(
    label: str, builder, kwargs: dict, num_qo_heads: int, num_kv_heads: int,
    head_dim: int, dtype: torch.dtype, skip: set,
):
    tree_info = builder(**kwargs)
    data = prepare_data(tree_info, num_qo_heads, num_kv_heads, head_dim, dtype)
    q = data["q"]

    # KV cache shared across all paths (BatchDecodeWithPagedKVCacheWrapper expects
    # a (k, v) tuple where each is [num_pages, page_size=1, num_kv_heads, head_dim]).
    k_cache = data["k_tree"].unsqueeze(1)  # (T, 1, H_kv, D)
    v_cache = data["v_tree"].unsqueeze(1)
    kv_data = (k_cache, v_cache)

    result = {
        "label": label,
        "batch": data["batch_size"],
        "max_seq": data["max_seqlen"],
        "flat_us": None, "cascade_us": None, "ml_cascade_us": None,
    }

    # --- Flat ---
    if "flat" not in skip:
        kv_indptr, kv_indices, kv_last_page_len, _ = build_flashinfer_flat_indices(
            tree_info, data
        )
        flat_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD",
            use_tensor_cores=True,
        )
        flat_wrapper.plan(
            kv_indptr, kv_indices, kv_last_page_len,
            num_qo_heads, num_kv_heads, head_dim, 1,
            q_data_type=dtype,
        )
        flat_wrapper.run(q, kv_data)
        out_buf = torch.empty_like(q)
        result["flat_us"] = bench_ms(lambda: flat_wrapper.run(q, kv_data, out=out_buf)) * 1000.0

    # --- Cascade indices (shared by cascade + ml_cascade paths) ---
    casc_idx = None
    if "cascade" not in skip or "ml_cascade" not in skip:
        casc_idx = build_flashinfer_cascade_indices(tree_info, data, num_prefixes=1, topk=1)
        q_casc = casc_idx["q_reorder"]

    if "cascade" not in skip:
        cascade = CascadeBatchAttentionWrapper(num_levels=2, kv_layout="NHD", device="cuda")
        cascade.plan(
            qo_indptr_arr=casc_idx["qo_indptr_arr"],
            kv_indptr_arr=casc_idx["kv_indptr_arr"],
            kv_indices_arr=casc_idx["kv_indices_arr"],
            kv_len_arr=casc_idx["kv_len_arr"],
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=1,
            causal=False,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        cascade.run(q_casc, kv_data)
        casc_out = torch.empty_like(q_casc)
        casc_lse = torch.empty(q_casc.shape[0], q_casc.shape[1], device="cuda", dtype=torch.float32)
        result["cascade_us"] = bench_ms(
            lambda: cascade.run(q_casc, kv_data, out=casc_out, lse=casc_lse)
        ) * 1000.0

    if "ml_cascade" not in skip:
        ml_cascade = MultiLevelCascadeAttentionWrapper(
            num_levels=2,
            float_workspace_buffer=torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
            kv_layout="NHD",
        )
        ml_cascade.plan(
            qo_indptr_arr=casc_idx["qo_indptr_arr"],
            paged_kv_indptr_arr=casc_idx["kv_indptr_arr"],
            paged_kv_indices_arr=casc_idx["kv_indices_arr"],
            paged_kv_last_page_len=[
                casc_idx["shared_last_page_len"], casc_idx["unique_last_page_len"]
            ],
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=1,
            causal=False,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        ml_cascade.run(q_casc, kv_data)
        result["ml_cascade_us"] = bench_ms(
            lambda: ml_cascade.run(q_casc, kv_data)
        ) * 1000.0

    return result


def fmt_ratio(num, den):
    if num is None or den is None or den == 0:
        return "    ?"
    return f"{num / den:5.2f}x"


def fmt_us(v):
    return "        ?" if v is None else f"{v:9.1f}"


def print_table(rows: List[dict]):
    header = (
        f"  {'archetype[params]':40s}  {'batch':>5s}  {'max_seq':>7s}  "
        f"{'flat us':>9s}  {'cascade us':>10s}  {'ml_casc us':>10s}  "
        f"{'casc/flat':>9s}  {'casc/ml':>9s}  beneficial?"
    )
    sep = "=" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        beneficial = "?"
        if r["flat_us"] is not None and r["cascade_us"] is not None:
            beneficial = "YES" if r["cascade_us"] < r["flat_us"] * 0.95 else "no"
        print(
            f"  {r['label']:40s}  {r['batch']:>5d}  {r['max_seq']:>7d}  "
            f"{fmt_us(r['flat_us'])}  {fmt_us(r['cascade_us'])}  {fmt_us(r['ml_cascade_us'])}  "
            f"{fmt_ratio(r['cascade_us'], r['flat_us']):>9s}  "
            f"{fmt_ratio(r['cascade_us'], r['ml_cascade_us']):>9s}  {beneficial}"
        )
    print(sep)


def write_csv(rows: List[dict], path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["archetype", "batch", "max_seq", "flat_us", "cascade_us",
                     "ml_cascade_us", "casc_over_flat", "casc_over_ml"])
        for r in rows:
            casc_over_flat = (
                None if r["flat_us"] is None or r["cascade_us"] is None
                else r["cascade_us"] / r["flat_us"]
            )
            casc_over_ml = (
                None if r["ml_cascade_us"] is None or r["cascade_us"] is None
                else r["cascade_us"] / r["ml_cascade_us"]
            )
            w.writerow([r["label"], r["batch"], r["max_seq"],
                        r["flat_us"], r["cascade_us"], r["ml_cascade_us"],
                        casc_over_flat, casc_over_ml])
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-qo-heads", type=int, default=64)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--only", default="", help="comma-separated archetype names; default = all")
    ap.add_argument("--skip", default="", help="comma-separated of: flat, cascade, ml_cascade")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    torch.manual_seed(0)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    cells = all_cells()
    if only:
        cells = [c for c in cells if c[0].split("[")[0] in only]

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Heads: hq={args.num_qo_heads} hkv={args.num_kv_heads} dim={args.head_dim} dtype={args.dtype}")
    print(f"Running {len(cells)} cells across {len(ARCHETYPES)} archetypes")
    print()

    rows: List[dict] = []
    for label, builder, kwargs in cells:
        try:
            r = run_cell(label, builder, kwargs, args.num_qo_heads,
                          args.num_kv_heads, args.head_dim, dtype, skip)
            rows.append(r)
        except Exception as e:
            print(f"  FAIL {label}: {type(e).__name__}: {e}")
            rows.append({"label": label, "batch": 0, "max_seq": 0,
                         "flat_us": None, "cascade_us": None, "ml_cascade_us": None})

    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
