"""5-way tree attention kernel benchmark: FlashInfer (flat, cascade, ML cascade) vs FastTree vs DeFT.

Compares kernel latency on EAGLE-style draft decode workloads where multiple
branches share a common KV prefix. All methods receive identical KV data.

Usage:
    python tests/bench_tree_attn.py
    python tests/bench_tree_attn.py --num-prefixes 1,4,8 --prefix-lens 1024,8192,16384
    python tests/bench_tree_attn.py --topk 16 --suffix-len 5 --csv results.csv
"""

import argparse
import csv
import math
import os
import sys
from typing import List, Tuple

import torch
import triton

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "3rdparty", "flashinfer"))
sys.path.insert(0, os.path.join(_root, "3rdparty", "FastTree-Artifact", "kernel_bench"))

import flashinfer
from flashinfer.attention import CascadeBatchAttentionWrapper
from flashinfer.cascade import MultiLevelCascadeAttentionWrapper

from kv_tree_simple import KVTreeNode
from fasttree import FastTreeParams, fasttree_preparation, fasttree_decode
import DeFT as deft_module


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def build_tree_info(
    num_prefixes: int, prefix_len: int, topk: int, suffix_len: int,
) -> List[KVTreeNode]:
    """Build a KVTreeNode list for the EAGLE draft decode workload.

    num_prefixes=1: root(prefix_len) -> topk leaves(suffix_len)
    num_prefixes>1: dummy_root(1) -> num_prefixes mid(prefix_len) -> topk leaves each(suffix_len)
    """
    nodes: List[KVTreeNode] = []

    if num_prefixes == 1:
        # Root = the single prefix
        root = KVTreeNode()
        root.parent = -1
        root.id = 0
        root.seqlen = prefix_len
        root.num_children = topk
        nodes.append(root)
        for k in range(topk):
            leaf = KVTreeNode()
            leaf.parent = 0
            leaf.id = 1 + k
            leaf.seqlen = suffix_len
            leaf.num_children = 0
            nodes.append(leaf)
    else:
        # Dummy root
        root = KVTreeNode()
        root.parent = -1
        root.id = 0
        root.seqlen = 1
        root.num_children = num_prefixes
        nodes.append(root)
        # Mid-level prefix nodes
        for p in range(num_prefixes):
            mid = KVTreeNode()
            mid.parent = 0
            mid.id = 1 + p
            mid.seqlen = prefix_len
            mid.num_children = topk
            nodes.append(mid)
        # Leaf nodes
        nid = 1 + num_prefixes
        for p in range(num_prefixes):
            for k in range(topk):
                leaf = KVTreeNode()
                leaf.parent = 1 + p
                leaf.id = nid
                leaf.seqlen = suffix_len
                leaf.num_children = 0
                nodes.append(leaf)
                nid += 1

    # Propagate request IDs (leaf-to-root)
    req_id = 0
    for n in range(len(nodes)):
        if nodes[n].num_children == 0:
            node_idx = n
            while node_idx != -1:
                nodes[node_idx].requests.append(req_id)
                node_idx = nodes[node_idx].parent
            req_id += 1

    return nodes


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(
    tree_info: List[KVTreeNode],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
):
    """Generate all KV data formats from tree_info.

    Returns dict with:
        q:              (batch_size, num_qo_heads, head_dim)
        k_tree, v_tree: (total_tokens, num_kv_heads, head_dim) - for FastTree + FlashInfer pages
        kv_ptrs:        list[int] - cumulative token offsets per node
        k_cache_flat, v_cache_flat: (batch_size, max_seqlen, num_kv_heads, head_dim) - for DeFT
        cache_seqlens:  (batch_size,) int32
        batch_size:     int
        leaf_paths:     list[list[int]] - node indices root-to-leaf per request
    """
    node_num = len(tree_info)

    # Generate random KV per node
    k_per_node = []
    v_per_node = []
    kv_ptrs = [0]
    for n in range(node_num):
        sl = tree_info[n].seqlen
        k_per_node.append(torch.randn(sl, num_kv_heads, head_dim, device="cuda", dtype=dtype))
        v_per_node.append(torch.randn(sl, num_kv_heads, head_dim, device="cuda", dtype=dtype))
        kv_ptrs.append(kv_ptrs[-1] + sl)

    # Concatenated tree tensors (for FastTree)
    k_tree = torch.cat(k_per_node, dim=0)  # (total_tokens, num_kv_heads, head_dim)
    v_tree = torch.cat(v_per_node, dim=0)

    # Find leaves and build root-to-leaf paths
    num_requests = 0
    max_seqlen = 0
    leaf_paths = []  # list of root-to-leaf node index chains
    for n in range(node_num):
        if tree_info[n].num_children == 0:
            chain = []
            node_idx = n
            while node_idx != -1:
                chain.append(node_idx)
                node_idx = tree_info[node_idx].parent
            chain.reverse()  # root-to-leaf
            leaf_paths.append(chain)
            path_len = sum(tree_info[c].seqlen for c in chain)
            max_seqlen = max(max_seqlen, path_len)
            num_requests += 1

    batch_size = num_requests

    # Flat K/V cache for DeFT: (batch_size, max_seqlen, num_kv_heads, head_dim)
    k_cache_flat = torch.zeros(batch_size, max_seqlen, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    v_cache_flat = torch.zeros(batch_size, max_seqlen, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device="cuda")

    for req_idx, chain in enumerate(leaf_paths):
        offset = 0
        for node_idx in chain:
            sl = tree_info[node_idx].seqlen
            k_cache_flat[req_idx, offset:offset + sl] = k_per_node[node_idx]
            v_cache_flat[req_idx, offset:offset + sl] = v_per_node[node_idx]
            offset += sl
        cache_seqlens[req_idx] = offset

    q = torch.randn(batch_size, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    return {
        "q": q,
        "k_tree": k_tree,
        "v_tree": v_tree,
        "kv_ptrs": kv_ptrs,
        "k_cache_flat": k_cache_flat,
        "v_cache_flat": v_cache_flat,
        "cache_seqlens": cache_seqlens,
        "batch_size": batch_size,
        "leaf_paths": leaf_paths,
        "max_seqlen": max_seqlen,
    }


# ---------------------------------------------------------------------------
# FlashInfer page index helpers
# ---------------------------------------------------------------------------

def build_flashinfer_flat_indices(tree_info, data):
    """Build paged KV indices for flat FlashInfer decode (page_size=1).

    Each request reads the full root-to-leaf path as pages into the tree tensor.
    Page index for token t of node n = kv_ptrs[n] + t.
    """
    kv_ptrs = data["kv_ptrs"]
    batch_size = data["batch_size"]
    indices_list = []
    kv_lens = []
    for chain in data["leaf_paths"]:
        req_indices = []
        for node_idx in chain:
            sl = tree_info[node_idx].seqlen
            start = kv_ptrs[node_idx]
            req_indices.append(torch.arange(start, start + sl, device="cuda", dtype=torch.int32))
        indices_list.append(torch.cat(req_indices))
        kv_lens.append(indices_list[-1].shape[0])

    kv_indices = torch.cat(indices_list)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    torch.cumsum(kv_lens_t, dim=0, out=kv_indptr[1:])
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device="cuda")
    return kv_indptr, kv_indices, kv_last_page_len, kv_lens_t


def build_flashinfer_cascade_indices(tree_info, data, num_prefixes, topk):
    """Build 2-level cascade indices for FlashInfer.

    Shared level: groups of topk queries share the prefix path (root to prefix node).
    Unique level: each query has its own suffix (leaf node tokens).
    """
    kv_ptrs = data["kv_ptrs"]
    batch_size = data["batch_size"]
    leaf_paths = data["leaf_paths"]

    # For each request, the "shared" part is everything except the last node (leaf),
    # and the "unique" part is the last node (leaf).
    # Requests sharing the same prefix node are grouped together.

    # Build prefix groups: map prefix_parent_id -> list of request indices
    from collections import OrderedDict
    prefix_groups = OrderedDict()
    for req_idx, chain in enumerate(leaf_paths):
        # The prefix path is chain[:-1], unique is chain[-1]
        prefix_key = tuple(chain[:-1])
        if prefix_key not in prefix_groups:
            prefix_groups[prefix_key] = []
        prefix_groups[prefix_key].append(req_idx)

    num_groups = len(prefix_groups)

    # Shared level
    shared_kv_indices_list = []
    shared_kv_lens = []
    qo_indptr_shared = [0]
    for prefix_chain, req_indices in prefix_groups.items():
        # KV indices for this shared prefix
        indices = []
        for node_idx in prefix_chain:
            sl = tree_info[node_idx].seqlen
            start = kv_ptrs[node_idx]
            indices.append(torch.arange(start, start + sl, device="cuda", dtype=torch.int32))
        shared_kv_indices_list.append(torch.cat(indices) if indices else torch.zeros(0, dtype=torch.int32, device="cuda"))
        shared_kv_lens.append(sum(tree_info[n].seqlen for n in prefix_chain))
        qo_indptr_shared.append(qo_indptr_shared[-1] + len(req_indices))

    shared_kv_indices = torch.cat(shared_kv_indices_list) if shared_kv_indices_list else torch.zeros(0, dtype=torch.int32, device="cuda")
    shared_kv_lens_t = torch.tensor(shared_kv_lens, dtype=torch.int32, device="cuda")
    shared_kv_indptr = torch.zeros(num_groups + 1, dtype=torch.int32, device="cuda")
    torch.cumsum(shared_kv_lens_t, dim=0, out=shared_kv_indptr[1:])
    qo_indptr_shared = torch.tensor(qo_indptr_shared, dtype=torch.int32, device="cuda")

    # Unique level: reorder requests to match group ordering
    ordered_req_indices = []
    for req_indices in prefix_groups.values():
        ordered_req_indices.extend(req_indices)

    unique_kv_indices_list = []
    unique_kv_lens = []
    for req_idx in ordered_req_indices:
        chain = leaf_paths[req_idx]
        leaf_node = chain[-1]
        sl = tree_info[leaf_node].seqlen
        start = kv_ptrs[leaf_node]
        unique_kv_indices_list.append(torch.arange(start, start + sl, device="cuda", dtype=torch.int32))
        unique_kv_lens.append(sl)

    unique_kv_indices = torch.cat(unique_kv_indices_list)
    unique_kv_lens_t = torch.tensor(unique_kv_lens, dtype=torch.int32, device="cuda")
    unique_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    torch.cumsum(unique_kv_lens_t, dim=0, out=unique_kv_indptr[1:])
    qo_indptr_unique = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")

    # Reorder Q to match group ordering
    q_reorder = data["q"][ordered_req_indices]

    return {
        "qo_indptr_arr": [qo_indptr_shared, qo_indptr_unique],
        "kv_indptr_arr": [shared_kv_indptr, unique_kv_indptr],
        "kv_indices_arr": [shared_kv_indices, unique_kv_indices],
        "kv_len_arr": [shared_kv_lens_t, unique_kv_lens_t],
        "q_reorder": q_reorder,
        "ordered_req_indices": ordered_req_indices,
        # For ML cascade
        "shared_last_page_len": torch.ones(num_groups, dtype=torch.int32, device="cuda"),
        "unique_last_page_len": torch.ones(batch_size, dtype=torch.int32, device="cuda"),
    }


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def bench_ms(func) -> float:
    """Benchmark a function using triton.testing.do_bench, return median ms."""
    result = triton.testing.do_bench(func, quantiles=[0.5, 0.2, 0.8])
    if isinstance(result, (list, tuple)):
        return float(result[0])
    return float(result)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    num_prefixes_list: List[int],
    prefix_lens: List[int],
    topk_list: List[int] = None,
    suffix_len: int = 5,
    num_qo_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
    skip_methods: List[str] = None,
):
    if topk_list is None:
        topk_list = [4, 8, 16]
    skip = set(skip_methods or [])
    sm_scale = 1.0 / math.sqrt(head_dim)
    results = []

    for num_prefixes in num_prefixes_list:
        for prefix_len in prefix_lens:
            for topk in topk_list:
                print(f"\n--- num_prefixes={num_prefixes}, prefix_len={prefix_len}, topk={topk}, suffix_len={suffix_len} ---")

                tree_info = build_tree_info(num_prefixes, prefix_len, topk, suffix_len)
                data = prepare_data(tree_info, num_qo_heads, num_kv_heads, head_dim, dtype)
                batch_size = data["batch_size"]
                q = data["q"]
                kv_data = (data["k_tree"], data["v_tree"])

                row = {
                    "num_prefixes": num_prefixes,
                    "prefix_len": prefix_len,
                    "topk": topk,
                    "suffix_len": suffix_len,
                    "batch_size": batch_size,
                }

                # ---- Flat FlashInfer ----
                flat_out = None
                if "flat" not in skip:
                    kv_indptr, kv_indices, kv_last_page_len, kv_lens = build_flashinfer_flat_indices(tree_info, data)
                    flat_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                        torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD",
                    )
                    flat_wrapper.plan(
                        kv_indptr, kv_indices, kv_last_page_len,
                        num_qo_heads, num_kv_heads, head_dim, 1,
                        q_data_type=dtype,
                    )
                    flat_out = flat_wrapper.run(q, kv_data)
                    row["flat_ms"] = bench_ms(lambda: flat_wrapper.run(q, kv_data))
                    print(f"  flat:       {row['flat_ms']:.4f} ms")
                else:
                    row["flat_ms"] = ""

                # ---- Cascade FlashInfer ----
                if "cascade" not in skip:
                    casc_idx = build_flashinfer_cascade_indices(tree_info, data, num_prefixes, topk)
                    q_casc = casc_idx["q_reorder"]
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
                    casc_out_reordered, _ = cascade.run(q_casc, kv_data)
                    # Un-reorder to match flat output
                    inv_order = [0] * batch_size
                    for i, ri in enumerate(casc_idx["ordered_req_indices"]):
                        inv_order[ri] = i
                    casc_out = casc_out_reordered[inv_order]

                    def run_cascade():
                        cascade.run(q_casc, kv_data)
                    row["cascade_ms"] = bench_ms(run_cascade)
                    if flat_out is not None:
                        row["cascade_diff"] = (flat_out - casc_out).abs().max().item()
                    else:
                        row["cascade_diff"] = ""
                    print(f"  cascade:    {row['cascade_ms']:.4f} ms  diff={row['cascade_diff']}")
                else:
                    row["cascade_ms"] = ""
                    row["cascade_diff"] = ""

                # ---- MultiLevel Cascade FlashInfer ----
                if "ml_cascade" not in skip:
                    if "cascade" in skip:
                        casc_idx = build_flashinfer_cascade_indices(tree_info, data, num_prefixes, topk)
                    q_casc = casc_idx["q_reorder"]
                    ml_cascade = MultiLevelCascadeAttentionWrapper(
                        num_levels=2,
                        float_workspace_buffer=torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
                        kv_layout="NHD",
                    )
                    ml_cascade.plan(
                        qo_indptr_arr=casc_idx["qo_indptr_arr"],
                        paged_kv_indptr_arr=casc_idx["kv_indptr_arr"],
                        paged_kv_indices_arr=casc_idx["kv_indices_arr"],
                        paged_kv_last_page_len=[casc_idx["shared_last_page_len"], casc_idx["unique_last_page_len"]],
                        num_qo_heads=num_qo_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        page_size=1,
                        causal=False,
                        q_data_type=dtype,
                        kv_data_type=dtype,
                    )
                    ml_out_reordered = ml_cascade.run(q_casc, kv_data)
                    inv_order = [0] * batch_size
                    for i, ri in enumerate(casc_idx["ordered_req_indices"]):
                        inv_order[ri] = i
                    ml_out = ml_out_reordered[inv_order]

                    def run_ml_cascade():
                        ml_cascade.run(q_casc, kv_data)
                    row["ml_cascade_ms"] = bench_ms(run_ml_cascade)
                    if flat_out is not None:
                        row["ml_cascade_diff"] = (flat_out - ml_out).abs().max().item()
                    else:
                        row["ml_cascade_diff"] = ""
                    print(f"  ml_cascade: {row['ml_cascade_ms']:.4f} ms  diff={row['ml_cascade_diff']}")
                else:
                    row["ml_cascade_ms"] = ""
                    row["ml_cascade_diff"] = ""

                # ---- FastTree ----
                if "fasttree" not in skip:
                    try:
                        # Need a fresh tree_info copy since fasttree_preparation may mutate requests
                        tree_info_ft = build_tree_info(num_prefixes, prefix_len, topk, suffix_len)
                        data_ft = prepare_data(tree_info_ft, num_qo_heads, num_kv_heads, head_dim, dtype)
                        q_ft = data_ft["q"].copy_(q)  # same Q values

                        params = FastTreeParams()
                        params.set_values(0.66, 0.33, 0.1)
                        params.set_q_tile_sizes([16, 4])
                        params.set_kv_tile_sizes([32, 32])
                        params.set_kv_group_num(num_qo_heads // num_kv_heads)

                        ft_aux, _ = fasttree_preparation(
                            tree_info_ft,
                            data_ft["kv_ptrs"],
                            data_ft["batch_size"],
                            num_qo_heads,
                            num_kv_heads,
                            head_dim,
                            [1024, 128],
                            [132, 528],
                            [132, 132],
                            params,
                        )
                        ft_out = torch.empty(batch_size, num_qo_heads, head_dim, device="cuda", dtype=dtype)

                        def run_fasttree():
                            fasttree_decode(
                                q_ft, data_ft["k_tree"], data_ft["v_tree"], ft_out,
                                *ft_aux, [16, 4], [32, 32], sm_scale,
                            )
                        run_fasttree()  # warmup / get output
                        row["fasttree_ms"] = bench_ms(run_fasttree)
                        row["fasttree_diff"] = ""  # different data layout, skip exact comparison
                        print(f"  fasttree:   {row['fasttree_ms']:.4f} ms")
                    except Exception as e:
                        row["fasttree_ms"] = ""
                        row["fasttree_diff"] = f"error:{type(e).__name__}"
                        print(f"  fasttree:   ERROR {e}")
                else:
                    row["fasttree_ms"] = ""
                    row["fasttree_diff"] = ""

                # ---- DeFT ----
                if "deft" not in skip:
                    try:
                        tree_info_deft = build_tree_info(num_prefixes, prefix_len, topk, suffix_len)
                        data_deft = prepare_data(tree_info_deft, num_qo_heads, num_kv_heads, head_dim, dtype)
                        q_deft = data_deft["q"].copy_(q)

                        deft_module.cur_length = 0
                        deft_aux = deft_module.DeFT_preparation(
                            tree_info_deft,
                            data_deft["k_cache_flat"],
                            128,  # subtree_len
                            64,   # mask_len
                            num_qo_heads,
                            head_dim,
                        )
                        deft_out = torch.empty(batch_size, num_qo_heads, head_dim, device="cuda", dtype=dtype)
                        k_buf = data_deft["k_cache_flat"].view(-1, num_kv_heads, head_dim)
                        v_buf = data_deft["v_cache_flat"].view(-1, num_kv_heads, head_dim)

                        def run_deft():
                            deft_module.DeFT_decode(
                                q_deft, k_buf, v_buf, deft_out,
                                *deft_aux, 16, 32, sm_scale, 64,
                            )
                        run_deft()  # warmup / get output
                        row["deft_ms"] = bench_ms(run_deft)
                        row["deft_diff"] = ""
                        print(f"  deft:       {row['deft_ms']:.4f} ms")
                    except Exception as e:
                        row["deft_ms"] = ""
                        row["deft_diff"] = f"error:{type(e).__name__}"
                        print(f"  deft:       ERROR {e}")
                else:
                    row["deft_ms"] = ""
                    row["deft_diff"] = ""

                results.append(row)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(results):
    print(f"\n{'='*130}")
    print(f"Tree Attention Kernel Benchmark: FlashInfer (flat/cascade/ml_cascade) vs FastTree vs DeFT")
    print(f"{'='*130}")
    header = (
        f"  {'#prefixes':>9}  {'prefix_len':>10}  {'topk':>4}  {'suffix_len':>10}  "
        f"{'flat(ms)':>9}  {'cascade':>9}  {'ml_casc':>9}  {'fasttree':>9}  {'deft':>9}  "
        f"{'casc_diff':>9}  {'ml_diff':>9}  {'fastest':>10}  {'speedup':>7}"
    )
    print(header)
    print(f"  {'-'*9}  {'-'*10}  {'-'*4}  {'-'*10}  "
          f"{'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  "
          f"{'-'*9}  {'-'*9}  {'-'*10}  {'-'*7}")
    for r in results:
        def fmt(v):
            if isinstance(v, (int, float)) and v != "":
                return f"{v:.4f}"
            return str(v) if v != "" else "---"

        # Find fastest kernel and speedup over flat
        method_times = {}
        for name, key in [("flat", "flat_ms"), ("cascade", "cascade_ms"),
                          ("ml_casc", "ml_cascade_ms"), ("fasttree", "fasttree_ms"),
                          ("deft", "deft_ms")]:
            v = r.get(key, "")
            if isinstance(v, (int, float)) and v != "":
                method_times[name] = v
        if method_times:
            fastest_name = min(method_times, key=method_times.get)
            flat_ms = r.get("flat_ms", "")
            if isinstance(flat_ms, (int, float)) and flat_ms != "" and flat_ms > 0:
                speedup = f"{flat_ms / method_times[fastest_name]:.2f}x"
            else:
                speedup = "---"
        else:
            fastest_name = "---"
            speedup = "---"

        print(
            f"  {r['num_prefixes']:>9}  {r['prefix_len']:>10}  {r['topk']:>4}  {r['suffix_len']:>10}  "
            f"{fmt(r.get('flat_ms', '')):>9}  {fmt(r.get('cascade_ms', '')):>9}  "
            f"{fmt(r.get('ml_cascade_ms', '')):>9}  {fmt(r.get('fasttree_ms', '')):>9}  "
            f"{fmt(r.get('deft_ms', '')):>9}  "
            f"{fmt(r.get('cascade_diff', '')):>9}  {fmt(r.get('ml_cascade_diff', '')):>9}  "
            f"{fastest_name:>10}  {speedup:>7}"
        )
    print(f"{'='*130}")


def write_csv(results, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fieldnames = [
        "num_prefixes", "prefix_len", "topk", "suffix_len", "batch_size",
        "flat_ms", "cascade_ms", "ml_cascade_ms", "fasttree_ms", "deft_ms",
        "cascade_diff", "ml_cascade_diff", "fasttree_diff", "deft_diff",
        "fastest", "speedup_over_flat",
    ]
    # Compute fastest / speedup for CSV rows
    for r in results:
        method_times = {}
        for name, key in [("flat", "flat_ms"), ("cascade", "cascade_ms"),
                          ("ml_casc", "ml_cascade_ms"), ("fasttree", "fasttree_ms"),
                          ("deft", "deft_ms")]:
            v = r.get(key, "")
            if isinstance(v, (int, float)) and v != "":
                method_times[name] = v
        if method_times:
            fastest_name = min(method_times, key=method_times.get)
            flat_ms = r.get("flat_ms", "")
            if isinstance(flat_ms, (int, float)) and flat_ms != "" and flat_ms > 0:
                r["speedup_over_flat"] = round(flat_ms / method_times[fastest_name], 2)
            else:
                r["speedup_over_flat"] = ""
            r["fastest"] = fastest_name
        else:
            r["fastest"] = ""
            r["speedup_over_flat"] = ""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {len(results)} rows to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="5-way tree attention kernel benchmark")
    parser.add_argument("--num-prefixes", default="1,4,8", help="Comma-separated num_prefixes values")
    parser.add_argument("--prefix-lens", default="1024,8192,16384", help="Comma-separated prefix lengths")
    parser.add_argument("--topk", default="4,8,16", help="Comma-separated topk values")
    parser.add_argument("--suffix-len", type=int, default=5)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--csv", default=None, help="Path to write CSV results")
    parser.add_argument("--skip", default="", help="Comma-separated methods to skip: flat,cascade,ml_cascade,fasttree,deft")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    results = run_benchmark(
        num_prefixes_list=[int(x) for x in args.num_prefixes.split(",")],
        prefix_lens=[int(x) for x in args.prefix_lens.split(",")],
        topk_list=[int(x) for x in args.topk.split(",")],
        suffix_len=args.suffix_len,
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        skip_methods=[s.strip() for s in args.skip.split(",") if s.strip()],
    )

    print_table(results)
    if args.csv:
        write_csv(results, args.csv)


if __name__ == "__main__":
    main()
