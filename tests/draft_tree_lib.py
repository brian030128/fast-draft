"""
Shared utilities for draft tree benchmarks.

Contains:
  - Tree topology builders (best, worst, lollipop, eagle)
  - Metadata generation for MultiLevelCascadeAttentionWrapper
  - KV cache allocation
  - CUDA Graph benchmark functions
  - Anomaly detection and batch replication helpers
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

import flashinfer


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Tree Builders
# ---------------------------------------------------------------------------

def build_min_sharing_tree(depth: int, width: int) -> List[Tuple[int, int]]:
    """Minimum-sharing tree: balanced fan-out, all `width` leaves at depth `depth`.

    Per-depth node count grows log-spaced from 1 (root) to `width`:

        n_d = round(width ** (d / depth))     for d in 1..depth

    Effective branching factor b = width^(1/depth) (depth = log_b(width)).
    The n_d children are distributed as evenly as possible among the n_{d-1}
    parents (each parent receives ≥1 child since n_d is forced
    non-decreasing), so every path extends to the leaf depth.

    Among the three topologies this has the *least* prefix sharing across
    leaves — paths diverge as early as possible.
    """
    counts: List[int] = []
    for d in range(1, depth + 1):
        c = max(1, round(width ** (d / depth)))
        if counts and c < counts[-1]:
            c = counts[-1]
        counts.append(c)

    edges: List[Tuple[int, int]] = []
    next_id = 1
    parents = [0]
    for n in counts:
        base, rem = divmod(n, len(parents))
        children: List[int] = []
        for i, p in enumerate(parents):
            k = base + (1 if i < rem else 0)
            for _ in range(k):
                edges.append((p, next_id))
                children.append(next_id)
                next_id += 1
        parents = children
    return edges


def build_max_sharing_tree(depth: int, width: int) -> List[Tuple[int, int]]:
    """Maximum-sharing tree: long lollipop (single stem + leaf burst at tip).

    Depth 1..(D-1): single stem (no branching).
    Depth D: burst remaining budget at the tip.
    Total nodes = W * D.
    Every leaf shares the entire (D-1)-long stem prefix — among the three
    topologies this has the *most* prefix sharing across leaves.

    Multi-Level merges the stem into one level → only 2 extra levels total.
    Forced Two-Level: each leaf reads O(D) redundant stem KV → Memory Wall.
    """
    budget = depth * width
    return build_lollipop_tree(depth - 1, budget)


def build_lollipop_tree(stem_length: int, budget: int) -> List[Tuple[int, int]]:
    """Parameterized Lollipop tree: stem of given length, then burst all remaining budget.

    Args:
        stem_length: number of sequential (non-branching) nodes before the burst
        budget: total number of nodes in the tree

    Two-Level redundancy = burst_nodes × stem_length = (budget - stem_length) × stem_length.
    This is maximized at stem_length ≈ budget / 2.
    """
    assert stem_length < budget, f"stem_length ({stem_length}) must be < budget ({budget})"
    edges: List[Tuple[int, int]] = []
    next_id = 1
    parent = 0
    # Build stem
    for _ in range(stem_length):
        edges.append((parent, next_id))
        parent = next_id
        next_id += 1
    # Burst remaining budget at tip
    burst_count = budget - stem_length
    for _ in range(burst_count):
        edges.append((parent, next_id))
        next_id += 1
    return edges


def build_sampled_tree(
    depth: int,
    width: int,
    seed: int = None,
    alpha: float = 0.3,
) -> List[Tuple[int, int]]:
    """Beam-search-style tree sampling with realistic peaked next-token probs.

    Simulates speculative decoding via beam search with beam size = `width`.
    Each beam's children get probabilities drawn from a Dirichlet(alpha)
    distribution — small `alpha` makes the distribution peaky like real LLM
    next-token output (alpha≈0.3 → top-3 tokens hold ~0.8 cumulative mass on
    a 16-way simplex). Top-`width` candidates by accumulated log-prob from
    root survive each step, so good prefixes attract many siblings (the
    source of prefix sharing).

    Each layer has exactly `width` nodes, giving total budget = width * depth.
    """
    rng = np.random.default_rng(seed)
    alphas = np.full(width, alpha)

    edges: List[Tuple[int, int]] = []
    next_id = 1

    # Step 0: expand root into `width` children.
    log_probs = np.log(rng.dirichlet(alphas))
    # beams: list of (node_id, accumulated_log_prob_from_root)
    beams: List[Tuple[int, float]] = []
    for i in range(width):
        edges.append((0, next_id))
        beams.append((next_id, float(log_probs[i])))
        next_id += 1

    # Steps 1..depth-1: each beam samples `width` children, keep top-`width`.
    for _ in range(1, depth):
        all_log_probs = np.log(np.stack(
            [rng.dirichlet(alphas) for _ in range(len(beams))]
        ))

        candidates: List[Tuple[int, float]] = []
        for (parent_id, parent_acc), child_lps in zip(beams, all_log_probs):
            for clp in child_lps:
                candidates.append((parent_id, parent_acc + float(clp)))

        # Keep top-`width` by accumulated log-prob.
        candidates.sort(key=lambda c: c[1], reverse=True)
        top = candidates[:width]

        new_beams = []
        for parent_id, acc in top:
            edges.append((parent_id, next_id))
            new_beams.append((next_id, acc))
            next_id += 1
        beams = new_beams

    return edges


# ---------------------------------------------------------------------------
# Metadata Helpers
# ---------------------------------------------------------------------------

def _build_depth_map(edges: List[Tuple[int, int]]) -> Dict[int, int]:
    """Returns node -> depth mapping. Root (0) is depth 0."""
    depth_map = {0: 0}
    for parent, child in edges:
        depth_map[child] = depth_map[parent] + 1
    return depth_map


def _build_children_map(edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    """Returns node -> list of children."""
    children = defaultdict(list)
    for parent, child in edges:
        children[parent].append(child)
    return children


def _get_path_to_root(node: int, parent_map: Dict[int, int]) -> List[int]:
    """Return the path from root to node (inclusive), excluding root (0)."""
    path = []
    cur = node
    while cur != 0:
        path.append(cur)
        cur = parent_map[cur]
    path.reverse()
    return path


def _get_all_nodes(edges: List[Tuple[int, int]]) -> List[int]:
    """Return all non-root nodes in BFS order."""
    nodes = set()
    for parent, child in edges:
        nodes.add(child)
    return sorted(nodes)


def _get_leaf_nodes(edges: List[Tuple[int, int]]) -> List[int]:
    """Return leaf nodes (nodes with no children) in sorted order.

    In SD draft phase, only leaf nodes are queries — stem/internal
    nodes are already in the KV cache.
    """
    children_map = _build_children_map(edges)
    all_nodes = _get_all_nodes(edges)
    return [n for n in all_nodes if not children_map.get(n)]


# ---------------------------------------------------------------------------
# Metadata Generation
# ---------------------------------------------------------------------------

def build_multi_level_metadata(
    edges: List[Tuple[int, int]],
    prompt_pages: int,
    node_page_map: Dict[int, int],
    page_size: int,
    prompt_len: int,
) -> Optional[dict]:
    """Build Multi-Level metadata for SD draft phase (leaf-only queries).

    Only leaf nodes (nodes with no children) are queries — stem/internal
    nodes are already in the KV cache.

    Level 0 = shared prompt.
    Then we walk the tree depth by depth. Consecutive depths without branching
    are merged into one level.  Each level's KV is the set of internal nodes
    at that segment's depths.
    """
    depth_map = _build_depth_map(edges)
    children_map = _build_children_map(edges)

    if not edges:
        return None

    all_nodes = _get_all_nodes(edges)
    leaf_nodes = _get_leaf_nodes(edges)
    if not leaf_nodes:
        return None

    max_depth = max(depth_map[n] for n in all_nodes)

    # Group nodes by depth
    nodes_at_depth = defaultdict(list)
    for n in all_nodes:
        nodes_at_depth[depth_map[n]].append(n)

    # Determine which depths trigger a new level
    level_boundaries = [1]
    for d in range(1, max_depth + 1):
        any_branching = False
        for n in nodes_at_depth[d]:
            if len(children_map.get(n, [])) > 1:
                any_branching = True
                break
        if any_branching and d + 1 <= max_depth:
            level_boundaries.append(d + 1)

    # Also check root branching
    if len(children_map.get(0, [])) > 1 and 1 not in level_boundaries:
        pass  # depth 1 is already in level_boundaries

    # Build segments
    segments = []
    for i, start in enumerate(level_boundaries):
        end = level_boundaries[i + 1] if i + 1 < len(level_boundaries) else max_depth + 1
        segments.append((start, end))

    # --- Leaf-only queries ---
    query_node_list = leaf_nodes
    total_q = len(query_node_list)
    node_to_query_idx = {n: i for i, n in enumerate(query_node_list)}

    # Level 0 (prompt)
    qo_indptr_0 = torch.tensor([0, total_q], device="cuda", dtype=torch.int32)
    kv_indptr_0 = torch.tensor([0, prompt_pages], device="cuda", dtype=torch.int32)
    kv_indices_0 = torch.arange(prompt_pages, device="cuda", dtype=torch.int32)
    last_page_len_0 = torch.tensor(
        [(prompt_len - 1) % page_size + 1], device="cuda", dtype=torch.int32
    )

    qo_indptr_arr = [qo_indptr_0]
    kv_indptr_arr = [kv_indptr_0]
    kv_indices_arr = [kv_indices_0]
    last_page_len_arr = [last_page_len_0]

    # Parent map for path tracing
    parent_map = {}
    for parent, child in edges:
        parent_map[child] = parent

    for seg_start, seg_end in segments:
        seg_nodes = set()
        for d in range(seg_start, seg_end):
            seg_nodes.update(nodes_at_depth[d])

        def get_ancestor_at_depth(node, target_depth):
            cur = node
            while depth_map[cur] > target_depth:
                cur = parent_map[cur]
            return cur if depth_map[cur] == target_depth else None

        # Group leaf queries by their path through this segment
        groups = defaultdict(list)
        for q_node in query_node_list:
            if depth_map[q_node] < seg_start:
                continue
            ancestor = get_ancestor_at_depth(q_node, seg_start)
            if ancestor is None or ancestor not in seg_nodes:
                continue
            path_nodes = []
            cur = q_node
            while cur != 0 and depth_map[cur] >= seg_start:
                if cur in seg_nodes:
                    path_nodes.append(cur)
                cur = parent_map[cur]
            path_nodes.reverse()
            path_key = tuple(path_nodes)
            groups[path_key].append(node_to_query_idx[q_node])

        if not groups:
            continue

        sorted_groups = sorted(groups.items(), key=lambda kv: kv[1][0])

        qo_indptr_list = [0]
        kv_indptr_list = [0]
        kv_indices_list = []
        last_page_lens = []

        for path_nodes, query_indices in sorted_groups:
            pages = [node_page_map[n] for n in path_nodes]
            kv_indices_list.extend(pages)
            kv_indptr_list.append(kv_indptr_list[-1] + len(pages))
            qo_indptr_list.append(qo_indptr_list[-1] + len(query_indices))
            last_page_lens.append(1)

        qo_indptr_arr.append(
            torch.tensor(qo_indptr_list, device="cuda", dtype=torch.int32)
        )
        kv_indptr_arr.append(
            torch.tensor(kv_indptr_list, device="cuda", dtype=torch.int32)
        )
        kv_indices_arr.append(
            torch.tensor(kv_indices_list, device="cuda", dtype=torch.int32)
        )
        last_page_len_arr.append(
            torch.tensor(last_page_lens, device="cuda", dtype=torch.int32)
        )

    return {
        "num_levels": len(qo_indptr_arr),
        "qo_indptr_arr": qo_indptr_arr,
        "kv_indptr_arr": kv_indptr_arr,
        "kv_indices_arr": kv_indices_arr,
        "last_page_len_arr": last_page_len_arr,
        "total_q": total_q,
        "query_node_list": query_node_list,
    }


def build_forced_two_level_metadata(
    edges: List[Tuple[int, int]],
    prompt_pages: int,
    node_page_map: Dict[int, int],
    page_size: int,
    prompt_len: int,
) -> Optional[dict]:
    """Build Forced Two-Level metadata for SD draft phase (leaf-only queries).

    Level 0: shared prompt (all leaf queries attend to it).
    Level 1: each leaf's full ancestor path (NOT including prompt).

    Only leaf nodes are queries — stem/internal nodes are already in KV cache.
    """
    parent_map = {}
    for parent, child in edges:
        parent_map[child] = parent

    leaf_nodes = _get_leaf_nodes(edges)
    total_q = len(leaf_nodes)

    if total_q == 0:
        return None

    # Level 0: prompt
    qo_indptr_0 = torch.tensor([0, total_q], device="cuda", dtype=torch.int32)
    kv_indptr_0 = torch.tensor([0, prompt_pages], device="cuda", dtype=torch.int32)
    kv_indices_0 = torch.arange(prompt_pages, device="cuda", dtype=torch.int32)
    last_page_len_0 = torch.tensor(
        [(prompt_len - 1) % page_size + 1], device="cuda", dtype=torch.int32
    )

    # Level 1: each leaf's full ancestor path (including itself)
    qo_indptr_list = [0]
    kv_indptr_list = [0]
    kv_indices_list = []
    last_page_lens = []

    for node in leaf_nodes:
        path = _get_path_to_root(node, parent_map)
        pages = [node_page_map[n] for n in path]
        kv_indices_list.extend(pages)
        kv_indptr_list.append(kv_indptr_list[-1] + len(pages))
        qo_indptr_list.append(qo_indptr_list[-1] + 1)
        last_page_lens.append(1)

    qo_indptr_1 = torch.tensor(qo_indptr_list, device="cuda", dtype=torch.int32)
    kv_indptr_1 = torch.tensor(kv_indptr_list, device="cuda", dtype=torch.int32)
    kv_indices_1 = torch.tensor(kv_indices_list, device="cuda", dtype=torch.int32)
    last_page_len_1 = torch.tensor(last_page_lens, device="cuda", dtype=torch.int32)

    return {
        "num_levels": 2,
        "qo_indptr_arr": [qo_indptr_0, qo_indptr_1],
        "kv_indptr_arr": [kv_indptr_0, kv_indptr_1],
        "kv_indices_arr": [kv_indices_0, kv_indices_1],
        "last_page_len_arr": [last_page_len_0, last_page_len_1],
        "total_q": total_q,
        "query_node_list": leaf_nodes,
    }


# ---------------------------------------------------------------------------
# KV Cache & Pages
# ---------------------------------------------------------------------------

def allocate_kv_cache_and_pages(
    edges: List[Tuple[int, int]],
    prompt_len: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, int, Dict[int, int]]:
    """Allocate paged KV cache for prompt + draft tree."""
    all_nodes = _get_all_nodes(edges)
    prompt_pages = ceil_div(prompt_len, page_size)
    total_pages = prompt_pages + len(all_nodes)

    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        device="cuda", dtype=dtype,
    )

    node_page_map = {}
    for i, node in enumerate(all_nodes):
        node_page_map[node] = prompt_pages + i

    return kv_data, prompt_pages, node_page_map


# ---------------------------------------------------------------------------
# CUDA Graph Benchmark
# ---------------------------------------------------------------------------

def _trimmed_mean(values: List[float], trim_frac: float = 0.2) -> float:
    """Trim top/bottom trim_frac of sorted values and return the mean."""
    if len(values) <= 2:
        return sum(values) / len(values)
    s = sorted(values)
    trim_n = max(1, int(len(s) * trim_frac))
    trimmed = s[trim_n:-trim_n]
    return sum(trimmed) / len(trimmed) if trimmed else sum(s) / len(s)


def benchmark_one_config(
    meta: dict,
    q: torch.Tensor,
    kv_data: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
    warmup: int = 50,
    repeat: int = 200,
    rounds: int = 1,
) -> float:
    """Benchmark a single config with MultiLevelCascadeAttentionWrapper + CUDA Graph."""
    num_levels = meta["num_levels"]

    wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        num_levels,
        torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda"),
        "NHD",
    )
    wrapper.plan(
        meta["qo_indptr_arr"],
        meta["kv_indptr_arr"],
        meta["kv_indices_arr"],
        meta["last_page_len_arr"],
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True,
        q_data_type=dtype,
    )

    for _ in range(warmup):
        wrapper.run(q, kv_data)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, kv_data)

    medians = []
    for _ in range(rounds):
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        for i in range(repeat):
            start_events[i].record()
            graph.replay()
            end_events[i].record()
        torch.cuda.synchronize()
        times = sorted(s.elapsed_time(e) for s, e in zip(start_events, end_events))
        medians.append(times[len(times) // 2])

    if rounds == 1:
        return medians[0]
    return _trimmed_mean(medians)


def benchmark_fused_config(
    meta: dict,
    q: torch.Tensor,
    kv_data: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
    warmup: int = 50,
    repeat: int = 200,
    rounds: int = 1,
) -> float:
    """Benchmark using CascadeBatchAttentionWrapper (fused multi-level).

    All cascade levels processed in 1 attention kernel + 1 reduction kernel.
    Following ref.py pattern: no CUDA graph, direct plan+run.
    """
    from flashinfer.attention import CascadeBatchAttentionWrapper

    num_levels = meta["num_levels"]

    # Convert last_page_len_arr → kv_len_arr
    # kv_len = total tokens per group = (num_pages - 1) * page_size + last_page_len
    kv_len_arr = []
    for lvl in range(num_levels):
        kv_indptr = meta["kv_indptr_arr"][lvl]
        last_page_len = meta["last_page_len_arr"][lvl]
        pages_per_group = kv_indptr[1:] - kv_indptr[:-1]
        kv_len = (pages_per_group - 1) * page_size + last_page_len
        kv_len = torch.where(pages_per_group > 0, kv_len, torch.zeros_like(kv_len))
        kv_len_arr.append(kv_len)

    wrapper = CascadeBatchAttentionWrapper(
        num_levels=num_levels,
        kv_layout="NHD",
        device="cuda",
    )
    wrapper.plan(
        meta["qo_indptr_arr"],
        meta["kv_indptr_arr"],
        meta["kv_indices_arr"],
        kv_len_arr,
        num_qo_heads, num_kv_heads, head_dim, head_dim,
        page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # Warmup
    for _ in range(warmup):
        wrapper.run(q, kv_data)
    torch.cuda.synchronize()

    # CUDA Graph capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, kv_data)

    # Benchmark with CUDA Graph replay
    medians = []
    for _ in range(rounds):
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        for i in range(repeat):
            start_events[i].record()
            graph.replay()
            end_events[i].record()
        torch.cuda.synchronize()
        times = sorted(s.elapsed_time(e) for s, e in zip(start_events, end_events))
        medians.append(times[len(times) // 2])

    if rounds == 1:
        return medians[0]
    return _trimmed_mean(medians)


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    results: List[Tuple],
    depths: List[int],
    widths: List[int],
    threshold: float = 0.15,
) -> List[Tuple[int, int, str]]:
    """Detect anomalous (depth, width, topo) configs."""
    anomalies = []
    sorted_depths = sorted(depths)

    for topo in ["best", "worst", "eagle"]:
        for w in widths:
            series = {}
            for d, ww, tp, _, _, spd in results:
                if tp == topo and ww == w:
                    series[d] = spd
            if len(series) < 3:
                continue

            ds = [d for d in sorted_depths if d in series]
            for i in range(1, len(ds) - 1):
                prev_s = series[ds[i - 1]]
                curr_s = series[ds[i]]
                next_s = series[ds[i + 1]]
                expected = (prev_s + next_s) / 2.0
                deviation = abs(curr_s - expected) / max(abs(expected), 1e-9)
                if deviation > threshold:
                    anomalies.append((ds[i], w, topo))

            if len(ds) >= 2:
                d0, d1 = ds[0], ds[1]
                if abs(series[d0] - series[d1]) / max(abs(series[d1]), 1e-9) > threshold * 2:
                    anomalies.append((d0, w, topo))
                dm1, dm2 = ds[-1], ds[-2]
                if abs(series[dm1] - series[dm2]) / max(abs(series[dm2]), 1e-9) > threshold * 2:
                    anomalies.append((dm1, w, topo))

    return anomalies


# ---------------------------------------------------------------------------
# Batch Replication
# ---------------------------------------------------------------------------

def replicate_metadata(
    single_meta: dict,
    batch_size: int,
    single_total_q: int,
    pages_per_request: int,
) -> dict:
    """Replicate single-request metadata for a batch of identical requests."""
    num_levels = single_meta["num_levels"]
    batch_total_q = single_total_q * batch_size

    batch_qo_indptr = []
    batch_kv_indptr = []
    batch_kv_indices = []
    batch_last_page_len = []

    for lvl in range(num_levels):
        s_qo = single_meta["qo_indptr_arr"][lvl].cpu().tolist()
        s_ki = single_meta["kv_indptr_arr"][lvl].cpu().tolist()
        s_kv = single_meta["kv_indices_arr"][lvl].cpu().tolist()
        s_lp = single_meta["last_page_len_arr"][lvl].cpu().tolist()

        n_groups = len(s_qo) - 1

        b_qo = [0]
        b_ki = [0]
        b_kv = []
        b_lp = []

        for b in range(batch_size):
            q_off = b * single_total_q
            p_off = b * pages_per_request

            for g in range(n_groups):
                b_qo.append(s_qo[g + 1] + q_off)

            kv_cum = b_ki[-1]
            for g in range(n_groups):
                n_pages = s_ki[g + 1] - s_ki[g]
                kv_cum += n_pages
                b_ki.append(kv_cum)

            b_kv.extend([idx + p_off for idx in s_kv])
            b_lp.extend(s_lp)

        batch_qo_indptr.append(torch.tensor(b_qo, dtype=torch.int32, device="cuda"))
        batch_kv_indptr.append(torch.tensor(b_ki, dtype=torch.int32, device="cuda"))
        batch_kv_indices.append(torch.tensor(b_kv, dtype=torch.int32, device="cuda"))
        batch_last_page_len.append(torch.tensor(b_lp, dtype=torch.int32, device="cuda"))

    return {
        "num_levels": num_levels,
        "total_q": batch_total_q,
        "qo_indptr_arr": batch_qo_indptr,
        "kv_indptr_arr": batch_kv_indptr,
        "kv_indices_arr": batch_kv_indices,
        "last_page_len_arr": batch_last_page_len,
    }
