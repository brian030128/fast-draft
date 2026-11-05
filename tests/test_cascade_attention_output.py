"""
Correctness test: CascadeBatchAttentionWrapper vs flat baselines.

Both compute the same mathematical operation (attention over prefix+suffix KV),
just organized differently. Their outputs must match.
"""

import itertools

import pytest
import torch

import flashinfer
from flashinfer.attention import CascadeBatchAttentionWrapper


# ── Helper: run cascade wrapper ──────────────────────────────────────────────
def _run_cascade(data, causal=True):
    """Run CascadeBatchAttentionWrapper and return (out, lse)."""
    total_batch = data["q"].shape[0]
    num_qo_heads = data["num_qo_heads"]
    num_kv_heads = data["shared_kv_len_tensor"].shape[0]  # num_seqs for level 0
    # Infer num_kv_heads from q shape and kv_data shape
    num_kv_heads = data["kv_data"].shape[3]
    head_dim = data["q"].shape[2]
    page_size = data["kv_data"].shape[2]
    dtype = data["q"].dtype

    cascade_wrapper = CascadeBatchAttentionWrapper(
        num_levels=2, kv_layout="NHD", device="cuda",
        use_cuda_graph=False,
    )
    cascade_wrapper.plan(
        [data["qo_indptr_shared"], data["qo_indptr_unique"]],
        [data["shared_kv_indptr"], data["unique_kv_indptr"]],
        [data["shared_kv_indices"], data["unique_kv_indices"]],
        [data["shared_kv_len_tensor"], data["unique_kv_len_tensor"]],
        num_qo_heads, num_kv_heads, head_dim, head_dim,
        page_size,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    cascade_out = torch.empty_like(data["q"])
    cascade_lse = torch.empty(total_batch, num_qo_heads, device="cuda", dtype=torch.float32)
    cascade_wrapper.run(data["q"], data["kv_data"], out=cascade_out, lse=cascade_lse)
    return cascade_out, cascade_lse


def _unwrap(result):
    """Unwrap output that may be (out, lse) tuple or just out."""
    if isinstance(result, tuple):
        return result[0]
    return result


def ceil_div(a, b):
    return (a + b - 1) // b


def build_kv_cache_and_indices(
    num_seqs: int,
    topk: int,
    prefix_len: int,
    suffix_len: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
):
    """Build a paged KV cache with shared prefix + per-branch unique suffix.

    Returns flat and cascade index structures, plus the KV data and query tensor.

    Layout: num_seqs independent sequences, each with topk branches sharing a prefix.
    Total branches = num_seqs * topk.
    """
    total_batch = num_seqs * topk
    num_prefix_pages = ceil_div(prefix_len, page_size)
    num_suffix_pages = ceil_div(suffix_len, page_size)
    total_prefix_pages = num_seqs * num_prefix_pages
    total_suffix_pages = total_batch * num_suffix_pages
    total_pages = total_prefix_pages + total_suffix_pages

    # Allocate paged KV cache: [num_pages, 2, page_size, num_kv_heads, head_dim]
    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        device="cuda", dtype=dtype,
    )

    # Zero out padding in last pages so it doesn't affect results
    prefix_last_page_valid = (prefix_len - 1) % page_size + 1 if prefix_len > 0 else 0
    suffix_last_page_valid = (suffix_len - 1) % page_size + 1 if suffix_len > 0 else 0

    # Zero padding in prefix last pages
    if prefix_last_page_valid < page_size:
        for s in range(num_seqs):
            last_page = s * num_prefix_pages + num_prefix_pages - 1
            kv_data[last_page, :, prefix_last_page_valid:, :, :] = 0.0

    # Zero padding in suffix last pages
    if suffix_last_page_valid < page_size:
        for b in range(total_batch):
            last_page = total_prefix_pages + b * num_suffix_pages + num_suffix_pages - 1
            kv_data[last_page, :, suffix_last_page_valid:, :, :] = 0.0

    # --- Flat indices: each branch gets [prefix_pages | suffix_pages] ---
    total_kv_len = prefix_len + suffix_len
    flat_pages_per_req = num_prefix_pages + num_suffix_pages
    flat_kv_indices_list = []
    for b in range(total_batch):
        seq_idx = b // topk
        # Prefix pages for this sequence
        prefix_start = seq_idx * num_prefix_pages
        flat_kv_indices_list.append(
            torch.arange(prefix_start, prefix_start + num_prefix_pages,
                         device="cuda", dtype=torch.int32)
        )
        # Suffix pages for this branch
        suffix_start = total_prefix_pages + b * num_suffix_pages
        flat_kv_indices_list.append(
            torch.arange(suffix_start, suffix_start + num_suffix_pages,
                         device="cuda", dtype=torch.int32)
        )
    flat_kv_indices = torch.cat(flat_kv_indices_list)
    flat_kv_indptr = (
        torch.arange(total_batch + 1, device="cuda", dtype=torch.int32) * flat_pages_per_req
    )
    flat_last_page_len = torch.full(
        (total_batch,), (total_kv_len - 1) % page_size + 1,
        device="cuda", dtype=torch.int32,
    )

    # --- Cascade indices ---
    # Level 0 (shared prefix): num_seqs entries
    shared_kv_indices_list = []
    shared_kv_indptr_list = [0]
    for s in range(num_seqs):
        prefix_start = s * num_prefix_pages
        shared_kv_indices_list.append(
            torch.arange(prefix_start, prefix_start + num_prefix_pages,
                         device="cuda", dtype=torch.int32)
        )
        shared_kv_indptr_list.append(shared_kv_indptr_list[-1] + num_prefix_pages)
    shared_kv_indices = torch.cat(shared_kv_indices_list)
    shared_kv_indptr = torch.tensor(shared_kv_indptr_list, device="cuda", dtype=torch.int32)
    shared_kv_len_tensor = torch.full(
        (num_seqs,), prefix_len, device="cuda", dtype=torch.int32,
    )

    # Level 1 (unique suffix): total_batch entries
    unique_kv_indices = (
        torch.arange(total_suffix_pages, device="cuda", dtype=torch.int32)
        + total_prefix_pages
    )
    unique_kv_indptr = (
        torch.arange(total_batch + 1, device="cuda", dtype=torch.int32) * num_suffix_pages
    )
    unique_kv_len_tensor = torch.full(
        (total_batch,), suffix_len, device="cuda", dtype=torch.int32,
    )

    # qo_indptr for cascade levels
    # Level 0: groups topk queries per sequence
    qo_indptr_shared = (
        torch.arange(num_seqs + 1, device="cuda", dtype=torch.int32) * topk
    )
    # Level 1: one query per branch
    qo_indptr_unique = torch.arange(total_batch + 1, device="cuda", dtype=torch.int32)

    # Query tensor: one query per branch (decode)
    num_qo_heads = 32
    q = torch.randn(total_batch, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    return {
        "kv_data": kv_data,
        "q": q,
        "num_qo_heads": num_qo_heads,
        # Flat
        "flat_kv_indices": flat_kv_indices,
        "flat_kv_indptr": flat_kv_indptr,
        "flat_last_page_len": flat_last_page_len,
        # Cascade
        "shared_kv_indices": shared_kv_indices,
        "shared_kv_indptr": shared_kv_indptr,
        "shared_kv_len_tensor": shared_kv_len_tensor,
        "unique_kv_indices": unique_kv_indices,
        "unique_kv_indptr": unique_kv_indptr,
        "unique_kv_len_tensor": unique_kv_len_tensor,
        "qo_indptr_shared": qo_indptr_shared,
        "qo_indptr_unique": qo_indptr_unique,
    }


# Test configurations from the plan
@pytest.mark.parametrize(
    "num_seqs,topk,prefix_len,suffix_len,page_size",
    list(itertools.product(
        [1, 4],        # num_seqs
        [1, 4],        # topk
        [128, 1024],   # prefix_len
        [1, 3, 5],     # suffix_len
        [1, 16],       # page_size
    )),
)
def test_cascade_vs_flat(num_seqs, topk, prefix_len, suffix_len, page_size):
    """CascadeBatchAttentionWrapper must match flat BatchDecodeWithPagedKVCacheWrapper."""
    torch.manual_seed(42)

    num_kv_heads = 8
    head_dim = 128
    dtype = torch.float16

    data = build_kv_cache_and_indices(
        num_seqs, topk, prefix_len, suffix_len,
        num_kv_heads, head_dim, page_size, dtype,
    )

    total_batch = num_seqs * topk
    q = data["q"]
    kv_data = data["kv_data"]
    num_qo_heads = data["num_qo_heads"]

    # --- Flat baseline ---
    flat_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
    )
    flat_wrapper.plan(
        data["flat_kv_indptr"],
        data["flat_kv_indices"],
        data["flat_last_page_len"],
        num_qo_heads, num_kv_heads, head_dim, page_size,
        q_data_type=dtype,
    )
    flat_out = flat_wrapper.run(q, kv_data)

    # --- Cascade ---
    total_kv_pages = data["shared_kv_indices"].shape[0] + data["unique_kv_indices"].shape[0]
    kv_indices_buffer = torch.empty(total_kv_pages, device="cuda", dtype=torch.int32)

    cascade_wrapper = CascadeBatchAttentionWrapper(
        num_levels=2, kv_layout="NHD", device="cuda",
        use_cuda_graph=False,
    )
    cascade_wrapper.plan(
        [data["qo_indptr_shared"], data["qo_indptr_unique"]],
        [data["shared_kv_indptr"], data["unique_kv_indptr"]],
        [data["shared_kv_indices"], data["unique_kv_indices"]],
        [data["shared_kv_len_tensor"], data["unique_kv_len_tensor"]],
        num_qo_heads, num_kv_heads, head_dim, head_dim,
        page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    cascade_out = torch.empty_like(q)
    cascade_lse = torch.empty(total_batch, num_qo_heads, device="cuda", dtype=torch.float32)
    cascade_wrapper.run(q, kv_data, out=cascade_out, lse=cascade_lse)

    # --- Compare ---
    # flat_out may be a tuple (out, lse) or just out depending on version
    if isinstance(flat_out, tuple):
        flat_out_tensor = flat_out[0]
    else:
        flat_out_tensor = flat_out

    atol, rtol = 1e-3, 1e-3
    if not torch.allclose(flat_out_tensor, cascade_out, atol=atol, rtol=rtol):
        diff = (flat_out_tensor - cascade_out).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        pytest.fail(
            f"Cascade output differs from flat!\n"
            f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}\n"
            f"  config: num_seqs={num_seqs}, topk={topk}, "
            f"prefix_len={prefix_len}, suffix_len={suffix_len}, page_size={page_size}"
        )


# Also test against MultiLevelCascadeAttentionWrapper as a second reference
@pytest.mark.parametrize(
    "num_seqs,topk,prefix_len,suffix_len,page_size",
    [
        (1, 4, 128, 5, 16),
        (4, 4, 1024, 3, 1),
    ],
)
def test_cascade_vs_multilevel(num_seqs, topk, prefix_len, suffix_len, page_size):
    """CascadeBatchAttentionWrapper must match MultiLevelCascadeAttentionWrapper."""
    torch.manual_seed(42)

    num_kv_heads = 8
    head_dim = 128
    dtype = torch.float16

    data = build_kv_cache_and_indices(
        num_seqs, topk, prefix_len, suffix_len,
        num_kv_heads, head_dim, page_size, dtype,
    )

    total_batch = num_seqs * topk
    q = data["q"]
    kv_data = data["kv_data"]
    num_qo_heads = data["num_qo_heads"]

    # Shared last_page_len for multilevel
    shared_last_page_len = torch.full(
        (num_seqs,), (prefix_len - 1) % page_size + 1,
        device="cuda", dtype=torch.int32,
    )
    unique_last_page_len = torch.full(
        (total_batch,), (suffix_len - 1) % page_size + 1,
        device="cuda", dtype=torch.int32,
    )

    # --- MultiLevel reference ---
    ref_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        2, torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda"), "NHD"
    )
    ref_wrapper.plan(
        [data["qo_indptr_shared"], data["qo_indptr_unique"]],
        [data["shared_kv_indptr"], data["unique_kv_indptr"]],
        [data["shared_kv_indices"], data["unique_kv_indices"]],
        [shared_last_page_len, unique_last_page_len],
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True,
        q_data_type=dtype,
    )
    ref_out = ref_wrapper.run(q, kv_data)

    # --- Cascade ---
    cascade_wrapper = CascadeBatchAttentionWrapper(
        num_levels=2, kv_layout="NHD", device="cuda",
        use_cuda_graph=False,
    )
    cascade_wrapper.plan(
        [data["qo_indptr_shared"], data["qo_indptr_unique"]],
        [data["shared_kv_indptr"], data["unique_kv_indptr"]],
        [data["shared_kv_indices"], data["unique_kv_indices"]],
        [data["shared_kv_len_tensor"], data["unique_kv_len_tensor"]],
        num_qo_heads, num_kv_heads, head_dim, head_dim,
        page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    cascade_out = torch.empty_like(q)
    cascade_lse = torch.empty(total_batch, num_qo_heads, device="cuda", dtype=torch.float32)
    cascade_wrapper.run(q, kv_data, out=cascade_out, lse=cascade_lse)

    # --- Compare ---
    if isinstance(ref_out, tuple):
        ref_out_tensor = ref_out[0]
    else:
        ref_out_tensor = ref_out

    atol, rtol = 1e-3, 1e-3
    if not torch.allclose(ref_out_tensor, cascade_out, atol=atol, rtol=rtol):
        diff = (ref_out_tensor - cascade_out).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        pytest.fail(
            f"Cascade output differs from MultiLevel!\n"
            f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}\n"
            f"  config: num_seqs={num_seqs}, topk={topk}, "
            f"prefix_len={prefix_len}, suffix_len={suffix_len}, page_size={page_size}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC TESTS — Step 1: Isolate the root cause
# ═══════════════════════════════════════════════════════════════════════════════

# 1a. MultiLevel vs Flat — validates our test setup (indices, KV data)
@pytest.mark.parametrize(
    "num_seqs,topk,prefix_len,suffix_len,page_size",
    [
        (1, 1, 128, 1, 1),
        (1, 4, 128, 5, 16),
        (4, 4, 1024, 3, 1),
    ],
)
def test_diag_multilevel_vs_flat(num_seqs, topk, prefix_len, suffix_len, page_size):
    """If MultiLevel matches flat, our test setup is correct."""
    torch.manual_seed(42)
    num_kv_heads, head_dim, dtype = 8, 128, torch.float16

    data = build_kv_cache_and_indices(
        num_seqs, topk, prefix_len, suffix_len,
        num_kv_heads, head_dim, page_size, dtype,
    )
    total_batch = num_seqs * topk
    q, kv_data = data["q"], data["kv_data"]
    num_qo_heads = data["num_qo_heads"]

    # Flat baseline
    flat_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
    )
    flat_wrapper.plan(
        data["flat_kv_indptr"], data["flat_kv_indices"],
        data["flat_last_page_len"],
        num_qo_heads, num_kv_heads, head_dim, page_size,
        q_data_type=dtype,
    )
    flat_out = _unwrap(flat_wrapper.run(q, kv_data))

    # MultiLevel
    shared_last_page_len = torch.full(
        (num_seqs,), (prefix_len - 1) % page_size + 1,
        device="cuda", dtype=torch.int32,
    )
    unique_last_page_len = torch.full(
        (total_batch,), (suffix_len - 1) % page_size + 1,
        device="cuda", dtype=torch.int32,
    )
    ml_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        2, torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda"), "NHD"
    )
    ml_wrapper.plan(
        [data["qo_indptr_shared"], data["qo_indptr_unique"]],
        [data["shared_kv_indptr"], data["unique_kv_indptr"]],
        [data["shared_kv_indices"], data["unique_kv_indices"]],
        [shared_last_page_len, unique_last_page_len],
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True, q_data_type=dtype,
    )
    ml_out = _unwrap(ml_wrapper.run(q, kv_data))

    diff = (flat_out - ml_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"\n  [1a] MultiLevel vs Flat: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    assert max_diff < 1e-2, (
        f"MultiLevel vs Flat mismatch! max_diff={max_diff:.6f} — test setup may be wrong"
    )


# 1b. Cascade vs Flat using BatchPrefillWithPagedKVCacheWrapper
@pytest.mark.parametrize(
    "num_seqs,topk,prefix_len,suffix_len,page_size",
    [
        (1, 1, 128, 1, 1),
        (1, 4, 128, 5, 16),
    ],
)
def test_diag_cascade_vs_flat_prefill(num_seqs, topk, prefix_len, suffix_len, page_size):
    """Compare cascade against flat-prefill (both use persistent kernel)."""
    torch.manual_seed(42)
    num_kv_heads, head_dim, dtype = 8, 128, torch.float16
    total_kv_len = prefix_len + suffix_len

    data = build_kv_cache_and_indices(
        num_seqs, topk, prefix_len, suffix_len,
        num_kv_heads, head_dim, page_size, dtype,
    )
    total_batch = num_seqs * topk
    q, kv_data = data["q"], data["kv_data"]
    num_qo_heads = data["num_qo_heads"]

    # Flat prefill baseline (qo_len=1 per request, like decode)
    pf_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.zeros(384 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
    )
    # For prefill, qo_indptr is [0, 1, 2, ..., total_batch]
    qo_indptr = torch.arange(total_batch + 1, device="cuda", dtype=torch.int32)
    pf_wrapper.plan(
        qo_indptr,
        data["flat_kv_indptr"],
        data["flat_kv_indices"],
        data["flat_last_page_len"],
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True,
        q_data_type=dtype,
    )
    pf_out = _unwrap(pf_wrapper.run(q, kv_data))

    # Cascade
    cascade_out, _ = _run_cascade(data, causal=True)

    diff = (pf_out - cascade_out).abs()
    max_diff = diff.max().item()
    assert max_diff < 1e-2, (
        f"Cascade vs FlatPrefill mismatch! max_diff={max_diff:.6f}"
    )


# 1c. Cascade with causal=False vs flat non-causal
@pytest.mark.parametrize(
    "num_seqs,topk,prefix_len,suffix_len,page_size",
    [
        (1, 1, 128, 1, 1),
        (1, 4, 128, 5, 16),
    ],
)
def test_diag_cascade_noncausal(num_seqs, topk, prefix_len, suffix_len, page_size):
    """If cascade matches flat when causal=False, the bug is in causal masking."""
    torch.manual_seed(42)
    num_kv_heads, head_dim, dtype = 8, 128, torch.float16

    data = build_kv_cache_and_indices(
        num_seqs, topk, prefix_len, suffix_len,
        num_kv_heads, head_dim, page_size, dtype,
    )
    total_batch = num_seqs * topk
    q, kv_data = data["q"], data["kv_data"]
    num_qo_heads = data["num_qo_heads"]

    # Flat prefill non-causal baseline
    pf_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.zeros(384 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
    )
    qo_indptr = torch.arange(total_batch + 1, device="cuda", dtype=torch.int32)
    pf_wrapper.plan(
        qo_indptr,
        data["flat_kv_indptr"],
        data["flat_kv_indices"],
        data["flat_last_page_len"],
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=False,
        q_data_type=dtype,
    )
    flat_noncausal_out = _unwrap(pf_wrapper.run(q, kv_data))

    # Cascade non-causal
    cascade_out, _ = _run_cascade(data, causal=False)

    diff = (flat_noncausal_out - cascade_out).abs()
    max_diff = diff.max().item()
    assert max_diff < 1e-2, (
        f"Cascade(causal=F) vs Flat(causal=F) mismatch! max_diff={max_diff:.6f}"
    )


# 1d. Minimal single-head test for manual tracing
def test_diag_minimal_single_head():
    """Minimal config: 1 head, small sizes, for easy manual tracing."""
    torch.manual_seed(42)
    num_seqs, topk, prefix_len, suffix_len, page_size = 1, 1, 16, 1, 1
    num_kv_heads, head_dim, dtype = 1, 128, torch.float16
    num_qo_heads = 1
    total_batch = num_seqs * topk
    total_kv_len = prefix_len + suffix_len

    # Build minimal KV cache
    num_prefix_pages = prefix_len  # page_size=1
    num_suffix_pages = suffix_len
    total_pages = num_prefix_pages + num_suffix_pages
    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        device="cuda", dtype=dtype,
    )
    q = torch.randn(1, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    # Flat indices
    flat_kv_indices = torch.arange(total_pages, device="cuda", dtype=torch.int32)
    flat_kv_indptr = torch.tensor([0, total_pages], device="cuda", dtype=torch.int32)
    flat_last_page_len = torch.tensor([1], device="cuda", dtype=torch.int32)

    # Flat baseline
    flat_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
    )
    flat_wrapper.plan(
        flat_kv_indptr, flat_kv_indices, flat_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        q_data_type=dtype,
    )
    flat_out = _unwrap(flat_wrapper.run(q, kv_data))

    # Cascade indices
    shared_kv_indices = torch.arange(num_prefix_pages, device="cuda", dtype=torch.int32)
    shared_kv_indptr = torch.tensor([0, num_prefix_pages], device="cuda", dtype=torch.int32)
    shared_kv_len = torch.tensor([prefix_len], device="cuda", dtype=torch.int32)
    unique_kv_indices = torch.arange(num_prefix_pages, total_pages, device="cuda", dtype=torch.int32)
    unique_kv_indptr = torch.tensor([0, num_suffix_pages], device="cuda", dtype=torch.int32)
    unique_kv_len = torch.tensor([suffix_len], device="cuda", dtype=torch.int32)
    qo_indptr_shared = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    qo_indptr_unique = torch.tensor([0, 1], device="cuda", dtype=torch.int32)

    cascade_wrapper = CascadeBatchAttentionWrapper(
        num_levels=2, kv_layout="NHD", device="cuda", use_cuda_graph=False,
    )
    cascade_wrapper.plan(
        [qo_indptr_shared, qo_indptr_unique],
        [shared_kv_indptr, unique_kv_indptr],
        [shared_kv_indices, unique_kv_indices],
        [shared_kv_len, unique_kv_len],
        num_qo_heads, num_kv_heads, head_dim, head_dim,
        page_size, causal=True, q_data_type=dtype, kv_data_type=dtype,
    )
    cascade_out = torch.empty_like(q)
    cascade_lse = torch.empty(1, num_qo_heads, device="cuda", dtype=torch.float32)
    cascade_wrapper.run(q, kv_data, out=cascade_out, lse=cascade_lse)

    diff = (flat_out - cascade_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"\n  [1d] Minimal single-head: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    assert max_diff < 1e-2, f"Even minimal case fails! max_diff={max_diff:.6f}"


# 1e. GQA isolation: vary num_qo_heads with fixed num_kv_heads
@pytest.mark.parametrize("num_qo_heads", [1, 2, 4, 8, 16, 32])
def test_diag_gqa_sweep(num_qo_heads):
    """Sweep GQA ratio to find where the bug appears."""
    torch.manual_seed(42)
    num_seqs, topk, prefix_len, suffix_len, page_size = 1, 1, 128, 1, 1
    num_kv_heads, head_dim, dtype = 8, 128, torch.float16
    if num_qo_heads < num_kv_heads:
        pytest.skip("num_qo_heads must be >= num_kv_heads")
    total_batch = 1
    total_kv_len = prefix_len + suffix_len

    # Build KV
    num_prefix_pages = prefix_len
    num_suffix_pages = suffix_len
    total_pages = num_prefix_pages + num_suffix_pages
    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        device="cuda", dtype=dtype,
    )
    q = torch.randn(1, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    # Flat
    flat_kv_indices = torch.arange(total_pages, device="cuda", dtype=torch.int32)
    flat_kv_indptr = torch.tensor([0, total_pages], device="cuda", dtype=torch.int32)
    flat_last_page_len = torch.tensor([1], device="cuda", dtype=torch.int32)
    flat_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
    )
    flat_wrapper.plan(
        flat_kv_indptr, flat_kv_indices, flat_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size, q_data_type=dtype,
    )
    flat_out = _unwrap(flat_wrapper.run(q, kv_data))

    # Cascade
    shared_kv_indices = torch.arange(num_prefix_pages, device="cuda", dtype=torch.int32)
    shared_kv_indptr = torch.tensor([0, num_prefix_pages], device="cuda", dtype=torch.int32)
    shared_kv_len = torch.tensor([prefix_len], device="cuda", dtype=torch.int32)
    unique_kv_indices = torch.arange(num_prefix_pages, total_pages, device="cuda", dtype=torch.int32)
    unique_kv_indptr = torch.tensor([0, num_suffix_pages], device="cuda", dtype=torch.int32)
    unique_kv_len = torch.tensor([suffix_len], device="cuda", dtype=torch.int32)
    qo_indptr_shared = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    qo_indptr_unique = torch.tensor([0, 1], device="cuda", dtype=torch.int32)

    cascade_wrapper = CascadeBatchAttentionWrapper(
        num_levels=2, kv_layout="NHD", device="cuda", use_cuda_graph=False,
    )
    cascade_wrapper.plan(
        [qo_indptr_shared, qo_indptr_unique],
        [shared_kv_indptr, unique_kv_indptr],
        [shared_kv_indices, unique_kv_indices],
        [shared_kv_len, unique_kv_len],
        num_qo_heads, num_kv_heads, head_dim, head_dim,
        page_size, causal=True, q_data_type=dtype, kv_data_type=dtype,
    )
    cascade_out = torch.empty_like(q)
    cascade_lse = torch.empty(1, num_qo_heads, device="cuda", dtype=torch.float32)
    cascade_wrapper.run(q, kv_data, out=cascade_out, lse=cascade_lse)

    diff = (flat_out - cascade_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    gqa_ratio = num_qo_heads // num_kv_heads
    print(f"\n  [1e] GQA={gqa_ratio}x ({num_qo_heads}/{num_kv_heads}): max_diff={max_diff:.6f}")
    assert max_diff < 1e-2, f"GQA={gqa_ratio}x fails! max_diff={max_diff:.6f}"
