"""Tests for plan_for_draft + update_draft_step correctness.

Verifies that planning once at max draft depth and patching per step via
update_draft_step produces identical attention output to planning each step
independently.

This is the key correctness test for the cascade plan overhead optimization.

Run:
    CUDA_VISIBLE_DEVICES=1 uv run pytest tests/test_plan_for_draft.py -v
"""

import itertools

import pytest
import torch
from flashinfer.attention import CascadeBatchAttentionWrapper


def ceil_div(a, b):
    return (a + b - 1) // b


def build_draft_kv_data(
    num_seqs: int,
    topk: int,
    prefix_len: int,
    max_suffix_len: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
):
    """Build paged KV cache for draft decode testing.

    Creates one shared KV cache at max_suffix_len. For smaller suffix steps,
    the attention just reads fewer suffix pages (controlled by kv_len).

    Returns kv_data, q, and index builders for any suffix_len <= max_suffix_len.
    """
    total_batch = num_seqs * topk
    num_prefix_pages = ceil_div(prefix_len, page_size)
    num_max_suffix_pages = ceil_div(max_suffix_len, page_size)
    total_prefix_pages = num_seqs * num_prefix_pages
    total_max_suffix_pages = total_batch * num_max_suffix_pages
    total_pages = total_prefix_pages + total_max_suffix_pages

    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        device="cuda", dtype=dtype,
    )

    # Zero padding in last prefix pages
    prefix_last_valid = (prefix_len - 1) % page_size + 1 if prefix_len > 0 else 0
    if prefix_last_valid < page_size:
        for s in range(num_seqs):
            last_page = s * num_prefix_pages + num_prefix_pages - 1
            kv_data[last_page, :, prefix_last_valid:, :, :] = 0.0

    # Zero padding in last suffix pages (at max depth)
    suffix_last_valid = (max_suffix_len - 1) % page_size + 1 if max_suffix_len > 0 else 0
    if suffix_last_valid < page_size:
        for b in range(total_batch):
            last_page = total_prefix_pages + b * num_max_suffix_pages + num_max_suffix_pages - 1
            kv_data[last_page, :, suffix_last_valid:, :, :] = 0.0

    num_qo_heads = 32  # standard for Llama
    q = torch.randn(total_batch, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    return {
        "kv_data": kv_data,
        "q": q,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "page_size": page_size,
        "num_seqs": num_seqs,
        "topk": topk,
        "total_batch": total_batch,
        "prefix_len": prefix_len,
        "num_prefix_pages": num_prefix_pages,
        "num_max_suffix_pages": num_max_suffix_pages,
        "total_prefix_pages": total_prefix_pages,
    }


def build_cascade_indices(data, suffix_len):
    """Build cascade indices for a given suffix_len using data's KV cache layout."""
    num_seqs = data["num_seqs"]
    topk = data["topk"]
    total_batch = data["total_batch"]
    prefix_len = data["prefix_len"]
    num_prefix_pages = data["num_prefix_pages"]
    num_max_suffix_pages = data["num_max_suffix_pages"]
    total_prefix_pages = data["total_prefix_pages"]
    page_size = data["page_size"]

    num_suffix_pages = ceil_div(suffix_len, page_size) if suffix_len > 0 else 0

    # Level 0 (shared prefix)
    shared_kv_indices_list = []
    for s in range(num_seqs):
        base = s * num_prefix_pages
        shared_kv_indices_list.append(
            torch.arange(base, base + num_prefix_pages, device="cuda", dtype=torch.int32)
        )
    shared_kv_indices = torch.cat(shared_kv_indices_list)
    shared_kv_indptr = torch.arange(
        0, (num_seqs + 1) * num_prefix_pages, num_prefix_pages,
        device="cuda", dtype=torch.int32,
    )[:num_seqs + 1]
    shared_kv_len = torch.full((num_seqs,), prefix_len, device="cuda", dtype=torch.int32)

    # Level 1 (unique suffix) — index into same KV data pages as max depth,
    # but only the first num_suffix_pages per branch
    unique_kv_indices_list = []
    for b in range(total_batch):
        base = total_prefix_pages + b * num_max_suffix_pages
        unique_kv_indices_list.append(
            torch.arange(base, base + num_suffix_pages, device="cuda", dtype=torch.int32)
        )
    unique_kv_indices = torch.cat(unique_kv_indices_list) if num_suffix_pages > 0 else \
        torch.empty(0, device="cuda", dtype=torch.int32)
    unique_kv_indptr = torch.arange(
        0, (total_batch + 1) * num_suffix_pages, num_suffix_pages,
        device="cuda", dtype=torch.int32,
    )[:total_batch + 1]
    unique_kv_len = torch.full((total_batch,), suffix_len, device="cuda", dtype=torch.int32)

    # qo_indptr
    qo_indptr_shared = torch.arange(
        0, (num_seqs + 1) * topk, topk, device="cuda", dtype=torch.int32,
    )[:num_seqs + 1]
    qo_indptr_unique = torch.arange(total_batch + 1, device="cuda", dtype=torch.int32)

    return {
        "shared_kv_indices": shared_kv_indices,
        "shared_kv_indptr": shared_kv_indptr,
        "shared_kv_len": shared_kv_len,
        "unique_kv_indices": unique_kv_indices,
        "unique_kv_indptr": unique_kv_indptr,
        "unique_kv_len": unique_kv_len,
        "qo_indptr_shared": qo_indptr_shared,
        "qo_indptr_unique": qo_indptr_unique,
    }


def run_per_step_plan(data, indices, causal=False):
    """Plan and run cascade attention for a specific suffix length."""
    wrapper = CascadeBatchAttentionWrapper(
        num_levels=2, kv_layout="NHD", device="cuda", use_cuda_graph=False,
    )
    wrapper.plan(
        [indices["qo_indptr_shared"], indices["qo_indptr_unique"]],
        [indices["shared_kv_indptr"], indices["unique_kv_indptr"]],
        [indices["shared_kv_indices"], indices["unique_kv_indices"]],
        [indices["shared_kv_len"], indices["unique_kv_len"]],
        data["num_qo_heads"], data["num_kv_heads"],
        data["head_dim"], data["head_dim"],
        data["page_size"],
        causal=causal,
        q_data_type=data["q"].dtype,
        kv_data_type=data["kv_data"].dtype,
    )
    out, lse = wrapper.run(data["q"], data["kv_data"])
    return out, lse


def run_plan_for_draft(data, max_indices, max_depth, step_offset, causal=False):
    """Plan once for max depth, then update_draft_step to the given step."""
    wrapper = CascadeBatchAttentionWrapper(
        num_levels=2, kv_layout="NHD", device="cuda", use_cuda_graph=False,
    )
    wrapper.plan_for_draft(
        max_draft_depth=max_depth,
        first_call=True,
        qo_indptr_host_arr=[
            max_indices["qo_indptr_shared"].cpu(),
            max_indices["qo_indptr_unique"].cpu(),
        ],
        kv_indptr_host_arr=[
            max_indices["shared_kv_indptr"].cpu(),
            max_indices["unique_kv_indptr"].cpu(),
        ],
        kv_indices_arr=[
            max_indices["shared_kv_indices"],
            max_indices["unique_kv_indices"],
        ],
        kv_len_host_arr=[
            max_indices["shared_kv_len"].cpu(),
            max_indices["unique_kv_len"].cpu(),
        ],
        num_qo_heads=data["num_qo_heads"],
        num_kv_heads=data["num_kv_heads"],
        head_dim_qk=data["head_dim"],
        head_dim_vo=data["head_dim"],
        page_size=data["page_size"],
        causal=causal,
        q_data_type=data["q"].dtype,
        kv_data_type=data["kv_data"].dtype,
    )
    # Patch for this step
    # For non-causal: kv_len_for_work = kv_len + qo_len, effective_kv_len = kv_len
    # qo_len per level-2 item = 1
    wrapper.update_draft_step(
        step_kv_len=step_offset + 1,
        step_kv_end=step_offset,
    )
    out, lse = wrapper.run(data["q"], data["kv_data"])
    return out, lse, wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# Main test: plan_for_draft + update_draft_step vs per-step plan
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "num_seqs,topk,num_qo_heads,num_kv_heads,prefix_len,num_steps,page_size",
    [
        # topk=2, GQA=4 (Llama-3.2-1B config) — Level 1 packed_qo=8 < 16
        # Both levels land in Task 1 → the bug we need to fix
        (1, 2, 32, 8, 256, 5, 1),
        (2, 2, 32, 8, 1024, 5, 1),
        # topk=4, GQA=4 — Level 1 packed_qo=16, borderline
        (1, 4, 32, 8, 512, 5, 1),
        # topk=6, GQA=8 — Level 1 packed_qo=48 > 16, original assumption holds
        (1, 6, 64, 8, 512, 5, 1),
        # topk=1, GQA=4 — degenerate: 1 branch per seq
        (1, 1, 32, 8, 256, 5, 1),
        # Large prefix
        (1, 2, 32, 8, 4096, 5, 1),
        # Multi-seq batch
        (4, 2, 32, 8, 256, 5, 1),
    ],
)
def test_plan_for_draft_vs_per_step_plan(
    num_seqs, topk, num_qo_heads, num_kv_heads, prefix_len, num_steps, page_size,
):
    """plan_for_draft + update_draft_step must match per-step plan for all steps."""
    torch.manual_seed(42)
    head_dim = 128
    dtype = torch.float16
    max_depth = num_steps - 1  # max suffix len

    # Build shared KV data at max depth
    data = build_draft_kv_data(
        num_seqs, topk, prefix_len, max_depth,
        num_kv_heads, head_dim, page_size, dtype,
    )
    # Override num_qo_heads from parameter
    data["num_qo_heads"] = num_qo_heads
    data["q"] = torch.randn(
        data["total_batch"], num_qo_heads, head_dim,
        device="cuda", dtype=dtype,
    )

    # Build max-depth indices
    max_indices = build_cascade_indices(data, max_depth)

    for step_i in range(max_depth):
        step_offset = step_i + 1  # suffix length for this step

        # --- Reference: per-step plan with exact suffix_len ---
        step_indices = build_cascade_indices(data, step_offset)
        ref_out, _ = run_per_step_plan(data, step_indices, causal=False)

        # --- Test: plan_for_draft at max depth + update_draft_step ---
        test_out, _, wrapper = run_plan_for_draft(
            data, max_indices, max_depth, step_offset, causal=False,
        )

        # Compare
        diff = (ref_out - test_out).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        assert max_diff < 1e-2, (
            f"Step {step_i} (offset={step_offset}): "
            f"plan_for_draft output differs from per-step plan!\n"
            f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}\n"
            f"  config: num_seqs={num_seqs}, topk={topk}, "
            f"GQA={num_qo_heads}/{num_kv_heads}, "
            f"prefix_len={prefix_len}, num_steps={num_steps}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Test: multiple update_draft_step calls on same wrapper (simulates EAGLE loop)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "num_seqs,topk,prefix_len,num_steps",
    [
        (1, 2, 256, 5),
        (2, 4, 512, 5),
        (1, 2, 4096, 5),
    ],
)
def test_sequential_update_draft_step(num_seqs, topk, prefix_len, num_steps):
    """Calling update_draft_step in sequence (like EAGLE loop) must produce correct output each time."""
    torch.manual_seed(42)
    num_qo_heads, num_kv_heads, head_dim = 32, 8, 128
    dtype = torch.float16
    max_depth = num_steps - 1

    data = build_draft_kv_data(
        num_seqs, topk, prefix_len, max_depth,
        num_kv_heads, head_dim, 1, dtype,
    )
    data["num_qo_heads"] = num_qo_heads
    data["q"] = torch.randn(
        data["total_batch"], num_qo_heads, head_dim,
        device="cuda", dtype=dtype,
    )

    max_indices = build_cascade_indices(data, max_depth)

    # Plan once
    wrapper = CascadeBatchAttentionWrapper(
        num_levels=2, kv_layout="NHD", device="cuda", use_cuda_graph=False,
    )
    wrapper.plan_for_draft(
        max_draft_depth=max_depth,
        first_call=True,
        qo_indptr_host_arr=[
            max_indices["qo_indptr_shared"].cpu(),
            max_indices["qo_indptr_unique"].cpu(),
        ],
        kv_indptr_host_arr=[
            max_indices["shared_kv_indptr"].cpu(),
            max_indices["unique_kv_indptr"].cpu(),
        ],
        kv_indices_arr=[
            max_indices["shared_kv_indices"],
            max_indices["unique_kv_indices"],
        ],
        kv_len_host_arr=[
            max_indices["shared_kv_len"].cpu(),
            max_indices["unique_kv_len"].cpu(),
        ],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=1,
        causal=False,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # Sequentially update and run for each step
    for step_i in range(max_depth):
        step_offset = step_i + 1

        # Reference
        step_indices = build_cascade_indices(data, step_offset)
        ref_out, _ = run_per_step_plan(data, step_indices, causal=False)

        # Update shared wrapper and run
        wrapper.update_draft_step(
            step_kv_len=step_offset + 1,
            step_kv_end=step_offset,
        )
        test_out, _ = wrapper.run(data["q"], data["kv_data"])

        diff = (ref_out - test_out).abs()
        max_diff = diff.max().item()

        assert max_diff < 1e-2, (
            f"Sequential step {step_i}: max_diff={max_diff:.6f}\n"
            f"  config: num_seqs={num_seqs}, topk={topk}, prefix_len={prefix_len}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
