"""
Correctness test: CascadeBatchAttentionWrapper with N > 2 levels.

Verifies that the fused cascade kernel produces the same output as the unfused
MultiLevelCascadeAttentionWrapper reference for 2, 3, and 4 levels.

Usage:
    pytest tests/test_cascade_multilevel.py -v
    python tests/test_cascade_multilevel.py
"""

import pytest
import torch

import flashinfer
from flashinfer.attention import CascadeBatchAttentionWrapper


def ceil_div(a, b):
    return (a + b - 1) // b


def build_multilevel_kv_cache(
    kv_lens_per_level,
    batch_sizes_per_level,
    num_kv_heads,
    head_dim,
    page_size,
    dtype=torch.bfloat16,
):
    """Build an N-level paged KV cache with random data.

    Args:
        kv_lens_per_level: list of KV lengths for each level
        batch_sizes_per_level: list of batch sizes for each level
            (coarsest first, e.g. [1, num_groups, total_batch])
        num_kv_heads: number of KV heads
        head_dim: head dimension
        page_size: paging granularity
        dtype: data type

    Returns:
        kv_data: the paged KV cache tensor
        level_metadata: list of dicts per level with keys:
            kv_indices, kv_indptr, last_page_len, kv_len_tensor
    """
    num_levels = len(kv_lens_per_level)
    assert len(batch_sizes_per_level) == num_levels

    # Compute total pages needed
    total_pages = 0
    pages_per_level = []
    for kv_len, bs in zip(kv_lens_per_level, batch_sizes_per_level):
        pages_per_req = ceil_div(kv_len, page_size)
        pages_per_level.append(pages_per_req)
        total_pages += bs * pages_per_req

    kv_data = torch.zeros(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        device="cuda", dtype=dtype,
    )

    page_offset = 0
    level_metadata = []

    for level in range(num_levels):
        kv_len = kv_lens_per_level[level]
        bs = batch_sizes_per_level[level]
        pages_per_req = pages_per_level[level]

        # Fill pages with random data
        k_data = torch.randn(bs * kv_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        v_data = torch.randn(bs * kv_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)

        kv_indices = (
            torch.arange(bs * pages_per_req, device="cuda", dtype=torch.int32)
            + page_offset
        )
        kv_indptr = (
            torch.arange(bs + 1, device="cuda", dtype=torch.int32) * pages_per_req
        )
        last_page_len = torch.full(
            (bs,), (kv_len - 1) % page_size + 1, device="cuda", dtype=torch.int32
        )

        append_indptr = torch.arange(bs + 1, device="cuda", dtype=torch.int32) * kv_len
        flashinfer.append_paged_kv_cache(
            k_data, v_data,
            *flashinfer.get_batch_indices_positions(
                append_indptr,
                flashinfer.get_seq_lens(kv_indptr, last_page_len, page_size),
                bs * kv_len,
            ),
            kv_data, kv_indices, kv_indptr, last_page_len, "NHD",
        )

        kv_len_tensor = torch.full(
            (bs,), kv_len, device="cuda", dtype=torch.int32
        )

        level_metadata.append(dict(
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            last_page_len=last_page_len,
            kv_len_tensor=kv_len_tensor,
        ))

        page_offset += bs * pages_per_req

    return kv_data, level_metadata


def build_qo_indptr_arr(batch_sizes_per_level, total_batch, qo_len):
    """Build qo_indptr arrays for each level.

    For level i with batch_size B_i, each "request" at that level covers
    (total_batch / B_i) * qo_len query tokens.
    """
    qo_indptr_arr = []
    for bs in batch_sizes_per_level:
        queries_per_req = (total_batch // bs) * qo_len
        qo_indptr = (
            torch.arange(bs + 1, device="cuda", dtype=torch.int32) * queries_per_req
        )
        qo_indptr_arr.append(qo_indptr)
    return qo_indptr_arr


def run_multilevel_reference(q, kv_data, level_metadata, batch_sizes_per_level,
                             total_batch, qo_len, num_qo_heads, num_kv_heads,
                             head_dim, page_size, dtype):
    """Run MultiLevelCascadeAttentionWrapper as reference."""
    num_levels = len(level_metadata)
    ref_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        num_levels, torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda"), "NHD"
    )
    qo_indptr_arr = build_qo_indptr_arr(batch_sizes_per_level, total_batch, qo_len)
    ref_wrapper.plan(
        qo_indptr_arr,
        [m["kv_indptr"] for m in level_metadata],
        [m["kv_indices"] for m in level_metadata],
        [m["last_page_len"] for m in level_metadata],
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True, q_data_type=dtype,
    )
    return ref_wrapper.run(q, kv_data)


def run_fused_cascade(q, kv_data, level_metadata, batch_sizes_per_level,
                      total_batch, qo_len, num_qo_heads, num_kv_heads,
                      head_dim, page_size, dtype):
    """Run CascadeBatchAttentionWrapper (fused)."""
    num_levels = len(level_metadata)
    cascade = CascadeBatchAttentionWrapper(
        num_levels=num_levels, kv_layout="NHD", device="cuda",
    )
    qo_indptr_arr = build_qo_indptr_arr(batch_sizes_per_level, total_batch, qo_len)
    cascade.plan(
        qo_indptr_arr,
        [m["kv_indptr"] for m in level_metadata],
        [m["kv_indices"] for m in level_metadata],
        [m["kv_len_tensor"] for m in level_metadata],
        num_qo_heads, num_kv_heads, head_dim, head_dim,
        page_size, causal=True,
        q_data_type=dtype, kv_data_type=dtype,
    )
    out, lse = cascade.run(q, kv_data)
    return out


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------

# (num_levels, kv_lens_per_level, batch_sizes_per_level, total_batch)
# batch_sizes must divide total_batch and go coarsest -> finest

CONFIGS_2_LEVEL = [
    # Classic shared-prefix: 1 prefix, 16 suffixes
    (2, [512, 5], [1, 16], 16),
    (2, [2048, 5], [4, 16], 16),
]

CONFIGS_3_LEVEL = [
    # 3-level: global prefix -> group prefix -> per-request suffix
    (3, [1024, 128, 5], [1, 4, 16], 16),
    (3, [512, 64, 8], [1, 2, 8], 8),
    (3, [2048, 256, 5], [1, 4, 32], 32),
]

CONFIGS_4_LEVEL = [
    # 4-level deep hierarchy
    (4, [1024, 256, 64, 5], [1, 2, 4, 16], 16),
]

ALL_CONFIGS = CONFIGS_2_LEVEL + CONFIGS_3_LEVEL + CONFIGS_4_LEVEL


@pytest.mark.parametrize(
    "num_levels,kv_lens,batch_sizes,total_batch",
    ALL_CONFIGS,
    ids=[
        f"{nl}L-kv{'_'.join(map(str, kvl))}-bs{'_'.join(map(str, bs))}"
        for nl, kvl, bs, _ in ALL_CONFIGS
    ],
)
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(32, 8), (8, 8)])
def test_multilevel_correctness(
    num_levels, kv_lens, batch_sizes, total_batch,
    num_qo_heads, num_kv_heads,
):
    torch.manual_seed(42)
    head_dim = 128
    page_size = 16
    qo_len = 1
    dtype = torch.bfloat16

    kv_data, level_metadata = build_multilevel_kv_cache(
        kv_lens, batch_sizes, num_kv_heads, head_dim, page_size, dtype=dtype,
    )

    q = torch.randn(total_batch * qo_len, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    ref_out = run_multilevel_reference(
        q, kv_data, level_metadata, batch_sizes, total_batch, qo_len,
        num_qo_heads, num_kv_heads, head_dim, page_size, dtype,
    )
    fused_out = run_fused_cascade(
        q, kv_data, level_metadata, batch_sizes, total_batch, qo_len,
        num_qo_heads, num_kv_heads, head_dim, page_size, dtype,
    )

    torch.testing.assert_close(fused_out, ref_out, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Standalone runner with debug output
# ---------------------------------------------------------------------------

def run_single_test(num_levels, kv_lens, batch_sizes, total_batch):
    """Run a single config with debug output."""
    torch.manual_seed(42)
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    qo_len = 1
    dtype = torch.bfloat16

    kv_data, level_metadata = build_multilevel_kv_cache(
        kv_lens, batch_sizes, num_kv_heads, head_dim, page_size, dtype=dtype,
    )

    q = torch.randn(total_batch * qo_len, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    ref_out = run_multilevel_reference(
        q, kv_data, level_metadata, batch_sizes, total_batch, qo_len,
        num_qo_heads, num_kv_heads, head_dim, page_size, dtype,
    )
    fused_out = run_fused_cascade(
        q, kv_data, level_metadata, batch_sizes, total_batch, qo_len,
        num_qo_heads, num_kv_heads, head_dim, page_size, dtype,
    )

    abs_diff = (fused_out.float() - ref_out.float()).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    close_frac = (abs_diff < 0.01).float().mean().item()
    print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
          f"frac_close(<0.01)={close_frac:.4f}")
    return max_diff


if __name__ == "__main__":
    for num_levels, kv_lens, batch_sizes, total_batch in ALL_CONFIGS:
        tag = f"{num_levels}L kv_lens={kv_lens} batch_sizes={batch_sizes}"
        print(f"Testing {tag}:")
        max_diff = run_single_test(num_levels, kv_lens, batch_sizes, total_batch)
        passed = max_diff < 0.02
        print(f"  {'PASS' if passed else 'FAIL'}")
    print("\nDone.")
