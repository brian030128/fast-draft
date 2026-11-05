"""Tests for cascade index generation.

Verifies that the 2-level cascade indices (shared prefix + unique suffix)
represent the same attention pattern as the flat indices produced by SGLang's
generate_draft_decode_kv_indices.

Test matrix:
  - num_seqs: 1, 4
  - topk: 1, 4, 6
  - speculative_num_steps: 3, 6
  - page_size: 1
  - various prefix lengths

Run:
    pytest tests/test_cascade_indices.py -v
    # or standalone:
    python tests/test_cascade_indices.py
"""

import itertools

import torch
import pytest

from sglang.srt.speculative.cascade_index_gen import (
    build_shared_indices,
    build_unique_indices,
    next_power_of_2,
)


def _make_req_to_token_pool(num_seqs: int, pool_len: int, seq_lens: list[int], topk: int, num_steps: int):
    """Build a fake req_to_token pool with sequential page indices.

    Layout (page_size=1):
      For each request r:
        [0..prefix_len-1] = prefix page indices (sequential from some base)
        Then topk * num_steps slots for draft tokens, laid out as:
          topk_id=0: [draft_0, draft_1, ..., draft_{num_steps-1}]
          topk_id=1: [draft_0, draft_1, ..., draft_{num_steps-1}]
          ...
    """
    req_to_token = torch.zeros(num_seqs, pool_len, dtype=torch.int32, device="cuda")
    req_pool_indices = torch.arange(num_seqs, dtype=torch.int32, device="cuda")

    page_counter = 0
    for r in range(num_seqs):
        prefix_len = seq_lens[r]
        # Fill prefix with sequential page indices
        for i in range(prefix_len):
            req_to_token[r, i] = page_counter
            page_counter += 1
        # Fill draft token slots: topk branches * num_steps tokens each
        for k in range(topk):
            for s in range(num_steps):
                req_to_token[r, prefix_len + k * num_steps + s] = page_counter
                page_counter += 1

    return req_pool_indices, req_to_token


def _get_flat_kv_set(kv_indptr, kv_indices, branch_idx):
    """Extract the set of page indices for a given branch from flat indptr/indices."""
    start = kv_indptr[branch_idx].item()
    end = kv_indptr[branch_idx + 1].item()
    return set(kv_indices[start:end].cpu().tolist())


def _get_cascade_kv_set(
    shared_indptr, shared_indices, shared_req_idx,
    unique_indptr, unique_indices, branch_idx,
):
    """Extract the combined set of page indices from both cascade levels."""
    # Shared prefix pages for this request
    s_start = shared_indptr[shared_req_idx].item()
    s_end = shared_indptr[shared_req_idx + 1].item()
    shared_set = set(shared_indices[s_start:s_end].cpu().tolist())

    # Unique suffix pages for this branch
    u_start = unique_indptr[branch_idx].item()
    u_end = unique_indptr[branch_idx + 1].item()
    unique_set = set(unique_indices[u_start:u_end].cpu().tolist())

    return shared_set | unique_set


@pytest.fixture(params=[
    # (num_seqs, topk, num_steps, prefix_lens)
    (1, 1, 3, [64]),
    (1, 4, 3, [64]),
    (1, 6, 6, [128]),
    (2, 4, 3, [32, 64]),
    (4, 6, 6, [16, 32, 48, 64]),
    (3, 4, 3, [100, 200, 50]),
])
def scenario(request):
    return request.param


def test_shared_indices_structure(scenario):
    """Verify shared prefix indices have correct structure."""
    num_seqs, topk, num_steps, prefix_lens = scenario
    pool_len = max(prefix_lens) + topk * num_steps + 16

    req_pool_indices, req_to_token = _make_req_to_token_pool(
        num_seqs, pool_len, prefix_lens, topk, num_steps
    )
    seq_lens = torch.tensor(prefix_lens, dtype=torch.int32, device="cuda")

    kv_indptr, kv_indices, kv_len = build_shared_indices(
        req_pool_indices, req_to_token, seq_lens, "cuda", pool_len
    )

    # Check shapes
    assert kv_indptr.shape == (num_seqs + 1,)
    assert kv_len.shape == (num_seqs,)

    # Check indptr starts at 0
    assert kv_indptr[0].item() == 0

    # Check each request's prefix
    for r in range(num_seqs):
        plen = prefix_lens[r]
        assert kv_len[r].item() == plen

        start = kv_indptr[r].item()
        end = kv_indptr[r + 1].item()
        assert end - start == plen

        # Verify the actual page indices match what we put in req_to_token
        actual = kv_indices[start:end].cpu().tolist()
        expected = req_to_token[r, :plen].cpu().tolist()
        assert actual == expected, f"Mismatch for request {r}"


def test_unique_indices_structure(scenario):
    """Verify unique suffix indices have correct structure."""
    num_seqs, topk, num_steps, prefix_lens = scenario
    pool_len = max(prefix_lens) + topk * num_steps + 16

    req_pool_indices, req_to_token = _make_req_to_token_pool(
        num_seqs, pool_len, prefix_lens, topk, num_steps
    )
    seq_lens = torch.tensor(prefix_lens, dtype=torch.int32, device="cuda")

    for step in range(1, num_steps):
        kv_indptr, kv_indices, kv_len = build_unique_indices(
            req_pool_indices, req_to_token, seq_lens,
            topk, step, num_steps, 1, "cuda", pool_len
        )

        total_branches = num_seqs * topk
        assert kv_indptr.shape == (total_branches + 1,)
        assert kv_len.shape == (total_branches,)
        assert kv_indptr[0].item() == 0

        for r in range(num_seqs):
            for k in range(topk):
                branch = r * topk + k
                assert kv_len[branch].item() == step

                start = kv_indptr[branch].item()
                end = kv_indptr[branch + 1].item()
                assert end - start == step

                # Verify page indices: draft tokens for branch (r, k), steps 0..step-1
                actual = kv_indices[start:end].cpu().tolist()
                expected = []
                for s in range(step):
                    idx = prefix_lens[r] + k * num_steps + s
                    expected.append(req_to_token[r, idx].item())
                assert actual == expected, (
                    f"Mismatch for request={r}, topk={k}, step={step}"
                )


def test_unique_indices_step_zero():
    """Step offset 0 should produce empty suffix indices."""
    num_seqs, topk, num_steps = 2, 4, 3
    prefix_lens = [32, 64]
    pool_len = 128

    req_pool_indices, req_to_token = _make_req_to_token_pool(
        num_seqs, pool_len, prefix_lens, topk, num_steps
    )
    seq_lens = torch.tensor(prefix_lens, dtype=torch.int32, device="cuda")

    kv_indptr, kv_indices, kv_len = build_unique_indices(
        req_pool_indices, req_to_token, seq_lens,
        topk, 0, num_steps, 1, "cuda", pool_len
    )

    total_branches = num_seqs * topk
    assert kv_indptr.shape == (total_branches + 1,)
    assert (kv_indptr == 0).all()
    assert (kv_len == 0).all()


def test_cascade_covers_same_pages_as_flat(scenario):
    """Verify cascade (shared + unique) covers the same pages as the flat layout.

    For each branch at each step, the union of shared prefix pages and unique
    suffix pages should equal the set of pages that SGLang's flat kernel would
    produce (prefix + draft tokens up to that step).
    """
    num_seqs, topk, num_steps, prefix_lens = scenario
    pool_len = max(prefix_lens) + topk * num_steps + 16

    req_pool_indices, req_to_token = _make_req_to_token_pool(
        num_seqs, pool_len, prefix_lens, topk, num_steps
    )
    seq_lens = torch.tensor(prefix_lens, dtype=torch.int32, device="cuda")

    # Build shared prefix (same across all steps)
    shared_indptr, shared_indices, _ = build_shared_indices(
        req_pool_indices, req_to_token, seq_lens, "cuda", pool_len
    )

    for step in range(1, num_steps):
        unique_indptr, unique_indices, _ = build_unique_indices(
            req_pool_indices, req_to_token, seq_lens,
            topk, step, num_steps, 1, "cuda", pool_len
        )

        for r in range(num_seqs):
            for k in range(topk):
                branch = r * topk + k

                # Expected: prefix pages + draft pages 0..step-1
                expected_pages = set(req_to_token[r, :prefix_lens[r]].cpu().tolist())
                for s in range(step):
                    idx = prefix_lens[r] + k * num_steps + s
                    expected_pages.add(req_to_token[r, idx].item())

                # Actual: cascade shared + unique
                cascade_pages = _get_cascade_kv_set(
                    shared_indptr, shared_indices, r,
                    unique_indptr, unique_indices, branch,
                )

                assert cascade_pages == expected_pages, (
                    f"Page mismatch at r={r}, k={k}, step={step}:\n"
                    f"  expected: {sorted(expected_pages)}\n"
                    f"  got:      {sorted(cascade_pages)}"
                )


def test_next_power_of_2():
    assert next_power_of_2(0) == 1
    assert next_power_of_2(1) == 1
    assert next_power_of_2(2) == 2
    assert next_power_of_2(3) == 4
    assert next_power_of_2(5) == 8
    assert next_power_of_2(128) == 128
    assert next_power_of_2(129) == 256


def test_qo_indptr_shapes():
    """Verify qo_indptr tensors have the right structure for cascade plan."""
    num_seqs = 3
    topk = 6

    # Shared level: groups topk Q tokens per request
    qo_indptr_shared = torch.arange(
        0, (num_seqs + 1) * topk, topk, dtype=torch.int32
    )
    assert qo_indptr_shared.shape == (num_seqs + 1,)
    assert qo_indptr_shared[0].item() == 0
    assert qo_indptr_shared[-1].item() == num_seqs * topk

    # Unique level: 1 Q token per branch
    total_branches = num_seqs * topk
    qo_indptr_unique = torch.arange(
        0, total_branches + 1, dtype=torch.int32
    )
    assert qo_indptr_unique.shape == (total_branches + 1,)
    assert qo_indptr_unique[0].item() == 0
    assert qo_indptr_unique[-1].item() == total_branches

    # Both must have the same total Q length
    assert qo_indptr_shared[-1].item() == qo_indptr_unique[-1].item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
