"""Packed-suffix cascade attention vs stock FlashInfer MultiLevelCascade.

CascadeBatchAttentionWrapper(pack_suffix=True) runs 16 // GQA single-token
branches per 16-row suffix tile with a branch-diagonal mask. Checked through the
E2E path (plan_for_draft once at max depth + update_draft_step per step) on
batches with unequal prefix lengths, so requests get different numbers of
prefix KV chunks (the case a group straddling two requests would break), and
top-k values that do not divide into whole groups.

    CUDA_VISIBLE_DEVICES=<idle gpu> uv run pytest tests/test_cascade_pack_suffix.py -v
"""

import pytest
import torch
from flashinfer.attention import CascadeBatchAttentionWrapper
from flashinfer.cascade import MultiLevelCascadeAttentionWrapper

DTYPE = torch.bfloat16

CASES = [
    # (prefix lens per request, topk, max depth D, hq, hkv)
    ([300], 10, 6, 16, 4),
    ([1000, 517], 10, 6, 16, 4),
    ([20000, 3000, 40000, 129], 10, 6, 16, 4),  # different prefix chunk counts
    ([5000, 70], 3, 4, 16, 4),  # topk < group size
    ([2048, 33000], 5, 16, 16, 4),  # G * D = 64: one full suffix KV tile
    ([4097, 900, 12000], 10, 6, 32, 4),  # GQA 8: 2 branches per tile
]


def ref_wrapper(ws, prefix_lens, topk, s, shared, unique, hq, hkv, d):
    bs = len(prefix_lens)
    nb = bs * topk
    i32 = dict(dtype=torch.int32)
    w = MultiLevelCascadeAttentionWrapper(2, ws, "NHD")
    w.plan(
        [torch.arange(0, (bs + 1) * topk, topk, **i32), torch.arange(nb + 1, **i32)],
        [torch.tensor([0] + prefix_lens, **i32).cumsum(0, dtype=torch.int32),
         torch.arange(nb + 1, **i32) * s],
        [shared, unique[:, :s].contiguous().view(-1)],
        [torch.ones(bs, **i32), torch.ones(nb, **i32)],
        num_qo_heads=hq, num_kv_heads=hkv, head_dim=d, page_size=1, causal=False,
        q_data_type=DTYPE, kv_data_type=DTYPE,
    )
    return w


@pytest.mark.parametrize("pack", [False, True])
@pytest.mark.parametrize("prefix_lens,topk,D,hq,hkv", CASES)
def test_packed_suffix_matches_reference(prefix_lens, topk, D, hq, hkv, pack):
    torch.manual_seed(0)
    d = 64
    bs, nb = len(prefix_lens), len(prefix_lens) * topk
    pool = sum(prefix_lens) + nb * D + 64
    perm = torch.randperm(pool).to(torch.int32).cuda()
    shared = perm[: sum(prefix_lens)]
    unique = perm[sum(prefix_lens): sum(prefix_lens) + nb * D].view(nb, D)
    k = torch.randn(pool, hkv, d, device="cuda", dtype=DTYPE)
    v = torch.randn(pool, hkv, d, device="cuda", dtype=DTYPE)
    q = torch.randn(nb, hq, d, device="cuda", dtype=DTYPE)
    i32 = dict(dtype=torch.int32)

    w = CascadeBatchAttentionWrapper(2, "NHD", "cuda")
    w.plan_for_draft(
        max_draft_depth=D, first_call=True,
        qo_indptr_host_arr=[torch.arange(0, (bs + 1) * topk, topk, **i32),
                            torch.arange(nb + 1, **i32)],
        kv_indptr_host_arr=[torch.tensor([0] + prefix_lens, **i32).cumsum(0, dtype=torch.int32),
                            torch.arange(nb + 1, **i32) * D],
        kv_indices_arr=[shared, unique.contiguous().view(-1)],
        kv_len_host_arr=[torch.tensor(prefix_lens, **i32), torch.full((nb,), D, **i32)],
        num_qo_heads=hq, num_kv_heads=hkv, head_dim_qk=d, head_dim_vo=d, page_size=1,
        causal=False, q_data_type=DTYPE, kv_data_type=DTYPE, pack_suffix=pack,
    )
    if pack:
        assert w._branch_group == 16 // (hq // hkv)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for s in range(1, D + 1):
        w.update_draft_step(step_kv_len=s + 1, step_kv_end=s)
        out, _ = w.run(q, (k, v))
        ref = ref_wrapper(ws, prefix_lens, topk, s, shared, unique, hq, hkv, d).run(q, (k, v))
        err = (out.float() - ref.float()).abs().max().item()
        assert err < 1e-2, f"step {s}: max err {err}"
