"""The DeFT draft backend's vectorized planner matches DeFT's own DeFT_preparation.

DeFT_preparation (FastTree-Artifact/kernel_bench/DeFT.py) is the reference: it
is run on the same 2-level draft tree the backend sees (prefix root, topk
branch leaves of length step+1), its virtual KV entries are mapped to the same
synthetic pool locations, and all 13 metadata tensors are compared exactly.

Runs on CPU: DeFT_preparation only uses CUDA as an allocation context.

    uv run pytest tests/test_deft_plan.py -v
"""

import contextlib
import os
import sys
import types

import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "3rdparty", "FastTree-Artifact", "kernel_bench"))

import DeFT  # noqa: E402
from kv_tree_simple import KVTreeNode  # noqa: E402

from sglang.srt.layers.attention.deft_backend import (  # noqa: E402
    MASK_LEN,
    SUBTREE_LEN,
    DeFTMultiStepDraftBackend,
)

NUM_QO_HEADS = 4
HEAD_DIM = 8


class _TorchOnCPU(types.ModuleType):
    """torch, except ``torch.device(...)`` used as a context manager is a no-op."""

    def __init__(self):
        super().__init__("torch")

    def __getattr__(self, name):
        if name == "device":
            return lambda *a, **k: contextlib.nullcontext()
        return getattr(torch, name)


def deft_reference(prefix_len, topk, depth):
    """Run DeFT_preparation on root(prefix_len) -> topk leaves(depth)."""
    nodes = []
    root = KVTreeNode()
    root.parent, root.id, root.seqlen, root.num_children = -1, 0, prefix_len, topk
    root.requests = []
    nodes.append(root)
    for k in range(topk):
        leaf = KVTreeNode()
        leaf.parent, leaf.id, leaf.seqlen, leaf.num_children = 0, 1 + k, depth, 0
        leaf.requests = []
        nodes.append(leaf)
    for r in range(topk):
        for n in (r + 1, 0):
            nodes[n].requests.append(r)

    max_seqlen = prefix_len + depth
    k_cache = torch.empty((topk, max_seqlen, 1, 1), device="meta")
    DeFT.cur_length = 0
    saved = DeFT.torch
    DeFT.torch = _TorchOnCPU()
    try:
        aux = DeFT.DeFT_preparation(nodes, k_cache, SUBTREE_LEN, MASK_LEN, NUM_QO_HEADS, HEAD_DIM)
    finally:
        DeFT.torch = saved
    return aux, max_seqlen


def make_backend(topk, num_draft_steps):
    be = DeFTMultiStepDraftBackend.__new__(DeFTMultiStepDraftBackend)
    be.topk = topk
    be.num_draft_steps = num_draft_steps
    be.device = torch.device("cpu")
    be.num_qo_heads = NUM_QO_HEADS
    be.head_dim = HEAD_DIM
    return be


CASES = [
    # (prefix_len, topk, num_draft_steps)
    (1, 1, 1),
    (5, 3, 4),
    (127, 4, 6),
    (128, 4, 6),
    (129, 10, 6),
    (250, 10, 6),
    (256, 2, 3),
    (1000, 10, 6),
    (5003, 8, 7),
    (130, 40, 3),  # Qset > 64: multiple mask blocks per subtree
]


@pytest.mark.parametrize("prefix_len,topk,steps", CASES)
def test_matches_deft_preparation(prefix_len, topk, steps):
    gen = torch.Generator().manual_seed(prefix_len * 131 + topk)
    shared = torch.randperm(10 * (prefix_len + 64), generator=gen)[:prefix_len]
    unique = 10 * (prefix_len + 64) + torch.randperm(topk * steps * 3, generator=gen)[: topk * steps]
    src = torch.cat([shared, unique]).long()  # pool locations, backend layout

    be = make_backend(topk, steps)
    step_aux = be.build_step_aux([prefix_len], src)

    for i in range(steps):
        depth = i + 1
        ref, max_seqlen = deft_reference(prefix_len, topk, depth)

        # Map DeFT's virtual entries (req * max_seqlen + path position) to pool locations.
        table = torch.full((topk * max_seqlen,), -7, dtype=torch.long)
        table[:prefix_len] = shared
        for r in range(topk):
            base = r * max_seqlen + prefix_len
            table[base : base + depth] = unique[r * steps : r * steps + depth]
        ref = list(ref)
        valid = ref[0] >= 0
        mapped = ref[0].clone().long()
        mapped[valid] = table[ref[0][valid].long()]
        assert (mapped[valid] >= 0).all()
        ref[0] = mapped.to(torch.int32)

        got = step_aux[i]
        names = [
            "vnode_to_kv_entries", "vnode_to_kv_offs", "vnode_to_kv_lens",
            "vnode_to_q_entries", "vnode_to_q_offs", "vnode_to_q_lens",
            "causalmask", "causalmask_offset", "req_to_vnode_entries",
            "req_to_vnode_offs", "req_to_vnode_lens",
        ]
        for name, a, b in zip(names, got[:11], ref[:11]):
            assert a.dtype == b.dtype, (name, a.dtype, b.dtype)
            assert torch.equal(a.cpu(), b.cpu()), f"step {i}: {name} differs"
        # mid_o / mid_lse: same shapes and dtypes
        for a, b in zip(got[11:], ref[11:]):
            assert a.shape == b.shape and a.dtype == b.dtype
