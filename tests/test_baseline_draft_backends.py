"""Attention outputs of the baseline draft backends match a dense reference.

Drives the real backend classes (FIMultiLevelCascade, Hydragen, DeFT) through
init_forward_metadata + forward_decode on a synthetic paged KV pool laid out
the way SGLang's EAGLE draft lays it out (branch k's draft token s sits at
req_to_token[rpi, seq_len + k * speculative_num_steps + s]), and compares every
draft step against fp32 attention over prefix + that branch's path.

The FI cascade backend is also checked under CUDA graphs: capture at one prefix
length, re-plan at another, replay, compare.

Needs one idle GPU and the flash-attn 2 build in .venv/fa2 (for Hydragen):
    CUDA_VISIBLE_DEVICES=<idle gpu> uv run pytest tests/test_baseline_draft_backends.py -v
"""

import types

import pytest
import torch

import sglang.srt.layers.attention.deft_backend as deft_mod
import sglang.srt.layers.attention.flashinfer_multilevel_cascade_backend as fi_mod
import sglang.srt.layers.attention.hydragen_backend as hy_mod

HQ, HKV, D = 16, 4, 64  # Llama-3.2-1B draft heads per rank at TP=2
DTYPE = torch.bfloat16
POOL_TOKENS = 1 << 18
POOL_LEN = 8192

for _m in (deft_mod, fi_mod, hy_mod):
    if hasattr(_m, "get_attention_tp_size"):
        _m.get_attention_tp_size = lambda: 1


class Pool:
    def __init__(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        self.k = torch.randn(POOL_TOKENS, HKV, D, device="cuda", dtype=DTYPE, generator=g)
        self.v = torch.randn(POOL_TOKENS, HKV, D, device="cuda", dtype=DTYPE, generator=g)

    def get_kv_buffer(self, layer_id):
        return self.k, self.v

    def set_kv_buffer(self, *a, **k):
        raise AssertionError("test passes k=v=None")


def make_model_runner(req_to_token):
    cfg = types.SimpleNamespace(
        num_attention_heads=HQ,
        head_dim=D,
        context_len=POOL_LEN,
        get_num_kv_heads=lambda tp: HKV,
    )
    return types.SimpleNamespace(
        page_size=1,
        req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token),
        device=torch.device("cuda"),
        model_config=cfg,
        dtype=DTYPE,
        kv_cache_dtype=DTYPE,
    )


def make_batch(prefix_lens, topk, num_steps, pool, seed):
    """Fake forward_batch + reference KV locations."""
    g = torch.Generator().manual_seed(seed)
    num_seqs = len(prefix_lens)
    req_to_token = torch.zeros(8, POOL_LEN, dtype=torch.int32)
    rpis = [3, 5][:num_seqs] if num_seqs <= 2 else list(range(num_seqs))
    need = sum(prefix_lens) + num_seqs * topk * num_steps
    locs = torch.randperm(POOL_TOKENS, generator=g)[:need].to(torch.int32)
    paths = []  # per branch: (prefix locs, draft locs)
    off = 0
    for p, L in enumerate(prefix_lens):
        pre = locs[off : off + L]
        off += L
        req_to_token[rpis[p], :L] = pre
        for k in range(topk):
            dr = locs[off : off + num_steps]
            off += num_steps
            req_to_token[rpis[p], L + k * num_steps : L + (k + 1) * num_steps] = dr
            paths.append((pre, dr))
    req_to_token = req_to_token.cuda()
    seq_lens_cpu = torch.tensor(prefix_lens, dtype=torch.int64)
    fb = types.SimpleNamespace(
        batch_size=num_seqs,
        req_pool_indices=torch.tensor(rpis, dtype=torch.int64, device="cuda"),
        req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token),
        seq_lens=seq_lens_cpu.cuda(),
        seq_lens_cpu=seq_lens_cpu,
        out_cache_loc=None,
        token_to_kv_pool=pool,
    )
    return fb, paths


def reference(q, pool, paths, step):
    outs = []
    for b, (pre, dr) in enumerate(paths):
        idx = torch.cat([pre, dr[: step + 1]]).long().cuda()
        k = pool.k[idx].float().repeat_interleave(HQ // HKV, dim=1)  # [n, HQ, D]
        v = pool.v[idx].float().repeat_interleave(HQ // HKV, dim=1)
        s = torch.einsum("hd,nhd->hn", q[b].float(), k) * D**-0.5
        outs.append(torch.einsum("hn,nhd->hd", s.softmax(-1), v))
    return torch.stack(outs).reshape(len(paths), HQ * D)


LAYER = types.SimpleNamespace(
    tp_q_head_num=HQ, head_dim=D, scaling=D**-0.5, logit_cap=0.0,
    layer_id=0, k_scale=None, v_scale=None,
)

BACKENDS = {
    "fi_cascade": fi_mod.FIMultiLevelCascadeMultiStepDraftBackend,
    "hydragen": hy_mod.HydragenMultiStepDraftBackend,
    "deft": deft_mod.DeFTMultiStepDraftBackend,
}

CONFIGS = [
    ([300], 10, 7),
    ([4097], 10, 7),
    ([1000, 517], 10, 7),
    ([129], 4, 3),
]


def check(out, ref, what):
    err = (out.float() - ref).abs().max().item()
    assert err < 3e-2, f"{what}: max abs err {err}"


@pytest.fixture(scope="module")
def pool():
    return Pool()


@pytest.mark.parametrize("name", list(BACKENDS))
@pytest.mark.parametrize("prefix_lens,topk,num_steps", CONFIGS)
def test_eager(name, prefix_lens, topk, num_steps, pool):
    fb, paths = make_batch(prefix_lens, topk, num_steps, pool, seed=sum(prefix_lens))
    be = BACKENDS[name](make_model_runner(fb.req_to_token_pool.req_to_token), topk, num_steps)
    g = torch.Generator(device="cuda").manual_seed(1)
    q = torch.randn(len(paths), HQ, D, device="cuda", dtype=DTYPE, generator=g)
    be.init_forward_metadata(fb)
    for i in range(num_steps - 1):
        out = be.attn_backends[i].forward_decode(q.view(len(paths), -1), None, None, LAYER, fb)
        check(out, reference(q, pool, paths, i), f"{name} step {i}")


def test_hydragen_incremental_prefix(pool):
    """Growing the same request's prefix appends to the cache and stays exact."""
    topk, num_steps = 10, 7
    fb, paths = make_batch([2000], topk, num_steps, pool, seed=11)
    be = hy_mod.HydragenMultiStepDraftBackend(
        make_model_runner(fb.req_to_token_pool.req_to_token), topk, num_steps
    )
    q = torch.randn(topk, HQ, D, device="cuda", dtype=DTYPE)
    # Same request (same slot, same token-0 location) with a growing prefix,
    # then a different request in the same slot (seed 12) that must not reuse
    # the cached prefix. Branch slots follow seq_len as they do in SGLang.
    for L, seed in ((1500, 11), (1700, 11), (1700, 11), (2000, 11), (2400, 12), (900, 11)):
        fb2, paths2 = make_batch([L], topk, num_steps, pool, seed=seed)
        be.init_forward_metadata(fb2)
        for i in range(num_steps - 1):
            out = be.attn_backends[i].forward_decode(q.view(topk, -1), None, None, LAYER, fb2)
            check(out, reference(q, pool, paths2, i), f"hydragen L={L} step {i}")


def test_fi_cascade_cuda_graph(pool):
    topk, num_steps = 10, 7
    fb1, _ = make_batch([700], topk, num_steps, pool, seed=21)
    fb2, paths2 = make_batch([1900], topk, num_steps, pool, seed=22)
    # Both batches use the same req_to_token tensor object in SGLang; copy fb2's in.
    req_to_token = fb1.req_to_token_pool.req_to_token
    be = fi_mod.FIMultiLevelCascadeMultiStepDraftBackend(
        make_model_runner(req_to_token), topk, num_steps
    )
    be.init_cuda_graph_state(max_bs=4, max_num_tokens=4 * topk)
    q = torch.randn(topk, HQ, D, device="cuda", dtype=DTYPE)
    out_bufs = [torch.empty(topk, HQ * D, device="cuda", dtype=DTYPE) for _ in range(num_steps - 1)]

    be.init_forward_metadata_capture_cuda_graph(fb1)
    graph = torch.cuda.CUDAGraph()
    for i in range(num_steps - 1):  # warm up outside capture
        be.attn_backends[i].forward_decode(q.view(topk, -1), None, None, LAYER, fb1)
    with torch.cuda.graph(graph):
        for i in range(num_steps - 1):
            out_bufs[i].copy_(
                be.attn_backends[i].forward_decode(q.view(topk, -1), None, None, LAYER, fb1)
            )

    req_to_token.copy_(fb2.req_to_token_pool.req_to_token)
    fb2.req_to_token_pool.req_to_token = req_to_token
    be.init_forward_metadata_replay_cuda_graph(fb2, bs=1)
    graph.replay()
    torch.cuda.synchronize()
    for i in range(num_steps - 1):
        check(out_bufs[i], reference(q, pool, paths2, i), f"fi_cascade CG replay step {i}")
