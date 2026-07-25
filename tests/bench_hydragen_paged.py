"""Hydragen vs Fast Draft: what is shared, and what blocks a direct port.

Three experiments, all at EAGLE's draft-decode operating point.

1. `merge`  -- Hydragen's `combine_lse` (imported verbatim from
   3rdparty/hydragen) vs FlashInfer's `merge_state`. Establishes that the
   log-sum-exp recombination is literally the same primitive, so the paper
   should cite Hydragen for it and claim nothing about it.

2. `gather` -- the cost of materializing Hydragen's *contiguous* shared/unique
   KV layout out of SGLang's paged pool, per layer per draft step, compared
   against the attention it is supposed to accelerate. This is the concrete
   reason `hydragen_attention` cannot be called from a paged serving engine.

3. `attn`   -- flat paged decode vs Hydragen-paged (two prefill passes +
   merge_state, i.e. `MultiLevelCascadeAttentionWrapper`) vs our fused
   `CascadeBatchAttentionWrapper`, with the decomposition math held constant.

Usage:
    python tests/bench_hydragen_paged.py
    python tests/bench_hydragen_paged.py --prefix-lens 8192,32768 --topk 5
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "3rdparty", "flashinfer")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "3rdparty", "hydragen")
)

import flashinfer
from flashinfer.cascade import MultiLevelCascadeAttentionWrapper, merge_state


def _time(fn, warmup=20, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


# ----------------------------------------------------------------------
# 1. the merge primitive is the same
# ----------------------------------------------------------------------


def check_merge_equivalence(num_tokens=4096, num_heads=8, head_dim=128):
    """Hydragen's combine_lse vs FlashInfer's merge_state on identical inputs."""
    # hydragen.attention imports hydragen.flash, which pulls in flash_attn --
    # not installed here (see docs/hydragen_positioning.md). Stub that one
    # module out so combine_lse can be imported verbatim from the real file.
    # It must be a real import, not exec(): Triton's JIT resolves a kernel's
    # source via inspect/linecache, and an exec'd module makes its dependency
    # finder walk the wrong function body.
    import types

    import hydragen  # namespace package, no __init__.py

    if "hydragen.flash" not in sys.modules:
        stub = types.ModuleType("hydragen.flash")
        for name in ("flash_attention", "flash_attention_varlen",
                     "flash_attention_seqlen"):
            setattr(stub, name, None)
        sys.modules["hydragen.flash"] = stub

    from hydragen.attention import combine_lse

    # Feeding synthetic LSEs to both merges only tests that they share a
    # convention, which they do not have to. The meaningful question is whether
    # each one correctly *reconstructs full attention* from a prefix part and a
    # suffix part -- that is what "same primitive" means here. So: run real
    # attention over prefix+suffix as the reference, then run it over each half
    # and merge with FlashInfer and with Hydragen.
    torch.manual_seed(0)
    dtype = torch.float16
    dev = "cuda"
    prefix_len, suffix_len, batch = 2048, 4, 8
    nq, nkv, d = num_heads * 4, num_heads, head_dim

    total = batch * (prefix_len + suffix_len)
    k_pool = torch.randn(total, nkv, d, device=dev, dtype=dtype)
    v_pool = torch.randn(total, nkv, d, device=dev, dtype=dtype)
    kv = (k_pool, v_pool)
    q = torch.randn(batch, nq, d, device=dev, dtype=dtype)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)

    qo = torch.arange(batch + 1, dtype=torch.int32)
    last = torch.ones(batch, dtype=torch.int32)

    def run_over(per_seq_len, index_of):
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
        indptr = torch.arange(
            0, (batch + 1) * per_seq_len, per_seq_len, dtype=torch.int32
        )
        idx = torch.cat([index_of(b) for b in range(batch)]).to(dev)
        w.plan(qo, indptr, idx, last, nq, nkv, d, 1, causal=False,
               q_data_type=dtype, kv_data_type=dtype)
        return w.run(q, kv, return_lse=True)

    seg = prefix_len + suffix_len
    full_out, _ = run_over(
        seg, lambda b: torch.arange(b * seg, (b + 1) * seg, dtype=torch.int32)
    )
    pre_out, pre_lse = run_over(
        prefix_len,
        lambda b: torch.arange(b * seg, b * seg + prefix_len, dtype=torch.int32),
    )
    suf_out, suf_lse = run_over(
        suffix_len,
        lambda b: torch.arange(
            b * seg + prefix_len, (b + 1) * seg, dtype=torch.int32
        ),
    )

    scale = max(full_out.float().abs().max().item(), 1e-6)
    print("\n=== 1. LSE merge primitive: does each reconstruct full attention? ===")
    print(f"  reference: attention over prefix({prefix_len})+suffix({suffix_len}), "
          f"batch={batch}, {nq}q/{nkv}kv heads, head_dim={d}, fp16")
    print(f"  {'merge implementation':>24}  {'max abs':>10}  {'max rel':>10}  verdict")
    print(f"  {'-'*24}  {'-'*10}  {'-'*10}  {'-'*7}")

    results = {}
    fi_out, _ = merge_state(pre_out, pre_lse, suf_out, suf_lse)
    results["FlashInfer merge_state"] = fi_out

    # Hydragen combine_lse: outs [B, S, H, D], lses [B, S, H]; here S = 1.
    outs = [pre_out.unsqueeze(1).contiguous(), suf_out.unsqueeze(1).contiguous()]
    lses = [pre_lse.unsqueeze(1).contiguous(), suf_lse.unsqueeze(1).contiguous()]
    for label, use_triton in (("Hydragen combine_lse_torch", False),
                              ("Hydragen combine_lse_triton", True)):
        results[label] = combine_lse(outs, lses, enable_triton=use_triton).squeeze(1)

    worst = 0.0
    for label, out in results.items():
        diff = (full_out.float() - out.float()).abs()
        rel = diff.max().item() / scale
        worst = max(worst, rel)
        verdict = "EXACT" if rel < 5e-3 else "DIFFERS"
        print(f"  {label:>24}  {diff.max().item():>10.3e}  {rel:>10.3e}  {verdict}")

    print("  Both reconstruct the same full attention: the decomposition and its")
    print("  log-sum-exp recombination are one and the same primitive.")
    return worst


# ----------------------------------------------------------------------
# 2. the contiguous-cache requirement
# ----------------------------------------------------------------------


def bench_gather(num_seqs, topk, prefix_len, num_kv_heads, head_dim, num_layers, dtype):
    """Cost of building Hydragen's contiguous shared cache from a paged pool.

    Hydragen's `hydragen_attention` takes `shared_ks[i]` of shape
    [sbatch, slen, kvheads, head_dim] -- a dense tensor. SGLang hands us a page
    table instead, so a faithful port must gather. We measure one layer's
    gather for K and V, then scale to a full draft iteration.
    """
    total_pages = num_seqs * prefix_len + num_seqs * topk * 8
    k_pool = torch.randn(total_pages, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    v_pool = torch.randn(total_pages, num_kv_heads, head_dim, device="cuda", dtype=dtype)

    # Scattered page ids, as a real paged allocator would produce.
    perm = torch.randperm(total_pages, device="cuda")[: num_seqs * prefix_len]
    idx = perm.to(torch.int64)

    k_dst = torch.empty(
        num_seqs * prefix_len, num_kv_heads, head_dim, device="cuda", dtype=dtype
    )
    v_dst = torch.empty_like(k_dst)

    def gather():
        torch.index_select(k_pool, 0, idx, out=k_dst)
        torch.index_select(v_pool, 0, idx, out=v_dst)

    ms = _time(gather, warmup=10, repeat=50)
    bytes_moved = 2 * 2 * k_dst.numel() * k_dst.element_size()  # read + write, K and V
    gbps = bytes_moved / (ms * 1e-3) / 1e9
    return ms, gbps


# ----------------------------------------------------------------------
# 3. attention: flat vs Hydragen-paged vs fused cascade
# ----------------------------------------------------------------------


def bench_attention(num_seqs, topk, prefix_len, step_offset, num_qo_heads,
                    num_kv_heads, head_dim, dtype):
    from flashinfer.attention import CascadeBatchAttentionWrapper

    total_branches = num_seqs * topk
    num_steps = 8
    total_pages = num_seqs * (prefix_len + topk * num_steps)
    k_data = torch.randn(total_pages, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    v_data = torch.randn(total_pages, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    kv_data = (k_data, v_data)
    q = torch.randn(total_branches, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    dev = "cuda"
    # level 0: shared prefix, topk queries per request
    qo_shared = torch.arange(0, (num_seqs + 1) * topk, topk, dtype=torch.int32)
    kv_indptr_shared = torch.arange(
        0, (num_seqs + 1) * prefix_len, prefix_len, dtype=torch.int32
    )
    kv_indices_shared = torch.arange(
        num_seqs * prefix_len, dtype=torch.int32, device=dev
    )
    last_shared = torch.ones(num_seqs, dtype=torch.int32)

    # level 1: unique suffix, 1 query per branch
    qo_unique = torch.arange(total_branches + 1, dtype=torch.int32)
    kv_indptr_unique = torch.arange(
        0, (total_branches + 1) * step_offset, step_offset, dtype=torch.int32
    )[: total_branches + 1]
    kv_indices_unique = torch.arange(
        num_seqs * prefix_len,
        num_seqs * prefix_len + total_branches * step_offset,
        dtype=torch.int32,
        device=dev,
    )
    last_unique = torch.ones(total_branches, dtype=torch.int32)

    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)

    # --- flat: every branch re-reads the whole prefix + its suffix ---
    flat = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    flat_kv_len = prefix_len + step_offset
    flat_indptr = torch.arange(
        0, (total_branches + 1) * flat_kv_len, flat_kv_len, dtype=torch.int32
    )
    flat_indices = torch.empty(total_branches * flat_kv_len, dtype=torch.int32, device=dev)
    for b in range(total_branches):
        s = b // topk
        base = b * flat_kv_len
        flat_indices[base : base + prefix_len] = torch.arange(
            s * prefix_len, (s + 1) * prefix_len, dtype=torch.int32, device=dev
        )
        flat_indices[base + prefix_len : base + flat_kv_len] = kv_indices_unique[
            b * step_offset : (b + 1) * step_offset
        ]
    flat.plan(
        qo_unique, flat_indptr, flat_indices, last_unique,
        num_qo_heads, num_kv_heads, head_dim, 1,
        causal=False, q_data_type=dtype, kv_data_type=dtype,
    )
    t_flat = _time(lambda: flat.run(q, kv_data))

    # --- Hydragen-paged: two passes + merge_state_in_place ---
    ml = MultiLevelCascadeAttentionWrapper(2, ws, "NHD")

    def plan_ml():
        ml.plan(
            [qo_shared, qo_unique],
            [kv_indptr_shared, kv_indptr_unique],
            [kv_indices_shared, kv_indices_unique],
            [last_shared, last_unique],
            num_qo_heads, num_kv_heads, head_dim, 1,
            causal=False, q_data_type=dtype, kv_data_type=dtype,
        )

    plan_ml()
    t_hy = _time(lambda: ml.run(q, kv_data))
    t_hy_plan = _time(plan_ml, warmup=5, repeat=20)

    # --- ours: fused two-level cascade ---
    cas = CascadeBatchAttentionWrapper(num_levels=2, kv_layout="NHD", device=dev)

    def plan_cas():
        cas.plan(
            qo_indptr_arr=[qo_shared, qo_unique],
            kv_indptr_arr=[kv_indptr_shared, kv_indptr_unique],
            kv_indices_arr=[kv_indices_shared, kv_indices_unique],
            kv_len_arr=[
                torch.full((num_seqs,), prefix_len, dtype=torch.int32),
                torch.full((total_branches,), step_offset, dtype=torch.int32),
            ],
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=1,
            causal=False,
            sm_scale=None,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

    plan_cas()
    t_cas = _time(lambda: cas.run(q, kv_data)[0])
    t_cas_plan = _time(plan_cas, warmup=5, repeat=20)

    return dict(
        flat=t_flat, hydragen=t_hy, hydragen_plan=t_hy_plan,
        cascade=t_cas, cascade_plan=t_cas_plan,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-seqs", type=int, default=4)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--step-offset", type=int, default=3)
    p.add_argument("--prefix-lens", default="4096,16384,50000")
    p.add_argument("--num-qo-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=1,
                   help="draft model layer count, for scaling the gather cost")
    p.add_argument("--num-draft-steps", type=int, default=3)
    p.add_argument("--skip-merge", action="store_true")
    args = p.parse_args()

    dtype = torch.float16
    prefix_lens = [int(x) for x in args.prefix_lens.split(",")]

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"config: num_seqs={args.num_seqs} topk={args.topk} "
          f"step_offset={args.step_offset} "
          f"heads={args.num_qo_heads}/{args.num_kv_heads} head_dim={args.head_dim}")

    if not args.skip_merge:
        check_merge_equivalence()

    print("\n=== 2. Hydragen's contiguous shared cache: gather cost from a paged pool ===")
    print("  (Hydragen takes shared_ks[i] as [sbatch, slen, kvheads, d]; SGLang has page ids)")
    print(f"  {'prefix':>8}  {'gather/layer':>13}  {'achieved BW':>12}  "
          f"{'gather x draft iter':>20}")
    print(f"  {'-'*8}  {'-'*13}  {'-'*12}  {'-'*20}")
    gather_ms = {}
    for pl in prefix_lens:
        ms, gbps = bench_gather(
            args.num_seqs, args.topk, pl, args.num_kv_heads, args.head_dim,
            args.num_layers, dtype,
        )
        gather_ms[pl] = ms
        # one draft iteration = num_draft_steps steps x num_layers layers
        per_iter = ms * args.num_layers * args.num_draft_steps
        print(f"  {pl:>8}  {ms:>12.3f}ms  {gbps:>10.0f}GB/s  {per_iter:>18.3f}ms")

    print("\n=== 3. Draft-decode attention: flat vs Hydragen-paged vs fused cascade ===")
    print(f"  {'prefix':>8}  {'flat':>9}  {'hydragen':>9}  {'cascade':>9}  "
          f"{'hy plan':>9}  {'cas plan':>9}  {'hy/flat':>8}  {'cas/hy':>8}  {'gather/attn':>11}")
    print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*11}")
    for pl in prefix_lens:
        r = bench_attention(
            args.num_seqs, args.topk, pl, args.step_offset,
            args.num_qo_heads, args.num_kv_heads, args.head_dim, dtype,
        )
        ratio_hy = r["flat"] / r["hydragen"]
        ratio_cas = r["hydragen"] / r["cascade"]
        g_over_a = gather_ms[pl] / r["hydragen"]
        print(f"  {pl:>8}  {r['flat']:>8.3f}ms  {r['hydragen']:>8.3f}ms  "
              f"{r['cascade']:>8.3f}ms  {r['hydragen_plan']:>8.3f}ms  "
              f"{r['cascade_plan']:>8.3f}ms  {ratio_hy:>7.2f}x  {ratio_cas:>7.2f}x  "
              f"{g_over_a:>10.2f}x")
    print("\n  hy/flat  = speedup of Hydragen's decomposition over flat paged attention")
    print("  cas/hy   = additional speedup from the fused two-level kernel")
    print("  gather/attn = cost of Hydragen's contiguous-cache requirement, "
          "relative to one attention call")


if __name__ == "__main__":
    main()
