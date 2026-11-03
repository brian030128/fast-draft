"""How much does suffix query padding cost? Prototype of a prefill + decode split.

Draft decode with topk branches on one request: every branch has 1 query token,
i.e. GQA = hq / hkv packed query rows. The fused cascade kernel (Ours) and stock
FlashInfer MultiLevelCascade both run the unique-suffix level with a 16-row
tensor-core query tile, so with GQA 4 three quarters of each suffix tile is padding.

Variants, timed per draft step (all layers, CUDA graph replay):
  ours        fused cascade kernel, plan_for_draft + update_draft_step (E2E path)
  ours_pk     same, with pack_suffix=True (16 // GQA branches per suffix tile)
  fi          stock MultiLevelCascade: prefill(prefix) + prefill(suffix) + merge
  paged       no prefix sharing (SGLang-style): CUDA-core decode, every branch
              reads its full prefix + suffix
  split       proposed: prefill(prefix, topk*GQA packed rows) + CUDA-core
              decode(suffix, GQA rows per CTA, no padding) + merge
Components: prefix prefill alone, suffix prefill vs suffix decode, merge.
"prefix alone" is a lower bound for any design that keeps a separate prefix pass.

    CUDA_VISIBLE_DEVICES=<idle gpu> [FLASHINFER_CASCADE_TUNE=64,2,4096] \
        uv run python tests/profile_query_padding.py
"""

import argparse

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper
from flashinfer.attention import CascadeBatchAttentionWrapper
from flashinfer.cascade import MultiLevelCascadeAttentionWrapper, merge_state_in_place

from profile_draft_kernels import DTYPE, graph_time_us


def make_batch(bs, prefix_len, topk, depth, hkv, d, layers, seed=0):
    """bs requests with equal prefix length; KV pages scattered, one pool per layer."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_pre, n_suf = bs * prefix_len, bs * topk * depth
    pool = n_pre + n_suf + 4096
    perm = torch.randperm(pool, generator=g).to(torch.int32)
    shared = perm[:n_pre].cuda()
    unique = perm[n_pre:n_pre + n_suf].view(bs * topk, depth).cuda()
    kvs = [(torch.randn(pool, hkv, d, device="cuda", dtype=DTYPE),
            torch.randn(pool, hkv, d, device="cuda", dtype=DTYPE)) for _ in range(layers)]
    return shared, unique, kvs


def host_indptrs(bs, prefix_len, topk, s):
    """Level-0 (per request) / level-1 (per branch) host arrays for suffix length s."""
    i32 = dict(dtype=torch.int32)
    nb = bs * topk
    return dict(
        qo0=torch.arange(0, (bs + 1) * topk, topk, **i32),
        qo1=torch.arange(nb + 1, **i32),
        kv0=torch.arange(0, (bs + 1) * prefix_len, prefix_len, **i32),
        kv1=torch.arange(0, (nb + 1) * s, s, **i32)[: nb + 1] if s else torch.zeros(nb + 1, **i32),
        len0=torch.full((bs,), prefix_len, **i32),
        len1=torch.full((nb,), s, **i32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-lens", default="16384,32768,52203,65536,97404")
    ap.add_argument("--batch-sizes", default="1")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--steps", type=int, default=7, help="speculative_num_steps")
    ap.add_argument("--hq", type=int, default=16)
    ap.add_argument("--hkv", type=int, default=4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    topk, D, hq, hkv, d, nl = args.topk, args.steps - 1, args.hq, args.hkv, args.d, args.layers
    ws = lambda: torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    fi_ws, pre_ws, dec_ws, sufp_ws, pg_ws = ws(), ws(), ws(), ws(), ws()
    ones = lambda n: torch.ones(n, dtype=torch.int32)
    print(f"GPU: {torch.cuda.get_device_name()}  hq={hq} hkv={hkv} (GQA {hq // hkv}) d={d} "
          f"topk={topk} draft_steps={D} layers={nl}")
    print("times: us per draft step, all layers; columns are summed over the 6 steps -> ms/iter\n")

    rows = []
    cases = [(bs, int(x)) for bs in map(int, args.batch_sizes.split(","))
             for x in args.prefix_lens.split(",")]
    for bs, L in cases:
        shared, unique, kvs = make_batch(bs, L, topk, D, hkv, d, nl)
        nq = bs * topk
        q = torch.randn(nq, hq, d, device="cuda", dtype=DTYPE)

        h = host_indptrs(bs, L, topk, D)
        ours, ours_pk = (CascadeBatchAttentionWrapper(2, "NHD", "cuda") for _ in range(2))
        for w, pk in ((ours, False), (ours_pk, True)):
            w.plan_for_draft(
                max_draft_depth=D, first_call=True,
                qo_indptr_host_arr=[h["qo0"], h["qo1"]], kv_indptr_host_arr=[h["kv0"], h["kv1"]],
                kv_indices_arr=[shared, unique.contiguous().view(-1)],
                kv_len_host_arr=[h["len0"], h["len1"]],
                num_qo_heads=hq, num_kv_heads=hkv, head_dim_qk=d, head_dim_vo=d, page_size=1,
                causal=False, sm_scale=None, q_data_type=DTYPE, kv_data_type=DTYPE,
                pack_suffix=pk)

        # Prefix: all topk query tokens of the request against the shared prefix
        # (planned once per iteration; it does not change across draft steps).
        pre = BatchPrefillWithPagedKVCacheWrapper(pre_ws, "NHD")
        pre.plan(h["qo0"], h["kv0"], shared, ones(bs), hq, hkv, d, 1, causal=False,
                 q_data_type=DTYPE, kv_data_type=DTYPE)

        t = {k: 0.0 for k in ("ours", "ours_pk", "paged", "fi", "split", "prefix", "suf_prefill", "suf_decode", "merge")}
        max_err = 0.0
        for s in range(1, D + 1):
            hs = host_indptrs(bs, L, topk, s)
            suf_idx = unique[:, :s].contiguous().view(-1)
            fi = MultiLevelCascadeAttentionWrapper(2, fi_ws, "NHD")
            fi.plan([hs["qo0"], hs["qo1"]], [hs["kv0"], hs["kv1"]], [shared, suf_idx],
                    [ones(bs), ones(nq)], num_qo_heads=hq, num_kv_heads=hkv, head_dim=d,
                    page_size=1, causal=False, q_data_type=DTYPE, kv_data_type=DTYPE)
            sufp = BatchPrefillWithPagedKVCacheWrapper(sufp_ws, "NHD")
            sufp.plan(hs["qo1"], hs["kv1"], suf_idx, ones(nq), hq, hkv, d, 1, causal=False,
                      q_data_type=DTYPE, kv_data_type=DTYPE)
            dec = BatchDecodeWithPagedKVCacheWrapper(dec_ws, "NHD", use_tensor_cores=False)
            dec.plan(hs["kv1"], suf_idx, ones(nq), hq, hkv, d, 1,
                     q_data_type=DTYPE, kv_data_type=DTYPE)
            # paged: branch b of request r reads prefix of r, then its own suffix
            pre_r = shared.view(bs, L)
            pg_idx = torch.cat([pre_r.repeat_interleave(topk, 0), unique[:, :s]], 1).reshape(-1)
            pg = BatchDecodeWithPagedKVCacheWrapper(pg_ws, "NHD", use_tensor_cores=False)
            pg.plan(torch.arange(nq + 1, dtype=torch.int32) * (L + s), pg_idx, ones(nq), hq, hkv, d, 1,
                    q_data_type=DTYPE, kv_data_type=DTYPE)
            ours.update_draft_step(step_kv_len=s + 1, step_kv_end=s)
            ours_pk.update_draft_step(step_kv_len=s + 1, step_kv_end=s)

            # Preallocated outputs so the split path is graph-capturable.
            o_pre = torch.empty_like(q); lse_pre = torch.empty(nq, hq, device="cuda")
            o_dec = torch.empty_like(q); lse_dec = torch.empty(nq, hq, device="cuda")

            def split(kv):
                pre.run(q, kv, out=o_pre, lse=lse_pre, return_lse=True)
                dec.run(q, kv, out=o_dec, lse=lse_dec, return_lse=True)
                merge_state_in_place(o_pre, lse_pre, o_dec, lse_dec)
                return o_pre

            # correctness vs stock FI
            ref = fi.run(q, kvs[0]).float()
            o_ours, _ = ours.run(q, kvs[0])
            o_pk, lse_pk = ours_pk.run(q, kvs[0])
            max_err = max(max_err, (split(kvs[0]).float() - ref).abs().max().item(),
                          (pg.run(q, kvs[0]).float() - ref).abs().max().item(),
                          (o_ours.float() - ref).abs().max().item(),
                          (o_pk.float() - ref).abs().max().item())

            layers = lambda f: (lambda: [f(kv) for kv in kvs])
            it = args.iters
            t["ours"] += graph_time_us(layers(lambda kv: ours.run(q, kv)), it)
            t["ours_pk"] += graph_time_us(layers(lambda kv: ours_pk.run(q, kv)), it)
            t["paged"] += graph_time_us(layers(lambda kv: pg.run(q, kv)), it)
            t["fi"] += graph_time_us(layers(lambda kv: fi.run(q, kv)), it)
            t["split"] += graph_time_us(layers(split), it)
            t["prefix"] += graph_time_us(
                layers(lambda kv: pre.run(q, kv, out=o_pre, lse=lse_pre, return_lse=True)), it)
            t["suf_prefill"] += graph_time_us(layers(lambda kv: sufp.run(q, kv, return_lse=True)), it)
            t["suf_decode"] += graph_time_us(
                layers(lambda kv: dec.run(q, kv, out=o_dec, lse=lse_dec, return_lse=True)), it)
            t["merge"] += graph_time_us(
                layers(lambda kv: merge_state_in_place(o_pre, lse_pre, o_dec, lse_dec)), it)
        ms = {k: v / 1e3 for k, v in t.items()}
        rows.append((bs, L, ms, max_err))
        print(f"bs {bs:>3} prefix {L:>6}: " + "  ".join(f"{k} {v:.2f}" for k, v in ms.items())
              + f"  (max err {max_err:.1e})", flush=True)
        del kvs, ours, ours_pk, pre, fi, sufp, dec, pg
        torch.cuda.empty_cache()

    print("\nAttention ms per draft iteration (16 layers x 6 steps)")
    print(f"{'bs':>3} {'prefix':>7} {'paged':>7} {'ours':>7} {'ours_pk':>7} {'fi':>7} {'split':>7} | {'prefix':>7} {'suf_pf':>7} "
          f"{'suf_dec':>7} {'merge':>7} | {'ours-prefix':>11}")
    for bs, L, m, _ in rows:
        print(f"{bs:>3} {L:>7} {m['paged']:>7.2f} {m['ours']:>7.2f} {m['ours_pk']:>7.2f} {m['fi']:>7.2f} {m['split']:>7.2f} | {m['prefix']:>7.2f} "
              f"{m['suf_prefill']:>7.2f} {m['suf_decode']:>7.2f} {m['merge']:>7.2f} | "
              f"{m['ours'] - m['prefix']:>11.2f}")


if __name__ == "__main__":
    main()
