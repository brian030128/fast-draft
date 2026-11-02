"""Profile Fast Draft's cascade kernel vs stock FlashInfer MultiLevelCascade on draft shapes.

Both wrappers are driven the way their SGLang draft backends drive them:
  * fi      : MultiLevelCascadeAttentionWrapper(2 levels), stock plan() every draft step
  * ours    : CascadeBatchAttentionWrapper, plan_for_draft() once per iteration at max
              depth + update_draft_step() per step (the E2E path)
  * ours_ps : CascadeBatchAttentionWrapper, plan() every step (control: same kernel,
              per-step schedule)

Reported per prefix length:
  1. GPU time of one attention call per draft step (CUDA graph replay, CUDA events)
  2. per-kernel GPU time breakdown (torch.profiler)
  3. host planning time per draft iteration
  4. roll-up per draft iteration: layers * sum over steps of (1) + (3)

Index generation (shared by both backends) is excluded. KV pages are scattered
through the pool as in a paged cache.

    CUDA_VISIBLE_DEVICES=<idle gpu> uv run python tests/profile_draft_kernels.py
"""

import argparse
import time
from collections import defaultdict

import torch
from flashinfer.attention import CascadeBatchAttentionWrapper
from flashinfer.cascade import MultiLevelCascadeAttentionWrapper

DTYPE = torch.bfloat16


def make_problem(prefix_len, topk, depth, hkv, d, layers, seed=0):
    """Page indices plus one K/V pool per layer, like the draft model's KV cache.

    Every timed call runs all layers back to back, so K/V is read from HBM as
    in a real draft step rather than staying resident in L2 (96 MB on Ada).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    pool_tokens = prefix_len + topk * depth + 4096
    perm = torch.randperm(pool_tokens, generator=g).to(torch.int32)
    shared = perm[:prefix_len].cuda()
    unique = perm[prefix_len : prefix_len + topk * depth].view(topk, depth).cuda()
    kvs = [
        (torch.randn(pool_tokens, hkv, d, device="cuda", dtype=DTYPE),
         torch.randn(pool_tokens, hkv, d, device="cuda", dtype=DTYPE))
        for _ in range(layers)
    ]
    return shared, unique, kvs


def host_indptrs(prefix_len, topk, s):
    """Level-0 / level-1 host arrays for suffix length s (one request)."""
    return dict(
        qo0=torch.tensor([0, topk], dtype=torch.int32),
        qo1=torch.arange(topk + 1, dtype=torch.int32),
        kv0=torch.tensor([0, prefix_len], dtype=torch.int32),
        kv1=torch.arange(0, (topk + 1) * s, s, dtype=torch.int32)[: topk + 1],
        len0=torch.tensor([prefix_len], dtype=torch.int32),
        len1=torch.full((topk,), s, dtype=torch.int32),
    )


def graph_time_us(fn, iters):
    """Mean GPU time per fn() call, measured by replaying a captured graph."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (iters * 10)


def kernel_breakdown(fn, reps=50):
    """Mean GPU time per call of each CUDA kernel fn() launches."""
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    acc = defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            acc[e.name] += e.device_time
    return {name: t / reps for name, t in acc.items()}


def short(name):
    """Kernel name without namespaces or template arguments."""
    return name.split("<")[0].split("(")[0].split("::")[-1][:48]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-lens", default="16384,32768,52203,65536,97404")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--steps", type=int, default=7, help="speculative_num_steps")
    ap.add_argument("--hq", type=int, default=16)
    ap.add_argument("--hkv", type=int, default=4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--plan-reps", type=int, default=50)
    args = ap.parse_args()

    topk, D = args.topk, args.steps - 1
    hq, hkv, d = args.hq, args.hkv, args.d
    common_ours = dict(
        num_qo_heads=hq, num_kv_heads=hkv, head_dim_qk=d, head_dim_vo=d, page_size=1,
        causal=False, sm_scale=None, q_data_type=DTYPE, kv_data_type=DTYPE,
    )
    common_fi = dict(
        num_qo_heads=hq, num_kv_heads=hkv, head_dim=d, page_size=1, causal=False,
        q_data_type=DTYPE, kv_data_type=DTYPE,
    )
    fi_ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    print(f"GPU: {torch.cuda.get_device_name()}  shapes: hq={hq} hkv={hkv} d={d} "
          f"topk={topk} draft_steps={D} layers={args.layers}\n")

    summary = []
    for L in [int(x) for x in args.prefix_lens.split(",")]:
        shared, unique, kvs = make_problem(L, topk, D, hkv, d, args.layers)
        kv = kvs[0]
        nl = args.layers

        def layers_of(run):
            """One draft step: the attention call for every layer, in order."""
            return lambda: [run(kv_l) for kv_l in kvs]
        q = torch.randn(topk, hq, d, device="cuda", dtype=DTYPE)

        fi = [MultiLevelCascadeAttentionWrapper(2, fi_ws, "NHD") for _ in range(D)]
        ours_ps = [CascadeBatchAttentionWrapper(2, "NHD", "cuda") for _ in range(D)]
        ours = CascadeBatchAttentionWrapper(2, "NHD", "cuda")

        def plan_fi(s):
            h = host_indptrs(L, topk, s)
            fi[s - 1].plan(
                [h["qo0"], h["qo1"]], [h["kv0"], h["kv1"]],
                [shared, unique[:, :s].contiguous().view(-1)],
                [torch.ones(1, dtype=torch.int32), torch.ones(topk, dtype=torch.int32)],
                **common_fi,
            )

        def plan_ours_ps(s):
            h = host_indptrs(L, topk, s)
            ours_ps[s - 1].plan(
                [h["qo0"], h["qo1"]], [h["kv0"], h["kv1"]],
                [shared, unique[:, :s].contiguous().view(-1)],
                [h["len0"], h["len1"]], **common_ours,
            )

        def plan_ours_once(first):
            h = host_indptrs(L, topk, D)
            ours.plan_for_draft(
                max_draft_depth=D, first_call=first,
                qo_indptr_host_arr=[h["qo0"], h["qo1"]],
                kv_indptr_host_arr=[h["kv0"], h["kv1"]],
                kv_indices_arr=[shared, unique.contiguous().view(-1)],
                kv_len_host_arr=[h["len0"], h["len1"]], **common_ours,
            )

        # Warm up / JIT.
        for s in range(1, D + 1):
            plan_fi(s)
            plan_ours_ps(s)
        plan_ours_once(True)
        torch.cuda.synchronize()

        # --- correctness: all three agree at every step ---
        max_diff = 0.0
        for s in range(1, D + 1):
            plan_fi(s)
            o_fi = fi[s - 1].run(q, kv)
            o_ps, _ = ours_ps[s - 1].run(q, kv)
            plan_ours_once(False)
            ours.update_draft_step(step_kv_len=s + 1, step_kv_end=s)
            o_once, _ = ours.run(q, kv)
            max_diff = max(max_diff, (o_fi.float() - o_ps.float()).abs().max().item(),
                           (o_fi.float() - o_once.float()).abs().max().item())

        # --- 1. GPU time per call, per step ---
        t = {"fi": [], "ours": [], "ours_ps": []}
        bd = {"fi": defaultdict(float), "ours": defaultdict(float)}
        for s in range(1, D + 1):
            plan_fi(s)
            run_fi = layers_of(lambda kv_l: fi[s - 1].run(q, kv_l))
            t["fi"].append(graph_time_us(run_fi, args.iters) / nl)
            plan_ours_ps(s)
            run_ps = layers_of(lambda kv_l: ours_ps[s - 1].run(q, kv_l))
            t["ours_ps"].append(graph_time_us(run_ps, args.iters) / nl)
            plan_ours_once(False)
            ours.update_draft_step(step_kv_len=s + 1, step_kv_end=s)
            run_ours = layers_of(lambda kv_l: ours.run(q, kv_l))
            t["ours"].append(graph_time_us(run_ours, args.iters) / nl)
            # --- 2. kernel breakdown (per layer call, averaged over steps) ---
            for k_, us in kernel_breakdown(run_fi, reps=10).items():
                bd["fi"][short(k_)] += us / nl / D
            for k_, us in kernel_breakdown(run_ours, reps=10).items():
                bd["ours"][short(k_)] += us / nl / D

        # --- 3. host planning time per draft iteration ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.plan_reps):
            for s in range(1, D + 1):
                plan_fi(s)
        torch.cuda.synchronize()
        host_fi = (time.perf_counter() - t0) / args.plan_reps * 1e3
        t0 = time.perf_counter()
        for _ in range(args.plan_reps):
            plan_ours_once(False)
            for s in range(1, D + 1):
                ours.update_draft_step(step_kv_len=s + 1, step_kv_end=s)
        torch.cuda.synchronize()
        host_ours = (time.perf_counter() - t0) / args.plan_reps * 1e3

        # --- report ---
        print(f"=== prefix {L}  (max |fi - ours| = {max_diff:.3g}) ===")
        print(f"  {'step':>4} {'fi (us)':>9} {'ours (us)':>10} {'ours_ps (us)':>13} {'fi/ours':>8}")
        for i in range(D):
            print(f"  {i+1:>4} {t['fi'][i]:>9.1f} {t['ours'][i]:>10.1f} {t['ours_ps'][i]:>13.1f} "
                  f"{t['fi'][i]/t['ours'][i]:>7.2f}x")
        for name in ("fi", "ours"):
            parts = ", ".join(f"{k_} {v_:.1f}" for k_, v_ in sorted(bd[name].items(), key=lambda x: -x[1]))
            print(f"  kernels/call ({name}, us, avg over steps): {parts}")
        attn_fi = args.layers * sum(t["fi"]) / 1e3
        attn_ours = args.layers * sum(t["ours"]) / 1e3
        print(f"  per draft iteration: attention fi {attn_fi:.2f} ms vs ours {attn_ours:.2f} ms; "
              f"planning fi {host_fi:.2f} ms vs ours {host_ours:.2f} ms")
        print(f"  => fi - ours = {attn_fi - attn_ours:.2f} ms attention + "
              f"{host_fi - host_ours:.2f} ms planning per iteration\n")
        summary.append((L, attn_fi, attn_ours, host_fi, host_ours))
        del fi, ours_ps, ours, kv, kvs
        torch.cuda.empty_cache()

    print("Summary per draft iteration (ms):")
    print(f"  {'prefix':>7} {'attn fi':>8} {'attn ours':>10} {'plan fi':>8} {'plan ours':>10} {'gap':>6}")
    for L, af, ao, hf, ho in summary:
        print(f"  {L:>7} {af:>8.2f} {ao:>10.2f} {hf:>8.2f} {ho:>10.2f} {af + hf - ao - ho:>6.2f}")


if __name__ == "__main__":
    main()
