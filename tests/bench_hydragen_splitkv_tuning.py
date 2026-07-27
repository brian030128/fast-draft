"""Can Hydragen's level-0 bandwidth be recovered by tuning FlashInfer's split-KV?

Stock BatchPrefillWithPagedKVCacheWrapper.plan() exposes fixed_split_size
(and split-kv is already enabled by default). If tuning it closes the gap to
our fused kernel, our contribution is a tuning artifact and we should say so.
"""
import sys, time, torch, flashinfer
from flashinfer.cascade import merge_state_in_place
sys.path.insert(0, "/home/u4320956/fast-draft/.claude/worktrees/hydragen-compare/3rdparty/flashinfer")
from flashinfer.attention import CascadeBatchAttentionWrapper

dev, dtype = "cuda", torch.float16
NS, TOPK, PREFIX, SOFF = 2, 5, 55664, 3
NQ, NKV, D = 32, 8, 64
NB = NS * TOPK
BYTES = NKV * D * 2 * 2  # K+V, fp16

def timeit(fn, warmup=20, repeat=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(repeat): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / repeat

tot = NS * (PREFIX + TOPK * 8)
k = torch.randn(tot, NKV, D, device=dev, dtype=dtype)
v = torch.randn(tot, NKV, D, device=dev, dtype=dtype)
kv = (k, v)
q = torch.randn(NB, NQ, D, device=dev, dtype=dtype)

qo0 = torch.arange(0, (NS + 1) * TOPK, TOPK, dtype=torch.int32)
kvp0 = torch.arange(0, (NS + 1) * PREFIX, PREFIX, dtype=torch.int32)
kvi0 = torch.arange(NS * PREFIX, dtype=torch.int32, device=dev)
lp0 = torch.ones(NS, dtype=torch.int32)

qo1 = torch.arange(NB + 1, dtype=torch.int32)
kvp1 = torch.arange(0, (NB + 1) * SOFF, SOFF, dtype=torch.int32)
kvi1 = torch.arange(NS * PREFIX, NS * PREFIX + NB * SOFF, dtype=torch.int32, device=dev)
lp1 = torch.ones(NB, dtype=torch.int32)

ws0 = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
ws1 = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)

prefix_bytes = NS * PREFIX * BYTES / 1e9

print(f"config: bs={NS} topk={TOPK} prefix={PREFIX} {NQ}q/{NKV}kv d={D}")
print(f"level-0 KV volume = {prefix_bytes:.3f} GB\n")
print(f"  {'level-0 config':>26}  {'l0':>9}  {'total hy':>9}  {'l0 BW':>10}")
print(f"  {'-'*26}  {'-'*9}  {'-'*9}  {'-'*10}")

w1 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws1, "NHD")
w1.plan(qo1, kvp1, kvi1, lp1, NQ, NKV, D, 1, causal=False,
        q_data_type=dtype, kv_data_type=dtype)

best = None
for label, kw in [("default (split-kv auto)", {}),
                  ("disable_split_kv=True", {"disable_split_kv": True}),
                  ("fixed_split_size=16384", {"fixed_split_size": 16384}),
                  ("fixed_split_size=8192", {"fixed_split_size": 8192}),
                  ("fixed_split_size=4096", {"fixed_split_size": 4096}),
                  ("fixed_split_size=2048", {"fixed_split_size": 2048}),
                  ("fixed_split_size=1024", {"fixed_split_size": 1024}),
                  ("fixed_split_size=512", {"fixed_split_size": 512}),
                  ("fixed_split_size=256", {"fixed_split_size": 256})]:
    try:
        w0 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws0, "NHD")
        w0.plan(qo0, kvp0, kvi0, lp0, NQ, NKV, D, 1, causal=False,
                q_data_type=dtype, kv_data_type=dtype, **kw)
        t0 = timeit(lambda: w0.run(q, kv, return_lse=True))
        def full():
            o, l = w1.run(q, kv, return_lse=True)
            oi, li = w0.run(q, kv, return_lse=True)
            merge_state_in_place(o, l, oi, li)
            return o
        thy = timeit(full)
        bw = prefix_bytes / (t0 / 1e3)
        print(f"  {label:>26}  {t0:>8.3f}ms  {thy:>8.3f}ms  {bw:>7.0f}GB/s")
        if best is None or thy < best[1]: best = (label, thy, bw)
    except Exception as ex:
        print(f"  {label:>26}  FAILED: {str(ex)[:40]}")

cas = CascadeBatchAttentionWrapper(num_levels=2, kv_layout="NHD", device=dev)
cas.plan(qo_indptr_arr=[qo0, qo1], kv_indptr_arr=[kvp0, kvp1],
         kv_indices_arr=[kvi0, kvi1],
         kv_len_arr=[torch.full((NS,), PREFIX, dtype=torch.int32),
                     torch.full((NB,), SOFF, dtype=torch.int32)],
         num_qo_heads=NQ, num_kv_heads=NKV, head_dim_qk=D, head_dim_vo=D,
         page_size=1, causal=False, sm_scale=None,
         q_data_type=dtype, kv_data_type=dtype)
tcas = timeit(lambda: cas.run(q, kv)[0])
print(f"\n  {'OURS (fused cascade)':>26}  {'':>9}  {tcas:>8.3f}ms  "
      f"{prefix_bytes/(tcas/1e3):>7.0f}GB/s")
print(f"\n  best tuned hydragen: {best[0]} -> {best[1]:.3f}ms")
print(f"  ours / best tuned hydragen = {best[1]/tcas:.2f}x faster")
