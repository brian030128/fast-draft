"""Is Hydragen's level-0 deficit a tuning problem or a regime problem?

Sweep the number of query rows per sequence (= top-k) at fixed prefix. If the
gap is occupancy/query-starvation, Hydragen's achieved bandwidth should climb
as query rows are added -- approaching its home regime (large batches of
independent completions) -- while ours is already high at low query counts.
"""
import sys, torch, flashinfer
from flashinfer.cascade import merge_state_in_place
sys.path.insert(0, "/home/u4320956/fast-draft/.claude/worktrees/hydragen-compare/3rdparty/flashinfer")
from flashinfer.attention import CascadeBatchAttentionWrapper

dev, dtype = "cuda", torch.float16
NS, PREFIX, SOFF, NQ, NKV, D = 2, 55664, 3, 32, 8, 64
BYTES = NKV * D * 2 * 2

def timeit(fn, warmup=15, repeat=60):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(repeat): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / repeat

MAXTOPK = 128
tot = NS * (PREFIX + MAXTOPK * 8)
k = torch.randn(tot, NKV, D, device=dev, dtype=dtype)
v = torch.randn(tot, NKV, D, device=dev, dtype=dtype)
kv = (k, v)
gb = NS * PREFIX * BYTES / 1e9

print(f"bs={NS} prefix={PREFIX} {NQ}q/{NKV}kv d={D}; level-0 KV = {gb:.3f} GB")
print("(level-0 KV volume is CONSTANT: adding query rows adds no memory traffic)\n")
print(f"  {'top-k':>6}  {'qrows':>6}  {'hydragen l0':>12}  {'hy BW':>9}  "
      f"{'ours':>9}  {'ours BW':>9}  {'ours/hy':>8}")
print(f"  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}")

ws0 = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev)
for TOPK in (1, 2, 5, 10, 16, 32, 64, 128):
    NB = NS * TOPK
    q = torch.randn(NB, NQ, D, device=dev, dtype=dtype)
    qo0 = torch.arange(0, (NS + 1) * TOPK, TOPK, dtype=torch.int32)
    kvp0 = torch.arange(0, (NS + 1) * PREFIX, PREFIX, dtype=torch.int32)
    kvi0 = torch.arange(NS * PREFIX, dtype=torch.int32, device=dev)
    lp0 = torch.ones(NS, dtype=torch.int32)
    qo1 = torch.arange(NB + 1, dtype=torch.int32)
    kvp1 = torch.arange(0, (NB + 1) * SOFF, SOFF, dtype=torch.int32)
    kvi1 = torch.arange(NS * PREFIX, NS * PREFIX + NB * SOFF,
                        dtype=torch.int32, device=dev)
    lp1 = torch.ones(NB, dtype=torch.int32)

    w0 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws0, "NHD")
    w0.plan(qo0, kvp0, kvi0, lp0, NQ, NKV, D, 1, causal=False,
            q_data_type=dtype, kv_data_type=dtype)
    thy = timeit(lambda: w0.run(q, kv, return_lse=True))

    cas = CascadeBatchAttentionWrapper(num_levels=2, kv_layout="NHD", device=dev)
    cas.plan(qo_indptr_arr=[qo0, qo1], kv_indptr_arr=[kvp0, kvp1],
             kv_indices_arr=[kvi0, kvi1],
             kv_len_arr=[torch.full((NS,), PREFIX, dtype=torch.int32),
                         torch.full((NB,), SOFF, dtype=torch.int32)],
             num_qo_heads=NQ, num_kv_heads=NKV, head_dim_qk=D, head_dim_vo=D,
             page_size=1, causal=False, sm_scale=None,
             q_data_type=dtype, kv_data_type=dtype)
    tcas = timeit(lambda: cas.run(q, kv)[0])
    print(f"  {TOPK:>6}  {NB:>6}  {thy:>11.3f}ms  {gb/(thy/1e3):>6.0f}GB/s  "
          f"{tcas:>8.3f}ms  {gb/(tcas/1e3):>6.0f}GB/s  {thy/tcas:>7.2f}x")

print("\n  EAGLE draft decode operates at top-k 4-16 (left end).")
