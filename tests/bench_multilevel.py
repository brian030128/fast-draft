"""Benchmark: CascadeBatchAttentionWrapper vs MultiLevelCascadeAttentionWrapper for N levels."""

import torch
import flashinfer
from flashinfer.attention import CascadeBatchAttentionWrapper


def ceil_div(a, b):
    return (a + b - 1) // b


def build_multilevel_kv_cache(kv_lens, batch_sizes, num_kv_heads, head_dim, page_size, dtype=torch.bfloat16):
    num_levels = len(kv_lens)
    total_pages = 0
    pages_per_level = []
    for kv_len, bs in zip(kv_lens, batch_sizes):
        ppr = ceil_div(kv_len, page_size)
        pages_per_level.append(ppr)
        total_pages += bs * ppr

    kv_data = torch.zeros(total_pages, 2, page_size, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    page_offset = 0
    level_metadata = []

    for level in range(num_levels):
        kv_len = kv_lens[level]
        bs = batch_sizes[level]
        ppr = pages_per_level[level]
        k = torch.randn(bs * kv_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(bs * kv_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        kv_indices = torch.arange(bs * ppr, device="cuda", dtype=torch.int32) + page_offset
        kv_indptr = torch.arange(bs + 1, device="cuda", dtype=torch.int32) * ppr
        last_page_len = torch.full((bs,), (kv_len - 1) % page_size + 1, device="cuda", dtype=torch.int32)
        append_indptr = torch.arange(bs + 1, device="cuda", dtype=torch.int32) * kv_len
        flashinfer.append_paged_kv_cache(
            k, v,
            *flashinfer.get_batch_indices_positions(
                append_indptr, flashinfer.get_seq_lens(kv_indptr, last_page_len, page_size), bs * kv_len,
            ),
            kv_data, kv_indices, kv_indptr, last_page_len, "NHD",
        )
        kv_len_tensor = torch.full((bs,), kv_len, device="cuda", dtype=torch.int32)
        level_metadata.append(dict(
            kv_indices=kv_indices, kv_indptr=kv_indptr,
            last_page_len=last_page_len, kv_len_tensor=kv_len_tensor,
        ))
        page_offset += bs * ppr
    return kv_data, level_metadata


def build_qo_indptr_arr(batch_sizes, total_batch, qo_len):
    arr = []
    for bs in batch_sizes:
        qpr = (total_batch // bs) * qo_len
        arr.append(torch.arange(bs + 1, device="cuda", dtype=torch.int32) * qpr)
    return arr


def bench(fn, warmup=50, repeat=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return times[len(times) // 2]


def main():
    configs = [
        ("2L", [512, 5],            [1, 16],        16),
        ("2L", [2048, 5],           [4, 16],        16),
        ("2L", [8192, 5],           [1, 16],        16),
        ("3L", [1024, 128, 5],      [1, 4, 16],     16),
        ("3L", [2048, 256, 5],      [1, 4, 32],     32),
        ("3L", [4096, 512, 5],      [1, 4, 16],     16),
        ("4L", [1024, 256, 64, 5],  [1, 2, 4, 16],  16),
        ("4L", [2048, 512, 128, 5], [1, 2, 8, 32],  32),
    ]

    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    qo_len = 1
    dtype = torch.bfloat16

    header = f"  {'tag':>4}  {'kv_lens':>25}  {'batch_sizes':>15}  {'MultiLevel':>12}  {'Fused':>12}  {'Speedup':>8}"
    print(header)
    print("  " + "-" * len(header))

    for tag, kv_lens, batch_sizes, total_batch in configs:
        torch.manual_seed(42)
        num_levels = len(kv_lens)
        kv_data, lm = build_multilevel_kv_cache(
            kv_lens, batch_sizes, num_kv_heads, head_dim, page_size, dtype,
        )
        q = torch.randn(total_batch * qo_len, num_qo_heads, head_dim, device="cuda", dtype=dtype)
        qo_arr = build_qo_indptr_arr(batch_sizes, total_batch, qo_len)

        # MultiLevel reference (unfused)
        ref = flashinfer.MultiLevelCascadeAttentionWrapper(
            num_levels, torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda"), "NHD",
        )
        ref.plan(
            qo_arr,
            [m["kv_indptr"] for m in lm],
            [m["kv_indices"] for m in lm],
            [m["last_page_len"] for m in lm],
            num_qo_heads, num_kv_heads, head_dim, page_size,
            causal=True, q_data_type=dtype,
        )
        ref_ms = bench(lambda: ref.run(q, kv_data))

        # Fused cascade
        cascade = CascadeBatchAttentionWrapper(
            num_levels=num_levels, kv_layout="NHD", device="cuda",
        )
        cascade.plan(
            qo_arr,
            [m["kv_indptr"] for m in lm],
            [m["kv_indices"] for m in lm],
            [m["kv_len_tensor"] for m in lm],
            num_qo_heads, num_kv_heads, head_dim, head_dim,
            page_size, causal=True,
            q_data_type=dtype, kv_data_type=dtype,
        )
        fused_ms = bench(lambda: cascade.run(q, kv_data))

        speedup = ref_ms / fused_ms
        print(f"  {tag:>4}  {str(kv_lens):>25}  {str(batch_sizes):>15}  {ref_ms:>10.4f}ms  {fused_ms:>10.4f}ms  {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
