"""Profile breakdown of a Llama 1B draft forward step.

Measures time spent in attention vs MLP vs other for a single draft
decode step, simulating EAGLE's operating point (topk branches, shared prefix).

Usage:
    uv run python tests/bench_draft_breakdown.py
    uv run python tests/bench_draft_breakdown.py --prefix-len 8192 --topk 10
"""

import argparse
import torch
import torch.nn as nn


def bench_breakdown(
    prefix_len=8192,
    topk=10,
    num_seqs=1,
    hidden_size=2048,
    intermediate_size=8192,
    num_qo_heads=32,
    num_kv_heads=8,
    head_dim=64,
    num_layers=16,
    dtype=torch.float16,
    warmup=20,
    repeat=100,
):
    device = "cuda"
    batch = num_seqs * topk  # draft decode batch size

    # Simulate layer components
    # Q/K/V projection: hidden_size -> (num_qo_heads + 2*num_kv_heads) * head_dim
    qkv_out_dim = (num_qo_heads + 2 * num_kv_heads) * head_dim
    qkv_proj = nn.Linear(hidden_size, qkv_out_dim, bias=False, dtype=dtype, device=device)

    # O projection: num_qo_heads * head_dim -> hidden_size
    o_proj = nn.Linear(num_qo_heads * head_dim, hidden_size, bias=False, dtype=dtype, device=device)

    # MLP: gate_up = hidden_size -> 2*intermediate_size, down = intermediate_size -> hidden_size
    gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False, dtype=dtype, device=device)
    down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)

    # RMSNorm
    rms_norm = nn.LayerNorm(hidden_size, elementwise_affine=True, dtype=dtype, device=device)

    # Simulated KV cache for attention
    total_kv_per_branch = prefix_len + 1
    kv_data = torch.randn(batch, num_kv_heads, total_kv_per_branch, head_dim, device=device, dtype=dtype)
    v_data = torch.randn(batch, num_kv_heads, total_kv_per_branch, head_dim, device=device, dtype=dtype)

    # Input hidden states
    x = torch.randn(batch, hidden_size, device=device, dtype=dtype)

    def time_fn(fn, warmup=warmup, repeat=repeat):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start_evts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        end_evts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        for i in range(repeat):
            start_evts[i].record()
            fn()
            end_evts[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_evts, end_evts)]
        return sorted(times)[len(times) // 2]

    # 1. QKV projection
    def qkv_fn():
        return qkv_proj(x)
    qkv_ms = time_fn(qkv_fn)

    # 2. Attention (SDPA as proxy for flat decode attention)
    q = torch.randn(batch, num_qo_heads, 1, head_dim, device=device, dtype=dtype)

    def attn_fn():
        # Expand KV for GQA
        gqa_ratio = num_qo_heads // num_kv_heads
        k_exp = kv_data.repeat_interleave(gqa_ratio, dim=1)
        v_exp = v_data.repeat_interleave(gqa_ratio, dim=1)
        return torch.nn.functional.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
    attn_ms = time_fn(attn_fn)

    # 2b. Attention using flashinfer (actual kernel used)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "3rdparty", "flashinfer"))
    import flashinfer

    # Flat decode attention via flashinfer
    flat_kv = torch.randn(
        num_seqs * (prefix_len + topk * 5), 2, 1, num_kv_heads, head_dim,
        device=device, dtype=dtype
    )
    stride = prefix_len + topk * 5
    flat_indices_list = []
    for r in range(num_seqs):
        base = r * stride
        prefix_pages = torch.arange(base, base + prefix_len, device=device, dtype=torch.int32)
        for k in range(topk):
            suffix_pages = torch.arange(base + prefix_len + k * 5, base + prefix_len + k * 5 + 1,
                                         device=device, dtype=torch.int32)
            flat_indices_list.append(torch.cat([prefix_pages, suffix_pages]))
    flat_kv_indices = torch.cat(flat_indices_list)
    flat_kv_indptr = torch.arange(0, (batch + 1) * (prefix_len + 1), prefix_len + 1,
                                   device=device, dtype=torch.int32)
    flat_last_page_len = torch.ones(batch, device=device, dtype=torch.int32)
    q_flat = torch.randn(batch, num_qo_heads, head_dim, device=device, dtype=dtype)

    flat_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device), "NHD",
    )
    flat_wrapper.plan(flat_kv_indptr, flat_kv_indices, flat_last_page_len,
                      num_qo_heads, num_kv_heads, head_dim, 1, q_data_type=dtype)

    def fi_attn_fn():
        return flat_wrapper.run(q_flat, flat_kv)
    fi_attn_ms = time_fn(fi_attn_fn)

    # 3. O projection
    attn_out = torch.randn(batch, num_qo_heads * head_dim, device=device, dtype=dtype)
    def o_fn():
        return o_proj(attn_out)
    o_ms = time_fn(o_fn)

    # 4. MLP (gate_up + SiLU + down)
    def mlp_fn():
        gu = gate_up_proj(x)
        gate, up = gu.chunk(2, dim=-1)
        return down_proj(torch.nn.functional.silu(gate) * up)
    mlp_ms = time_fn(mlp_fn)

    # 5. RMSNorm
    def norm_fn():
        return rms_norm(x)
    norm_ms = time_fn(norm_fn)

    # Summary
    total_per_layer = qkv_ms + fi_attn_ms + o_ms + mlp_ms + 2 * norm_ms
    total_all_layers = total_per_layer * num_layers

    print(f"\n{'='*70}")
    print(f"Draft Model Forward Step Breakdown")
    print(f"  Llama 3.2 1B: hidden={hidden_size}, inter={intermediate_size}")
    print(f"  heads={num_qo_heads}/{num_kv_heads}, head_dim={head_dim}, layers={num_layers}")
    print(f"  batch={batch} (topk={topk}, seqs={num_seqs}), prefix_len={prefix_len}")
    print(f"{'='*70}")
    print(f"  {'Component':>20}  {'per-layer(ms)':>13}  {'all-layers(ms)':>14}  {'% of total':>10}")
    print(f"  {'-'*20}  {'-'*13}  {'-'*14}  {'-'*10}")
    components = [
        ("QKV proj", qkv_ms),
        ("Attention (FI flat)", fi_attn_ms),
        ("Attention (SDPA)", attn_ms),
        ("O proj", o_ms),
        ("MLP (gate+up+down)", mlp_ms),
        ("RMSNorm (×2)", 2 * norm_ms),
    ]
    for name, ms in components:
        pct = ms / total_per_layer * 100 if "SDPA" not in name else 0
        all_ms = ms * num_layers
        marker = " ←" if "FI flat" in name else ""
        print(f"  {name:>20}  {ms:>13.4f}  {all_ms:>14.4f}  {pct:>9.1f}%{marker}")

    print(f"  {'-'*20}  {'-'*13}  {'-'*14}  {'-'*10}")
    print(f"  {'TOTAL (excl SDPA)':>20}  {total_per_layer:>13.4f}  {total_all_layers:>14.4f}  {'100.0%':>10}")

    # Cascade savings estimate
    cascade_attn_ms = fi_attn_ms * 0.28  # ~3.6x speedup from kernel benchmark
    saved_per_layer = fi_attn_ms - cascade_attn_ms
    saved_all_layers = saved_per_layer * num_layers
    new_total = total_all_layers - saved_all_layers
    speedup = total_all_layers / new_total

    print(f"\n  Cascade attention estimate:")
    print(f"    Flat attention:    {fi_attn_ms:.4f}ms/layer × {num_layers} = {fi_attn_ms*num_layers:.4f}ms")
    print(f"    Cascade attention: {cascade_attn_ms:.4f}ms/layer × {num_layers} = {cascade_attn_ms*num_layers:.4f}ms")
    print(f"    Saved per step:    {saved_all_layers:.4f}ms")
    print(f"    Step speedup:      {speedup:.3f}x")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-len", type=int, default=8192)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--num-seqs", type=int, default=1)
    args = parser.parse_args()

    bench_breakdown(prefix_len=args.prefix_len, topk=args.topk, num_seqs=args.num_seqs)
