"""
vLLM EAGLE Tree Draft Benchmark — Original vs Cascade Attention

Benchmarks tree-structured EAGLE speculative decoding in vLLM with
the TREE_ATTN backend, comparing original unified_attention kernel
against FlashInfer cascade attention (VLLM_CASCADE_DRAFT=1).

Usage:
    CUDA_VISIBLE_DEVICES=5 CUDA_HOME=/usr/local/cuda-12.8 \
    uv run python tests/bench_vllm_tree.py \
        --model Qwen/Qwen3-4B \
        --draft-model brian920128/Qwen3-4B_eagle3 \
        --method eagle3 \
        --prompt-length 50000 \
        --max-model-len 55000 \
        --eagle-topk 4 --tree-depth 3 \
        --output-len 256 --num-requests 3
"""

import os
# Must be set before any vLLM imports
os.environ["VLLM_DRAFT_TIMING"] = "1"
# Run engine in-process so we can read draft timing events from the same process
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import argparse
import json
import subprocess
import sys
import time
from typing import Optional

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def make_tree_choices(topk: int, depth: int) -> list[tuple[int, ...]]:
    """Generate a beam-style tree: topk nodes per level, depth levels.

    At level 1, root expands into topk children.
    At subsequent levels, each node from the previous level expands
    its first child (index 0), giving topk nodes per level.
    Total nodes = topk * depth.

    Example: topk=3, depth=2 ->
      Level 1: (0,), (1,), (2,)
      Level 2: (0,0), (1,0), (2,0)
    """
    choices = []
    # Level 1: topk children of root
    level_nodes = [(i,) for i in range(topk)]
    choices.extend(level_nodes)
    # Subsequent levels: extend each node with child index 0
    for _ in range(1, depth):
        level_nodes = [node + (0,) for node in level_nodes]
        choices.extend(level_nodes)
    return choices


def build_prompt_ids(tokenizer, prompt_length: int) -> list[int]:
    """Build a prompt of approximately `prompt_length` tokens."""
    # Use a simple repeated sentence to fill up to desired length.
    seed_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In a distant galaxy, stars are born and die in cycles "
        "spanning billions of years. "
    )
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    repeats = (prompt_length // len(seed_ids)) + 1
    prompt_ids = (seed_ids * repeats)[:prompt_length]
    return prompt_ids


def run_benchmark(
    model: str,
    draft_model: str,
    method: str,
    prompt_ids: list[int],
    tree_choices: list[tuple[int, ...]],
    num_spec_tokens: int,
    output_len: int,
    num_requests: int,
    max_model_len: int,
    enforce_eager: bool,
    gpu_mem_util: float,
    cascade: bool,
    max_num_batched_tokens: Optional[int] = None,
) -> dict:
    """Run a single benchmark configuration and return results."""
    mode = "cascade" if cascade else "original"
    print(f"\n{'='*60}")
    print(f"  Mode: {mode.upper()}")
    print(f"  VLLM_CASCADE_DRAFT={os.environ.get('VLLM_CASCADE_DRAFT', '0')}")
    print(f"  VLLM_ATTENTION_BACKEND={os.environ.get('VLLM_ATTENTION_BACKEND', 'not set')}")
    print(f"{'='*60}\n")

    speculative_config = {
        "method": method,
        "model": draft_model,
        "num_speculative_tokens": num_spec_tokens,
        "speculative_token_tree": str(tree_choices),
    }

    llm = LLM(
        model=model,
        trust_remote_code=True,
        speculative_config=speculative_config,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        disable_log_stats=False,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=output_len)
    prompts = [TokensPrompt(prompt_token_ids=prompt_ids)] * num_requests

    # Warmup
    print("Warmup run...")
    _ = llm.generate(prompts[:1], sampling_params=sampling_params)

    # Clear any warmup draft timing events
    from vllm.v1.spec_decode.eagle import get_draft_timing_ms
    get_draft_timing_ms()  # discard warmup events

    # Timed run
    print(f"Benchmark run ({num_requests} requests)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Collect draft timing
    draft_total_ms, draft_calls = get_draft_timing_ms()

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    # Collect metrics
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted = 0
    ttft_sum = 0.0
    ttft_count = 0
    prefill_time_sum = 0.0
    prefill_count = 0
    decode_time_sum = 0.0
    decode_count = 0
    try:
        from vllm.v1.metrics.reader import Counter, Histogram
        metrics = llm.get_metrics()
        for m in metrics:
            if m.name == "vllm:spec_decode_num_drafts" and isinstance(m, Counter):
                num_drafts += m.value
            elif m.name == "vllm:spec_decode_num_draft_tokens" and isinstance(m, Counter):
                num_draft_tokens += m.value
            elif m.name == "vllm:spec_decode_num_accepted_tokens" and isinstance(m, Counter):
                num_accepted += m.value
            elif m.name == "vllm:time_to_first_token_seconds" and isinstance(m, Histogram):
                ttft_sum += m.sum
                ttft_count += m.count
            elif m.name == "vllm:request_prefill_time_seconds" and isinstance(m, Histogram):
                prefill_time_sum += m.sum
                prefill_count += m.count
            elif m.name == "vllm:request_decode_time_seconds" and isinstance(m, Histogram):
                decode_time_sum += m.sum
                decode_count += m.count
    except Exception:
        pass

    accept_len = 1 + (num_accepted / num_drafts) if num_drafts > 0 else float("nan")
    mean_ttft = ttft_sum / ttft_count if ttft_count > 0 else float("nan")
    mean_prefill = prefill_time_sum / prefill_count if prefill_count > 0 else float("nan")
    mean_decode = decode_time_sum / decode_count if decode_count > 0 else float("nan")
    draft_tps = num_draft_tokens / mean_decode if mean_decode > 0 else float("nan")

    # Draft/verify breakdown from CUDA event timing
    draft_time_s = draft_total_ms / 1000.0
    mean_draft_per_req = draft_time_s / num_requests if num_requests > 0 else float("nan")
    mean_verify_per_req = mean_decode - mean_draft_per_req if mean_decode > 0 else float("nan")

    result = {
        "mode": mode,
        "elapsed_s": elapsed,
        "num_requests": num_requests,
        "prompt_length": len(prompt_ids),
        "total_output_tokens": total_output_tokens,
        "tokens_per_sec": total_output_tokens / elapsed if elapsed > 0 else 0,
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted": num_accepted,
        "mean_accept_len": accept_len,
        "mean_ttft_s": mean_ttft,
        "mean_prefill_s": mean_prefill,
        "mean_decode_s": mean_decode,
        "mean_draft_s": mean_draft_per_req,
        "mean_verify_s": mean_verify_per_req,
        "draft_calls": draft_calls,
        "draft_tokens_per_sec": draft_tps,
    }

    print(f"\n--- {mode.upper()} Results ---")
    print(f"  TTFT:           {mean_ttft:.3f}s")
    print(f"  Prefill time:   {mean_prefill:.3f}s")
    print(f"  Decode time:    {mean_decode:.3f}s  (draft+verify)")
    print(f"    Draft time:   {mean_draft_per_req:.3f}s  ({draft_calls} calls)")
    print(f"    Verify time:  {mean_verify_per_req:.3f}s")
    print(f"  Total time:     {elapsed:.2f}s")
    print(f"  Output tokens:  {total_output_tokens}")
    print(f"  Throughput:     {result['tokens_per_sec']:.1f} tok/s")
    print(f"  Draft tok/s:    {draft_tps:.1f}")
    print(f"  Mean accept:    {accept_len:.2f}")

    # Clean up GPU memory
    del llm
    torch.cuda.empty_cache()

    return result


def run_ar_benchmark(
    model: str,
    prompt_ids: list[int],
    output_len: int,
    num_requests: int,
    max_model_len: int,
    enforce_eager: bool,
    gpu_mem_util: float,
    max_num_batched_tokens: Optional[int] = None,
) -> dict:
    """Run autoregressive (no speculative decoding) benchmark."""
    print(f"\n{'='*60}")
    print(f"  Mode: AUTOREGRESSIVE (no spec decode)")
    print(f"{'='*60}\n")

    llm = LLM(
        model=model,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        disable_log_stats=False,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=output_len)
    prompts = [TokensPrompt(prompt_token_ids=prompt_ids)] * num_requests

    # Warmup
    print("Warmup run...")
    _ = llm.generate(prompts[:1], sampling_params=sampling_params)

    # Timed run
    print(f"Benchmark run ({num_requests} requests)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    # Collect metrics
    ttft_sum = 0.0
    ttft_count = 0
    prefill_time_sum = 0.0
    prefill_count = 0
    decode_time_sum = 0.0
    decode_count = 0
    try:
        from vllm.v1.metrics.reader import Counter, Histogram
        metrics = llm.get_metrics()
        for m in metrics:
            if m.name == "vllm:time_to_first_token_seconds" and isinstance(m, Histogram):
                ttft_sum += m.sum
                ttft_count += m.count
            elif m.name == "vllm:request_prefill_time_seconds" and isinstance(m, Histogram):
                prefill_time_sum += m.sum
                prefill_count += m.count
            elif m.name == "vllm:request_decode_time_seconds" and isinstance(m, Histogram):
                decode_time_sum += m.sum
                decode_count += m.count
    except Exception:
        pass

    mean_ttft = ttft_sum / ttft_count if ttft_count > 0 else float("nan")
    mean_prefill = prefill_time_sum / prefill_count if prefill_count > 0 else float("nan")
    mean_decode = decode_time_sum / decode_count if decode_count > 0 else float("nan")

    result = {
        "mode": "ar",
        "elapsed_s": elapsed,
        "num_requests": num_requests,
        "prompt_length": len(prompt_ids),
        "total_output_tokens": total_output_tokens,
        "tokens_per_sec": total_output_tokens / elapsed if elapsed > 0 else 0,
        "mean_ttft_s": mean_ttft,
        "mean_prefill_s": mean_prefill,
        "mean_decode_s": mean_decode,
    }

    print(f"\n--- AUTOREGRESSIVE Results ---")
    print(f"  TTFT:           {mean_ttft:.3f}s")
    print(f"  Prefill time:   {mean_prefill:.3f}s")
    print(f"  Decode time:    {mean_decode:.3f}s")
    print(f"  Total time:     {elapsed:.2f}s")
    print(f"  Output tokens:  {total_output_tokens}")
    print(f"  Throughput:     {result['tokens_per_sec']:.1f} tok/s")

    del llm
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="vLLM EAGLE Tree Draft Benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, default=None,
                        help="Draft model (required for spec decode modes)")
    parser.add_argument("--method", type=str, default="eagle3",
                        choices=["eagle", "eagle3"])
    parser.add_argument("--prompt-length", type=int, default=50000)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-requests", type=int, default=3)
    parser.add_argument("--eagle-topk", type=int, default=4)
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=55000)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="Max tokens per batch (default: auto)")
    parser.add_argument("--only", type=str, default=None,
                        choices=["original", "cascade", "ar"],
                        help="Run only one mode (ar=autoregressive, no spec decode)")
    parser.add_argument("--result-file", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    if args.only != "ar" and args.draft_model is None:
        parser.error("--draft-model is required for spec decode modes")

    # Set TREE_ATTN backend
    os.environ["VLLM_ATTENTION_BACKEND"] = "TREE_ATTN"

    tree_choices = make_tree_choices(args.eagle_topk, args.tree_depth)
    num_spec_tokens = len(tree_choices)
    print(f"Tree: topk={args.eagle_topk}, depth={args.tree_depth}, "
          f"total_nodes={num_spec_tokens}")
    print(f"Tree choices: {tree_choices[:10]}{'...' if len(tree_choices) > 10 else ''}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = build_prompt_ids(tokenizer, args.prompt_length)
    print(f"Prompt length: {len(prompt_ids)} tokens")

    results = []

    if args.only is not None:
        # Single mode: run in-process
        modes = [args.only]
    else:
        # All modes: run each in a subprocess for clean GPU memory
        modes = ["ar", "original", "cascade"]

    for mode in modes:
        if args.only is not None:
            # In-process path
            if mode == "ar":
                r = run_ar_benchmark(
                    model=args.model,
                    prompt_ids=prompt_ids,
                    output_len=args.output_len,
                    num_requests=args.num_requests,
                    max_model_len=args.max_model_len,
                    enforce_eager=args.enforce_eager,
                    gpu_mem_util=args.gpu_mem_util,
                    max_num_batched_tokens=args.max_num_batched_tokens,
                )
            else:
                os.environ["VLLM_CASCADE_DRAFT"] = "1" if mode == "cascade" else "0"
                r = run_benchmark(
                    model=args.model,
                    draft_model=args.draft_model,
                    method=args.method,
                    prompt_ids=prompt_ids,
                    tree_choices=tree_choices,
                    num_spec_tokens=num_spec_tokens,
                    output_len=args.output_len,
                    num_requests=args.num_requests,
                    max_model_len=args.max_model_len,
                    enforce_eager=args.enforce_eager,
                    gpu_mem_util=args.gpu_mem_util,
                    cascade=(mode == "cascade"),
                    max_num_batched_tokens=args.max_num_batched_tokens,
                )
        else:
            # Subprocess path: re-invoke with --only and --result-file
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            tmp.close()
            cmd = [sys.executable, __file__,
                   "--model", args.model,
                   "--draft-model", args.draft_model,
                   "--method", args.method,
                   "--prompt-length", str(args.prompt_length),
                   "--output-len", str(args.output_len),
                   "--num-requests", str(args.num_requests),
                   "--eagle-topk", str(args.eagle_topk),
                   "--tree-depth", str(args.tree_depth),
                   "--max-model-len", str(args.max_model_len),
                   "--gpu-mem-util", str(args.gpu_mem_util),
                   "--only", mode,
                   "--result-file", tmp.name]
            if args.enforce_eager:
                cmd.append("--enforce-eager")
            if args.max_num_batched_tokens is not None:
                cmd.extend(["--max-num-batched-tokens",
                             str(args.max_num_batched_tokens)])
            print(f"\n>>> Spawning subprocess for {mode} mode...")
            env = os.environ.copy()
            env["VLLM_CASCADE_DRAFT"] = "1" if mode == "cascade" else "0"
            proc = subprocess.run(cmd, env=env)
            if proc.returncode != 0:
                print(f"ERROR: {mode} subprocess failed")
                continue
            with open(tmp.name) as f:
                data = json.load(f)
            r = data["results"][0]
            os.unlink(tmp.name)
        results.append(r)

    # Summary
    by_mode = {r["mode"]: r for r in results}
    spec_modes = [m for m in ["original", "cascade"] if m in by_mode]

    if len(spec_modes) == 2:
        orig, casc = by_mode["original"], by_mode["cascade"]
        speedup = orig["elapsed_s"] / casc["elapsed_s"] if casc["elapsed_s"] > 0 else float("nan")
        decode_speedup = orig["mean_decode_s"] / casc["mean_decode_s"] if casc["mean_decode_s"] > 0 else float("nan")
        draft_speedup = orig["mean_draft_s"] / casc["mean_draft_s"] if casc["mean_draft_s"] > 0 else float("nan")
        verify_speedup = orig["mean_verify_s"] / casc["mean_verify_s"] if casc["mean_verify_s"] > 0 else float("nan")
        ar = by_mode.get("ar")
        prompt_len = orig["prompt_length"]
        print(f"\n{'='*72}")
        print(f"  SUMMARY  (prompt={prompt_len} tokens)")
        print(f"{'='*72}")
        if ar:
            print(f"  {'':20s} {'AR':>10s} {'Original':>12s} {'Cascade':>12s} {'Speedup':>8s}")
            print(f"  {'-'*62}")
            print(f"  {'TTFT':20s} {ar['mean_ttft_s']:>9.3f}s {orig['mean_ttft_s']:>11.3f}s {casc['mean_ttft_s']:>11.3f}s")
            print(f"  {'Prefill':20s} {ar['mean_prefill_s']:>9.3f}s {orig['mean_prefill_s']:>11.3f}s {casc['mean_prefill_s']:>11.3f}s")
            print(f"  {'Decode':20s} {ar['mean_decode_s']:>9.3f}s {orig['mean_decode_s']:>11.3f}s {casc['mean_decode_s']:>11.3f}s {decode_speedup:>7.2f}x")
            print(f"  {'  Draft':20s} {'':>10s} {orig['mean_draft_s']:>11.3f}s {casc['mean_draft_s']:>11.3f}s {draft_speedup:>7.2f}x")
            print(f"  {'  Verify':20s} {'':>10s} {orig['mean_verify_s']:>11.3f}s {casc['mean_verify_s']:>11.3f}s {verify_speedup:>7.2f}x")
            print(f"  {'Total':20s} {ar['elapsed_s']:>9.2f}s {orig['elapsed_s']:>11.2f}s {casc['elapsed_s']:>11.2f}s {speedup:>7.2f}x")
            print(f"  {'Output tok/s':20s} {ar['tokens_per_sec']:>9.1f}  {orig['tokens_per_sec']:>11.1f}  {casc['tokens_per_sec']:>11.1f}")
            print(f"  {'Draft tok/s':20s} {'':>10s} {orig['draft_tokens_per_sec']:>11.1f}  {casc['draft_tokens_per_sec']:>11.1f}")
            print(f"  {'Mean accept len':20s} {'':>10s} {orig['mean_accept_len']:>11.2f}  {casc['mean_accept_len']:>11.2f}")
            ar_vs_casc = ar["elapsed_s"] / casc["elapsed_s"] if casc["elapsed_s"] > 0 else float("nan")
            ar_vs_orig = ar["elapsed_s"] / orig["elapsed_s"] if orig["elapsed_s"] > 0 else float("nan")
            print(f"  {'-'*62}")
            print(f"  Spec vs AR:  Original {ar_vs_orig:.2f}x,  Cascade {ar_vs_casc:.2f}x")
        else:
            print(f"  {'':20s} {'Original':>12s} {'Cascade':>12s} {'Speedup':>8s}")
            print(f"  {'-'*52}")
            print(f"  {'TTFT':20s} {orig['mean_ttft_s']:>11.3f}s {casc['mean_ttft_s']:>11.3f}s")
            print(f"  {'Prefill':20s} {orig['mean_prefill_s']:>11.3f}s {casc['mean_prefill_s']:>11.3f}s")
            print(f"  {'Decode (draft+vfy)':20s} {orig['mean_decode_s']:>11.3f}s {casc['mean_decode_s']:>11.3f}s {decode_speedup:>7.2f}x")
            print(f"  {'  Draft':20s} {orig['mean_draft_s']:>11.3f}s {casc['mean_draft_s']:>11.3f}s {draft_speedup:>7.2f}x")
            print(f"  {'  Verify':20s} {orig['mean_verify_s']:>11.3f}s {casc['mean_verify_s']:>11.3f}s {verify_speedup:>7.2f}x")
            print(f"  {'Total':20s} {orig['elapsed_s']:>11.2f}s {casc['elapsed_s']:>11.2f}s {speedup:>7.2f}x")
            print(f"  {'Output tok/s':20s} {orig['tokens_per_sec']:>11.1f}  {casc['tokens_per_sec']:>11.1f}")
            print(f"  {'Draft tok/s':20s} {orig['draft_tokens_per_sec']:>11.1f}  {casc['draft_tokens_per_sec']:>11.1f}")
            print(f"  {'Mean accept len':20s} {orig['mean_accept_len']:>11.2f}  {casc['mean_accept_len']:>11.2f}")
        print(f"{'='*72}")

    if args.result_file:
        with open(args.result_file, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"Results saved to {args.result_file}")


if __name__ == "__main__":
    main()
