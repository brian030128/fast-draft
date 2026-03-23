"""E2E benchmark: original vs flat-EAGLE vs cascade-EAGLE draft decode.

Three phases:
  1. Cascade  — EAGLE speculative decoding with cascade draft attention
  2. Flat     — EAGLE speculative decoding with standard flat attention
  3. Original — vanilla autoregressive decoding (no speculation)

Uses random token IDs so arbitrarily long prompts can be tested without
needing a dataset.

Usage:
    CUDA_VISIBLE_DEVICES=1,2 uv run python tests/bench_e2e.py \
        --model-path meta-llama/Llama-3.1-8B \
        --draft-model-path meta-llama/Llama-3.2-1B \
        --prompt-lengths 30000 \
        --eagle-topk 10 \
        --max-new-tokens 512 \
        --num-requests 1 \
        --context-length 31000 \
        --mem-fraction-static 0.40 \
        --batch-size 1 \
        --tp 2
"""

import argparse
import dataclasses
import json
import os
import random
import subprocess
import sys
import tempfile
import time


def make_random_input_ids(length: int, vocab_size: int = 128000, seed: int = 42):
    """Generate random token IDs avoiding special tokens (start from 100)."""
    rng = random.Random(seed)
    return [rng.randint(100, vocab_size - 1) for _ in range(length)]


def create_engine(
    model_path: str,
    draft_model_path: str = None,
    speculative_algorithm: str = "STANDALONE",
    eagle_topk: int = None,
    speculative_num_steps: int = None,
    tp: int = 1,
    context_length: int = None,
    mem_fraction_static: float = None,
):
    """Create an sglang Engine with the given configuration."""
    from sglang.srt.entrypoints.engine import Engine

    kwargs = {
        "model_path": model_path,
        "tp_size": tp,
        "log_level": "warning",
        "enable_metrics": True,
    }
    if context_length is not None:
        kwargs["context_length"] = context_length
    if mem_fraction_static is not None:
        kwargs["mem_fraction_static"] = mem_fraction_static

    if draft_model_path is not None:
        kwargs["speculative_algorithm"] = speculative_algorithm
        kwargs["speculative_draft_model_path"] = draft_model_path
        # sglang requires all three spec params to be None (auto) or all set
        if eagle_topk is not None or speculative_num_steps is not None:
            kwargs["speculative_eagle_topk"] = eagle_topk or 4
            kwargs["speculative_num_steps"] = speculative_num_steps or 10
            kwargs["speculative_num_draft_tokens"] = (speculative_num_steps or 10) + 3

    return Engine(**kwargs)


def run_phase(args, phase: str):
    """Run one benchmark phase. phase is 'original', 'flat', or 'cascade'."""
    # Set cascade env var before engine creation
    if phase == "cascade":
        os.environ["SGLANG_CASCADE_DRAFT"] = "1"
    else:
        os.environ["SGLANG_CASCADE_DRAFT"] = "0"

    # Allow long context
    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

    # Clean prometheus state for multi-engine runs in same process
    try:
        from prometheus_client import REGISTRY
        for name in list(REGISTRY._names_to_collectors):
            try:
                REGISTRY.unregister(REGISTRY._names_to_collectors[name])
            except Exception:
                pass
    except ImportError:
        pass

    draft = args.draft_model_path if phase != "original" else None
    topk = args.eagle_topk if phase != "original" else None
    num_steps = args.speculative_num_steps if phase != "original" else None

    print(f"\n{'='*70}")
    print(f"  Phase: {phase.upper()}"
          + (f"  (SGLANG_CASCADE_DRAFT={os.environ.get('SGLANG_CASCADE_DRAFT', '0')})"
             if phase != "original" else "  (no speculation)"))
    print(f"{'='*70}")

    engine = create_engine(
        model_path=args.model_path,
        draft_model_path=draft,
        speculative_algorithm=args.speculative_algorithm,
        eagle_topk=topk,
        speculative_num_steps=num_steps,
        tp=args.tp,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
    )

    prompt_lengths = [int(x) for x in args.prompt_lengths.split(",")]
    results = []

    for prompt_len in prompt_lengths:
        # Build batch of random-token prompts
        input_ids_batch = [
            make_random_input_ids(prompt_len, seed=42 + i)
            for i in range(args.num_requests)
        ]
        sampling_params = {
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": True,
        }

        # Warmup
        print(f"  Warmup (prompt_len={prompt_len}) ...")
        warmup_ids = make_random_input_ids(min(256, prompt_len), seed=99)
        engine.generate(
            input_ids=warmup_ids,
            sampling_params={"temperature": 0, "max_new_tokens": 4, "ignore_eos": True},
        )

        # Measure TTFT (prefill time) per request via max_new_tokens=1
        print(f"  Measuring TTFT ...")
        ttft_params = {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True}
        ttfts = []
        for i in range(args.num_requests):
            out = engine.generate(input_ids=input_ids_batch[i], sampling_params=ttft_params)
            ttfts.append(out["meta_info"].get("e2e_latency", 0))

        # Timed run — send requests in batch_size chunks
        print(f"  Benchmarking prompt_len={prompt_len}, "
              f"num_requests={args.num_requests}, "
              f"max_new_tokens={args.max_new_tokens} ...")

        batch_outputs = []
        t0 = time.perf_counter()
        for batch_start in range(0, args.num_requests, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.num_requests)
            batch_ids = input_ids_batch[batch_start:batch_end]
            batch_params = [sampling_params] * len(batch_ids)

            if len(batch_ids) == 1:
                out = engine.generate(input_ids=batch_ids[0], sampling_params=sampling_params)
                batch_outputs.append(out)
            else:
                out = engine.generate(input_ids=batch_ids, sampling_params=batch_params)
                batch_outputs.extend(out)
        elapsed = time.perf_counter() - t0

        total_input_tokens = sum(len(ids) for ids in input_ids_batch)
        total_output_tokens = sum(
            o["meta_info"]["completion_tokens"] for o in batch_outputs
        )

        # Per-request details
        for i, out in enumerate(batch_outputs):
            meta = out["meta_info"]
            completion_tokens = meta.get("completion_tokens", 0)
            e2e = meta.get("e2e_latency", elapsed / args.num_requests)
            spec_verify_ct = meta.get("spec_verify_ct", 0)
            accept_length = meta.get("spec_accept_length", 0)
            if accept_length == 0 and spec_verify_ct > 0:
                accept_length = completion_tokens / spec_verify_ct

            # Decode time = e2e minus measured TTFT (prefill)
            ttft = ttfts[i] if i < len(ttfts) else 0
            decode_time = max(e2e - ttft, 1e-6)
            decode_tps = completion_tokens / decode_time

            results.append({
                "phase": phase,
                "prompt_len": prompt_len,
                "request_idx": i,
                "input_tokens": len(input_ids_batch[i]),
                "completion_tokens": completion_tokens,
                "e2e_latency_s": e2e,
                "ttft_s": ttft,
                "decode_time_s": decode_time,
                "tokens_per_sec": completion_tokens / e2e if e2e > 0 else 0,
                "decode_throughput": decode_tps,
                "spec_verify_ct": spec_verify_ct,
                "accept_length": accept_length,
                "eagle_topk": (args.eagle_topk or 4) if phase != "original" else None,
                "speculative_num_steps": (args.speculative_num_steps or 10) if phase != "original" else None,
            })

        # Get server-reported throughput
        try:
            server_info = engine.get_server_info()
            last_gen_tps = server_info["internal_states"][0].get("last_gen_throughput", -1)
        except Exception:
            last_gen_tps = -1

        avg_decode_tps = sum(
            r["decode_throughput"] for r in results if r["prompt_len"] == prompt_len
        ) / max(1, sum(1 for r in results if r["prompt_len"] == prompt_len))
        avg_ttft = sum(
            r["ttft_s"] for r in results if r["prompt_len"] == prompt_len
        ) / max(1, sum(1 for r in results if r["prompt_len"] == prompt_len))

        print(f"    output={total_output_tokens}  "
              f"e2e={elapsed:.2f}s  ttft={avg_ttft:.2f}s  "
              f"e2e_tps={total_output_tokens/elapsed:.1f}  "
              f"decode_tps={avg_decode_tps:.1f}  "
              f"server_gen={last_gen_tps:.1f}")

    engine.shutdown()
    return results


def print_results(all_results, prompt_lengths):
    """Print comparison table across all phases."""
    from collections import defaultdict

    # Group by (phase, prompt_len)
    groups = defaultdict(list)
    for r in all_results:
        groups[(r["phase"], r["prompt_len"])].append(r)

    # Compute aggregates
    agg = {}
    for key, items in groups.items():
        n = len(items)
        total_comp = sum(r["completion_tokens"] for r in items)
        total_e2e = sum(r["e2e_latency_s"] for r in items)
        agg[key] = {
            "n": n,
            "total_comp": total_comp,
            "total_e2e": total_e2e,
            "mean_e2e_tps": sum(r["tokens_per_sec"] for r in items) / n,
            "mean_decode_tps": sum(r["decode_throughput"] for r in items) / n,
            "mean_latency": total_e2e / n,
            "mean_accept": sum(r["accept_length"] for r in items) / n,
            "mean_output_tokens": total_comp / n,
            "eagle_topk": items[0].get("eagle_topk"),
            "speculative_num_steps": items[0].get("speculative_num_steps"),
        }

    prompt_lens = sorted(set(int(x) for x in prompt_lengths))
    phases_present = []
    for p in ["original", "flat", "cascade"]:
        if any(r["phase"] == p for r in all_results):
            phases_present.append(p)

    # Extract shared params from first spec phase
    spec_data = next((agg[k] for k in agg if k[0] in ("flat", "cascade")), None)
    topk_val = spec_data["eagle_topk"] if spec_data else "-"
    depth_val = spec_data["speculative_num_steps"] if spec_data else "-"
    # mean output tokens (use first available phase)
    first_data = next(iter(agg.values()), None)
    out_tok_val = f"{first_data['mean_output_tokens']:.0f}" if first_data else "-"
    n_val = first_data["n"] if first_data else "-"

    print(f"\n{'='*120}")
    print(f"  E2E Benchmark Results")
    print(f"{'='*120}")
    print(f"  topk={topk_val}  depth={depth_val}  "
          f"out_tokens={out_tok_val}  n={n_val}  "
          f"prompt_lengths={','.join(str(p) for p in prompt_lens)}")
    print(f"{'-'*120}")
    print(f"  {'prompt':>7}  {'phase':>9}  {'lat(s)':>8}  "
          f"{'e2e tok/s':>10}  {'dec tok/s':>10}  {'accept':>7}  "
          f"{'dec vs orig':>11}  {'dec vs flat':>11}")
    print(f"  {'-'*7}  {'-'*9}  {'-'*8}  "
          f"{'-'*10}  {'-'*10}  {'-'*7}  "
          f"{'-'*11}  {'-'*11}")

    for pl in prompt_lens:
        orig = agg.get(("original", pl))
        for phase in phases_present:
            data = agg.get((phase, pl))
            if data is None:
                continue
            vs_orig = ""
            vs_flat = ""
            if phase != "original" and orig and orig["mean_decode_tps"] > 0:
                vs_orig = f"{data['mean_decode_tps'] / orig['mean_decode_tps']:.2f}x"
            if phase == "cascade":
                flat = agg.get(("flat", pl))
                if flat and flat["mean_decode_tps"] > 0:
                    vs_flat = f"{data['mean_decode_tps'] / flat['mean_decode_tps']:.2f}x"
            print(f"  {pl:>7}  {phase:>9}  "
                  f"{data['mean_latency']:>8.2f}  {data['mean_e2e_tps']:>10.1f}  "
                  f"{data['mean_decode_tps']:>10.1f}  {data['mean_accept']:>7.2f}  "
                  f"{vs_orig:>11}  {vs_flat:>11}")
        if pl != prompt_lens[-1]:
            print()

    print(f"{'='*120}")


def add_common_args(parser):
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model-path", required=True,
                        help="EAGLE draft model path")
    parser.add_argument("--prompt-lengths", default="2048",
                        help="Comma-separated prompt lengths")
    parser.add_argument("--speculative-algorithm", default="STANDALONE",
                        help="Speculative decoding algorithm (default: STANDALONE)")
    parser.add_argument("--eagle-topk", type=int, default=None,
                        help="EAGLE topk (default: auto-chosen by sglang)")
    parser.add_argument("--speculative-num-steps", type=int, default=None,
                        help="Number of speculative steps (default: auto-chosen by sglang)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--only", choices=["original", "flat", "cascade"],
                        default=None, help="Run only one phase")
    parser.add_argument("--skip-original", action="store_true",
                        help="Skip the original (no speculation) phase")
    parser.add_argument("--result-file", default=None,
                        help="Write JSON results to file")


def run_phase_subprocess(phase: str, argv: list[str]) -> list[dict]:
    """Run a single phase in a fresh subprocess to avoid GPU state leaks."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        result_path = f.name

    cmd = [sys.executable, __file__, "--_run-phase", phase,
           "--_result-path", result_path] + argv
    proc = subprocess.run(cmd, timeout=600)
    if proc.returncode != 0:
        print(f"\n  Phase {phase} failed (exit {proc.returncode})")
        return []

    try:
        with open(result_path) as f:
            return json.load(f)
    finally:
        os.unlink(result_path)


def main():
    # Internal entrypoint: run a single phase in-process
    if "--_run-phase" in sys.argv:
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        parser.add_argument("--_run-phase", required=True)
        parser.add_argument("--_result-path", required=True)
        args = parser.parse_args()
        if args.batch_size > args.num_requests:
            args.num_requests = args.batch_size
        results = run_phase(args, args._run_phase)
        with open(args._result_path, "w") as f:
            json.dump(results, f)
        return

    # Main entrypoint: orchestrate phases as subprocesses
    parser = argparse.ArgumentParser(
        description="E2E benchmark: original vs flat-EAGLE vs cascade-EAGLE"
    )
    add_common_args(parser)
    args = parser.parse_args()

    phases = ["cascade", "flat", "original"]
    if args.only:
        phases = [args.only]
    elif args.skip_original:
        phases = ["cascade", "flat"]

    # Forward all original argv (minus script name) to subprocesses
    forwarded_argv = sys.argv[1:]

    all_results = []
    for phase in phases:
        results = run_phase_subprocess(phase, forwarded_argv)
        all_results.extend(results)

    prompt_lengths = [int(x) for x in args.prompt_lengths.split(",")]
    print_results(all_results, prompt_lengths)

    if args.result_file:
        with open(args.result_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.result_file}")


if __name__ == "__main__":
    main()
