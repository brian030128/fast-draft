"""Correctness test: verify original, flat-EAGLE, and cascade-EAGLE produce identical outputs.

Runs all three backends with temperature=0 (greedy) on the same random-token
prompts and asserts that the generated text is identical.  Each phase runs in
a subprocess (like bench_e2e.py) to avoid GPU state leaks between engines.

Usage:
    CUDA_VISIBLE_DEVICES=2,3 uv run python tests/sglang_correctness_test.py \
        --model-path meta-llama/Llama-3.1-8B \
        --draft-model-path meta-llama/Llama-3.2-1B \
        --prompt-lengths 512 \
        --eagle-topk 4 \
        --max-new-tokens 64 \
        --num-requests 2 \
        --context-length 4096 \
        --mem-fraction-static 0.40 \
        --batch-size 1 \
        --tp 2
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile


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
    }
    if context_length is not None:
        kwargs["context_length"] = context_length
    if mem_fraction_static is not None:
        kwargs["mem_fraction_static"] = mem_fraction_static

    if draft_model_path is not None:
        kwargs["speculative_algorithm"] = speculative_algorithm
        kwargs["speculative_draft_model_path"] = draft_model_path
        if eagle_topk is not None or speculative_num_steps is not None:
            kwargs["speculative_eagle_topk"] = eagle_topk or 4
            kwargs["speculative_num_steps"] = speculative_num_steps or 10
            kwargs["speculative_num_draft_tokens"] = (speculative_num_steps or 10) + 3

    return Engine(**kwargs)


def run_phase(args, phase: str):
    """Run one phase and return list of output dicts with text and token info."""
    if phase == "cascade":
        os.environ["SGLANG_CASCADE_DRAFT"] = "1"
    else:
        os.environ["SGLANG_CASCADE_DRAFT"] = "0"

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

    print(f"\n{'='*60}")
    print(f"  Phase: {phase.upper()}")
    print(f"{'='*60}")

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
    sampling_params = {
        "temperature": 0,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": True,
    }

    results = []
    for prompt_len in prompt_lengths:
        input_ids_batch = [
            make_random_input_ids(prompt_len, seed=42 + i)
            for i in range(args.num_requests)
        ]

        for i, input_ids in enumerate(input_ids_batch):
            out = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
            results.append({
                "phase": phase,
                "prompt_len": prompt_len,
                "request_idx": i,
                "text": out["text"],
                "completion_tokens": out["meta_info"]["completion_tokens"],
            })
            print(f"  prompt_len={prompt_len} req={i}: "
                  f"{out['meta_info']['completion_tokens']} tokens generated")

    engine.shutdown()
    return results


def run_phase_subprocess(phase: str, argv: list[str]) -> list[dict]:
    """Run a single phase in a fresh subprocess to avoid GPU state leaks."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        result_path = f.name

    cmd = [sys.executable, __file__, "--_run-phase", phase,
           "--_result-path", result_path] + argv
    proc = subprocess.run(cmd, timeout=600)
    if proc.returncode != 0:
        print(f"\n  Phase {phase} FAILED (exit {proc.returncode})")
        return []

    try:
        with open(result_path) as f:
            return json.load(f)
    finally:
        os.unlink(result_path)


def compare_outputs(all_results):
    """Compare outputs across phases and report mismatches."""
    from collections import defaultdict

    # Group by (prompt_len, request_idx)
    by_key = defaultdict(dict)
    for r in all_results:
        key = (r["prompt_len"], r["request_idx"])
        by_key[key][r["phase"]] = r

    all_match = True
    for key in sorted(by_key):
        phases = by_key[key]
        prompt_len, req_idx = key

        phase_names = sorted(phases.keys())
        if len(phase_names) < 2:
            continue

        reference_phase = "original" if "original" in phases else phase_names[0]
        reference_text = phases[reference_phase]["text"]
        reference_tokens = phases[reference_phase]["completion_tokens"]

        for phase in phase_names:
            if phase == reference_phase:
                continue
            other_text = phases[phase]["text"]
            other_tokens = phases[phase]["completion_tokens"]

            if other_text != reference_text:
                all_match = False
                # Find first divergence point
                min_len = min(len(reference_text), len(other_text))
                diff_pos = min_len
                for j in range(min_len):
                    if reference_text[j] != other_text[j]:
                        diff_pos = j
                        break

                print(f"\n  MISMATCH: prompt_len={prompt_len} req={req_idx} "
                      f"| {reference_phase} vs {phase}")
                print(f"    {reference_phase}: {reference_tokens} tokens, "
                      f"{phase}: {other_tokens} tokens")
                print(f"    First diff at char {diff_pos}")
                print(f"    {reference_phase}[{diff_pos}:{diff_pos+80}]: "
                      f"{repr(reference_text[diff_pos:diff_pos+80])}")
                print(f"    {phase}[{diff_pos}:{diff_pos+80}]: "
                      f"{repr(other_text[diff_pos:diff_pos+80])}")
            else:
                print(f"  OK: prompt_len={prompt_len} req={req_idx} "
                      f"| {reference_phase} == {phase} "
                      f"({reference_tokens} tokens)")

    return all_match


def add_common_args(parser):
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--prompt-lengths", default="512",
                        help="Comma-separated prompt lengths")
    parser.add_argument("--speculative-algorithm", default="STANDALONE")
    parser.add_argument("--eagle-topk", type=int, default=None)
    parser.add_argument("--speculative-num-steps", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--only", nargs="+",
                        choices=["original", "flat", "cascade"],
                        default=None, help="Run only specified phases")
    parser.add_argument("--result-file", default=None,
                        help="Write JSON results to file")


def main():
    # Internal entrypoint: run a single phase in-process
    if "--_run-phase" in sys.argv:
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        parser.add_argument("--_run-phase", required=True)
        parser.add_argument("--_result-path", required=True)
        args = parser.parse_args()
        results = run_phase(args, args._run_phase)
        with open(args._result_path, "w") as f:
            json.dump(results, f)
        return

    # Main entrypoint
    parser = argparse.ArgumentParser(
        description="Correctness test: verify original, flat, and cascade produce identical outputs"
    )
    add_common_args(parser)
    args = parser.parse_args()

    phases = args.only or ["original", "flat", "cascade"]
    forwarded_argv = sys.argv[1:]

    all_results = []
    for phase in phases:
        results = run_phase_subprocess(phase, forwarded_argv)
        if not results:
            print(f"\n  Phase {phase} produced no results, aborting.")
            sys.exit(1)
        all_results.extend(results)

    print(f"\n{'='*60}")
    print(f"  Correctness Comparison")
    print(f"{'='*60}")

    all_match = compare_outputs(all_results)

    if args.result_file:
        with open(args.result_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.result_file}")

    print()
    if all_match:
        print("  ALL OUTPUTS MATCH")
    else:
        print("  SOME OUTPUTS DIFFER")
        sys.exit(1)


if __name__ == "__main__":
    main()
