"""Benchmark speculative decoding on a local JSONL dataset.

Runs original / paged (flat) / cascade phases on real prompts from a JSONL file
(each line: {"token_len": N, "prompt": "..."}). Exports per-method CSV files
with consistent row ordering for side-by-side comparison.

CSV naming:
  - Speculative: standalone_{cascade|paged}_{target}_{draft}_top{k}_{depth}_{dataset}.csv
  - Original:    {target}_{dataset}.csv

Usage:
    CUDA_VISIBLE_DEVICES=2,3 uv run python tests/bench_dataset.py \
        --dataset-path data/gov_report.jsonl \
        --model-path meta-llama/Llama-3.1-8B \
        --draft-model-path meta-llama/Llama-3.2-1B \
        --eagle-topk 10 --speculative-num-steps 10 \
        --max-new-tokens 512 --context-length 65000 \
        --mem-fraction-static 0.40 --tp 2 --time-spec
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time


def load_dataset(path, num_samples=None):
    """Load JSONL dataset. Returns list of {"token_len": int, "prompt": str}."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if num_samples and len(records) >= num_samples:
                break
    return records


def create_engine(
    model_path,
    draft_model_path=None,
    speculative_algorithm="STANDALONE",
    eagle_topk=None,
    speculative_num_steps=None,
    speculative_num_draft_tokens=None,
    tp=1,
    context_length=None,
    mem_fraction_static=None,
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
        if eagle_topk is not None or speculative_num_steps is not None:
            kwargs["speculative_eagle_topk"] = eagle_topk or 4
            kwargs["speculative_num_steps"] = speculative_num_steps or 10
            kwargs["speculative_num_draft_tokens"] = speculative_num_draft_tokens or (speculative_num_steps or 10) + 3

    return Engine(**kwargs)


def model_short_name(path):
    """Extract short model name from HuggingFace path (e.g. 'meta-llama/Llama-3.1-8B' -> 'Llama-3.1-8B')."""
    return path.rstrip("/").split("/")[-1]


def dataset_short_name(path):
    """Extract dataset name from file path (e.g. 'data/gov_report.jsonl' -> 'gov_report')."""
    return os.path.splitext(os.path.basename(path))[0]


def temp_tag(temperature):
    """Format temperature for filenames (e.g. 0.0 -> 'temp0', 0.7 -> 'temp0p7')."""
    t = float(temperature)
    if t == 0.0:
        return "temp0"
    return "temp" + f"{t:g}".replace(".", "p")


def csv_filename(phase, args):
    """Generate CSV filename for a given phase."""
    prefix = getattr(args, "result_prefix", "") or ""
    target = model_short_name(args.model_path)
    ds = dataset_short_name(args.dataset_path)
    bs = args.batch_size
    temp = temp_tag(getattr(args, "temperature", 0.0))
    if phase == "original":
        return f"{prefix}{target}_{ds}_{temp}_bs{bs}.csv"
    else:
        method = "cascade" if phase == "cascade" else "cascade_no_cg" if phase == "cascade_no_cg" else "fasttree" if phase == "fasttree" else "paged"
        draft = model_short_name(args.draft_model_path)
        topk = args.eagle_topk or 4
        depth = args.speculative_num_steps or 10
        return f"{prefix}standalone_{method}_{target}_{draft}_top{topk}_{depth}_{ds}_{temp}_bs{bs}.csv"


def run_phase(args, phase, records):
    """Run one benchmark phase on the given records. Returns list of per-prompt result dicts."""
    # Reset all draft env vars
    os.environ["SGLANG_CASCADE_DRAFT"] = "0"
    os.environ["SGLANG_CASCADE_DRAFT_NO_CG"] = "0"
    os.environ.pop("SGLANG_FASTTREE_DRAFT", None)

    if phase == "cascade":
        os.environ["SGLANG_CASCADE_DRAFT"] = "1"
    elif phase == "cascade_no_cg":
        os.environ["SGLANG_CASCADE_DRAFT_NO_CG"] = "1"
    elif phase == "fasttree":
        os.environ["SGLANG_FASTTREE_DRAFT"] = "1"

    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

    if getattr(args, "time_spec", False) and phase != "original":
        os.environ["SGLANG_TIME_SPEC"] = "1"
    else:
        os.environ.pop("SGLANG_TIME_SPEC", None)

    # Clean prometheus state for multi-engine runs
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
    if phase == "original":
        phase_info = "  (no speculation)"
    elif phase == "cascade_no_cg":
        phase_info = "  (SGLANG_CASCADE_DRAFT_NO_CG=1, no CUDA graph for attn)"
    elif phase == "fasttree":
        phase_info = f"  (SGLANG_FASTTREE_DRAFT={os.environ.get('SGLANG_FASTTREE_DRAFT', '0')})"
    else:
        phase_info = f"  (SGLANG_CASCADE_DRAFT={os.environ.get('SGLANG_CASCADE_DRAFT', '0')})"
    print(f"  Phase: {phase.upper()}{phase_info}")
    print(f"{'='*70}")

    engine = create_engine(
        model_path=args.model_path,
        draft_model_path=draft,
        speculative_algorithm=args.speculative_algorithm,
        eagle_topk=topk,
        speculative_num_steps=num_steps,
        speculative_num_draft_tokens=getattr(args, "speculative_num_draft_tokens", None),
        tp=args.tp,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
    )

    # Pre-tokenize all prompts and truncate to fit context_length - max_new_tokens
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    max_prompt_tokens = None
    if args.context_length is not None:
        max_prompt_tokens = args.context_length - args.max_new_tokens - 1

    all_input_ids = []
    truncated = 0
    for rec in records:
        ids = _tokenizer.encode(rec["prompt"])
        if max_prompt_tokens is not None and len(ids) > max_prompt_tokens:
            ids = ids[:max_prompt_tokens]
            truncated += 1
        rec["token_len"] = len(ids)
        all_input_ids.append(ids)
    if truncated:
        print(f"  Truncated {truncated}/{len(records)} prompts to {max_prompt_tokens} tokens")

    temperature = getattr(args, "temperature", 0.0)

    # Warmup
    print("  Warmup ...")
    warmup_ids = all_input_ids[0][:256]
    engine.generate(
        input_ids=warmup_ids,
        sampling_params={"temperature": temperature, "max_new_tokens": 4, "ignore_eos": True},
    )

    results = []
    sampling_params = {
        "temperature": temperature,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": True,
    }

    import random
    bs = args.batch_size
    ttft_params = {"temperature": temperature, "max_new_tokens": 1, "ignore_eos": True}
    for batch_start in range(0, len(records), bs):
        batch_end = min(batch_start + bs, len(records))
        batch_input_ids = [all_input_ids[i] for i in range(batch_start, batch_end)]
        batch_records = [records[i] for i in range(batch_start, batch_end)]
        B = len(batch_input_ids)

        # Measure TTFT (batched). Shuffle each prompt so we don't hit prefix cache.
        ttft_ids = []
        for i, ids in enumerate(batch_input_ids):
            shuffled = list(ids)
            random.Random(10000 + batch_start + i).shuffle(shuffled)
            ttft_ids.append(shuffled)
        t0 = time.perf_counter()
        if B == 1:
            engine.generate(input_ids=ttft_ids[0], sampling_params=ttft_params)
        else:
            engine.generate(input_ids=ttft_ids, sampling_params=[ttft_params] * B)
        ttft_batch = time.perf_counter() - t0
        ttft_per = ttft_batch / B

        # Timed generation (batched)
        t0 = time.perf_counter()
        if B == 1:
            outs = [engine.generate(input_ids=batch_input_ids[0], sampling_params=sampling_params)]
        else:
            outs = engine.generate(input_ids=batch_input_ids, sampling_params=[sampling_params] * B)
        e2e_batch = time.perf_counter() - t0
        e2e_per = e2e_batch / B

        for i, out in enumerate(outs):
            idx = batch_start + i
            token_len = batch_records[i]["token_len"]
            meta = out["meta_info"]
            completion_tokens = meta.get("completion_tokens", 0)
            decode_time = max(e2e_per - ttft_per, 0.01)
            throughput = completion_tokens / decode_time
            draft_time = meta.get("spec_draft_time", 0)
            verify_time = meta.get("spec_verify_time", 0)
            spec_verify_ct = meta.get("spec_verify_ct", 0)
            accept_length = meta.get("spec_accept_length", 0)
            if accept_length == 0 and spec_verify_ct > 0:
                accept_length = completion_tokens / spec_verify_ct

            results.append({
                "prompt_idx": idx,
                "prompt_len": token_len,
                "batch_size": bs,
                "completion_tokens": completion_tokens,
                "e2e_s": e2e_per,
                "ttft_s": ttft_per,
                "decode_time_s": decode_time,
                "draft_time_s": draft_time,
                "verify_time_s": verify_time,
                "throughput": throughput,
                "accept_length": accept_length,
            })

        done = batch_end
        if done % max(10, bs) < bs or done == len(records):
            last = results[-1]
            print(f"  [{done}/{len(records)}] bs={bs} prompt_len={last['prompt_len']}  "
                  f"e2e={last['e2e_s']:.2f}s  ttft={last['ttft_s']:.2f}s  "
                  f"decode={last['decode_time_s']:.2f}s  tps={last['throughput']:.1f}")

    engine.shutdown()
    return results


def write_csv(results, filepath):
    """Write per-prompt results to CSV."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fieldnames = [
        "prompt_idx", "prompt_len", "batch_size", "e2e", "ttft",
        "draft_time", "verify_time", "throughput", "accept_length",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "prompt_idx": r["prompt_idx"],
                "prompt_len": r["prompt_len"],
                "batch_size": r["batch_size"],
                "e2e": f"{r['e2e_s']:.4f}",
                "ttft": f"{r['ttft_s']:.4f}",
                "draft_time": f"{r['draft_time_s']:.4f}",
                "verify_time": f"{r['verify_time_s']:.4f}",
                "throughput": f"{r['throughput']:.2f}",
                "accept_length": f"{r['accept_length']:.2f}",
            })


def print_summary(all_phase_results, args):
    """Print averaged results and speedup comparison across phases."""
    if not all_phase_results:
        print("\n  No results to summarize.")
        return

    # Compute per-phase averages
    phase_avgs = {}
    for phase, results in all_phase_results.items():
        n = len(results)
        phase_avgs[phase] = {
            "n": n,
            "avg_prompt_len": sum(r["prompt_len"] for r in results) / n,
            "avg_throughput": sum(r["throughput"] for r in results) / n,
            "avg_draft_time": sum(r["draft_time_s"] for r in results) / n,
            "avg_verify_time": sum(r["verify_time_s"] for r in results) / n,
            "avg_decode_time": sum(r["decode_time_s"] for r in results) / n,
            "avg_ttft": sum(r["ttft_s"] for r in results) / n,
            "avg_e2e": sum(r["e2e_s"] for r in results) / n,
            "avg_accept": sum(r["accept_length"] for r in results) / n,
        }

    orig_tps = phase_avgs.get("original", {}).get("avg_throughput")
    flat_tps = phase_avgs.get("flat", {}).get("avg_throughput")

    ds_name = dataset_short_name(args.dataset_path)
    target = model_short_name(args.model_path)

    print(f"\n{'='*90}")
    print(f"  Summary: {target} on {ds_name} "
          f"(topk={args.eagle_topk or 4}, depth={args.speculative_num_steps or 10})")
    print(f"{'='*90}")
    print(f"  {'phase':>10}  {'n':>4}  {'avg_prompt':>10}  {'e2e(s)':>8}  {'ttft(s)':>8}  "
          f"{'decode(s)':>10}  {'draft(s)':>9}  {'verify(s)':>10}  "
          f"{'tput':>8}  {'accept':>7}  {'vs orig':>8}  {'vs paged':>9}")
    print(f"  {'-'*10}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*8}  "
          f"{'-'*10}  {'-'*9}  {'-'*10}  "
          f"{'-'*8}  {'-'*7}  {'-'*8}  {'-'*9}")

    for phase in ["original", "flat", "cascade_no_cg", "cascade", "fasttree"]:
        if phase not in phase_avgs:
            continue
        a = phase_avgs[phase]
        label = "paged" if phase == "flat" else phase

        vs_orig = ""
        if orig_tps and phase != "original":
            vs_orig = f"{a['avg_throughput'] / orig_tps:.2f}x"
        vs_flat = ""
        if flat_tps and phase in ("cascade", "cascade_no_cg", "fasttree"):
            vs_flat = f"{a['avg_throughput'] / flat_tps:.2f}x"

        draft_str = f"{a['avg_draft_time']:.3f}" if a['avg_draft_time'] > 0 else "-"
        verify_str = f"{a['avg_verify_time']:.3f}" if a['avg_verify_time'] > 0 else "-"
        accept_str = f"{a['avg_accept']:.2f}" if a['avg_accept'] > 0 else "-"

        print(f"  {label:>10}  {a['n']:>4}  {a['avg_prompt_len']:>10.0f}  "
              f"{a['avg_e2e']:>8.2f}  {a['avg_ttft']:>8.2f}  "
              f"{a['avg_decode_time']:>10.2f}  "
              f"{draft_str:>9}  {verify_str:>10}  "
              f"{a['avg_throughput']:>8.1f}  {accept_str:>7}  "
              f"{vs_orig:>8}  {vs_flat:>9}")

    print(f"{'='*90}")

    # CSV file listing
    if not getattr(args, "no_save", False):
        print(f"\n  CSV files in {args.result_dir}/:")
        for phase in all_phase_results:
            fname = csv_filename(phase, args)
            fpath = os.path.join(args.result_dir, fname)
            if os.path.exists(fpath):
                print(f"    {fname}")
        print()


def run_phase_subprocess(phase, argv):
    """Run a single phase in a fresh subprocess to avoid GPU state leaks."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        result_path = f.name

    cmd = [sys.executable, __file__, "--_run-phase", phase,
           "--_result-path", result_path] + argv
    proc = subprocess.run(cmd, timeout=86400)
    if proc.returncode != 0:
        print(f"\n  Phase {phase} failed (exit {proc.returncode})")
        return []

    try:
        with open(result_path) as f:
            return json.load(f)
    finally:
        os.unlink(result_path)


def add_common_args(parser):
    parser.add_argument("--dataset-path", required=True,
                        help="Path to local JSONL dataset file")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model-path", required=True,
                        help="EAGLE draft model path")
    parser.add_argument("--speculative-algorithm", default="STANDALONE")
    parser.add_argument("--eagle-topk", type=int, default=None)
    parser.add_argument("--speculative-num-steps", type=int, default=None)
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=None,
                        help="Verify budget (default: speculative_num_steps + 3)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of prompts to submit per engine.generate() call")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--only", choices=["original", "flat", "cascade", "cascade_no_cg", "fasttree"],
                        default=None, help="Run only one phase")
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--skip", action="append", default=[],
                        choices=["original", "flat", "cascade", "cascade_no_cg", "fasttree"],
                        help="Skip one or more phases (repeatable, e.g. --skip cascade_no_cg --skip flat)")
    parser.add_argument("--time-spec", action="store_true",
                        help="Enable draft/verify timing (adds sync overhead)")
    parser.add_argument("--result-dir", default="results",
                        help="Directory for CSV output (default: results/)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit number of prompts from the dataset")
    parser.add_argument("--result-prefix", required=True,
                        help="Prefix for CSV filenames (e.g. 'H100_')")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't write CSV output files")


def main():
    # Internal entrypoint: run a single phase in-process
    if "--_run-phase" in sys.argv:
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        parser.add_argument("--_run-phase", required=True)
        parser.add_argument("--_result-path", required=True)
        parser.add_argument("--_shared-ttfts", default=None)
        args = parser.parse_args()

        if args._shared_ttfts:
            args._shared_ttfts = {int(k): v for k, v in json.loads(args._shared_ttfts).items()}
        else:
            args._shared_ttfts = {}

        records = load_dataset(args.dataset_path, args.num_samples)
        results = run_phase(args, args._run_phase, records)

        with open(args._result_path, "w") as f:
            json.dump(results, f)
        return

    # Main entrypoint: orchestrate phases as subprocesses
    parser = argparse.ArgumentParser(
        description="Benchmark speculative decoding on a local JSONL dataset"
    )
    add_common_args(parser)
    args = parser.parse_args()

    phases = ["original", "flat", "cascade_no_cg", "cascade", "fasttree"]
    if args.only:
        phases = [args.only]
    elif args.skip_original:
        phases = ["flat", "cascade_no_cg", "cascade", "fasttree"]
    if args.skip:
        phases = [p for p in phases if p not in args.skip]

    forwarded_argv = sys.argv[1:]

    all_phase_results = {}  # phase -> list of result dicts
    for phase in phases:
        results = run_phase_subprocess(phase, forwarded_argv)

        if not results:
            continue

        all_phase_results[phase] = results

        # Save CSV
        if not args.no_save:
            fname = csv_filename(phase, args)
            fpath = os.path.join(args.result_dir, fname)
            write_csv(results, fpath)
            print(f"  Results saved to {fpath}")

    # Print summary
    print_summary(all_phase_results, args)


if __name__ == "__main__":
    main()
