"""Torch-profile one decode batch in SGLang and break down GPU kernel time.

Profiles a batch of short prompts with SGLang's built-in profiler (one trace per
TP rank), then reports for TP rank 0: wall span of the profiled window, GPU-busy
time (union of kernel intervals), and the kernels with the most GPU time.

    CUDA_VISIBLE_DEVICES=<2 idle gpus> SGLANG_CASCADE_PACK_SUFFIX=1 \
        uv run python tests/profile_e2e_draft.py --method cascade --out results/e2e_profile/pack1
"""

import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))


def analyze(trace_path, top=15):
    with gzip.open(trace_path) if trace_path.endswith(".gz") else open(trace_path) as f:
        events = json.load(f)["traceEvents"]
    kern = [e for e in events if e.get("cat") == "kernel" and "dur" in e]
    if not kern:
        print(f"{trace_path}: no kernel events")
        return
    t0 = min(e["ts"] for e in kern)
    t1 = max(e["ts"] + e["dur"] for e in kern)
    ivs = sorted((e["ts"], e["ts"] + e["dur"]) for e in kern)
    busy, cs, ce = 0.0, *ivs[0]
    for s, e in ivs[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    per = defaultdict(lambda: [0, 0.0])
    for e in kern:
        name = e["name"].split("<")[0].split("(")[0].split("::")[-1][:60]
        per[name][0] += 1
        per[name][1] += e["dur"]
    total = sum(v[1] for v in per.values())
    print(f"{os.path.basename(trace_path)}: span {(t1 - t0) / 1e3:.1f} ms, GPU busy "
          f"{busy / 1e3:.1f} ms ({busy / (t1 - t0):.0%}), kernel time {total / 1e3:.1f} ms")
    for name, (n, d) in sorted(per.items(), key=lambda kv: -kv[1][1])[:top]:
        print(f"   {d / 1e3:8.2f} ms  {d / total:5.1%}  n={n:6}  {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["cascade", "paged", "fi_cascade"], required=True,
                    help="draft attention backend (same env switches as bench_dataset.py)")
    ap.add_argument("--prefix-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    out = os.path.abspath(args.out)

    if not args.analyze_only:
        os.environ["SGLANG_CASCADE_DRAFT"] = "1" if args.method == "cascade" else "0"
        os.environ["SGLANG_FI_CASCADE_DRAFT"] = "1" if args.method == "fi_cascade" else "0"
        os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        from bench_dataset import create_engine, load_dataset
        from transformers import AutoTokenizer

        os.makedirs(out, exist_ok=True)
        engine = create_engine(
            "meta-llama/Llama-3.1-8B", draft_model_path="meta-llama/Llama-3.2-1B",
            eagle_topk=10, speculative_num_steps=7, speculative_num_draft_tokens=16, tp=2,
            context_length=args.prefix_tokens + args.max_new_tokens + 64,
            mem_fraction_static=0.6, cuda_graph_max_bs=max(args.batch_size, 32),
        )
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        recs = load_dataset("data/narrativeqa.jsonl", 2 * args.batch_size)
        ids = [tok.encode(r["prompt"])[: args.prefix_tokens] for r in recs]
        sp = {"temperature": 0, "max_new_tokens": args.max_new_tokens, "ignore_eos": True}
        bs = args.batch_size
        engine.generate(input_ids=ids[bs:], sampling_params=[sp] * bs)  # warm up
        engine.start_profile(output_dir=out, activities=["GPU"], with_stack=False,
                             record_shapes=False)
        engine.generate(input_ids=ids[:bs], sampling_params=[sp] * bs)
        engine.stop_profile()
        engine.shutdown()

    for path in sorted(glob.glob(os.path.join(out, "*TP-0*.trace.json*"))):
        analyze(path)


if __name__ == "__main__":
    main()
