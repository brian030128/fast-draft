"""Tune the Fast Draft cascade kernel for draft-model shapes on A6000.

Knobs (see 3rdparty/flashinfer/include/flashinfer/attention/cascade_tune.cuh),
set per process through FLASHINFER_CASCADE_TUNE="<tile_q1>,<ctas_per_sm>,<min_chunk_kv>":
  tile_q1      query tile of the shared-prefix runner (64 or 128; draft decode has
               topk * GQA = 40 packed query rows)
  ctas_per_sm  CTAs per SM the scheduler plans for
  min_chunk_kv minimum KV chunk count is ceil(max_kv / min_chunk_kv)

    # 1. compile every variant on CPU (no GPU needed), sm_86
    uv run python tests/tune_cascade_a6000.py build --jobs 4
    # 2. time every variant on an idle A6000
    CUDA_VISIBLE_DEVICES=<idle A6000> uv run python tests/tune_cascade_a6000.py bench

bench times the E2E path (plan_for_draft once + update_draft_step per step) over
16 per-layer KV pools so K/V comes from HBM, for all draft steps, and checks the
output against stock FlashInfer MultiLevelCascade.
"""

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time

TILES = [64, 128]
CTAS = [2, 3, 4]
CHUNKS = [512, 1024, 2048, 4096]
DEFAULT = (128, 2, 1024)  # current A6000 tuning (bdc2db50)
# --grid batch: tile 64 (the bs=1 winner) with fewer CTAs and larger chunk caps
BATCH_CTAS = [1, 2, 3, 4]
BATCH_CHUNKS = [1024, 2048, 4096, 8192, 16384, 65536]
BATCH_CASES = "1:4096,1:16384,1:32768,1:65536,1:97404,4:4096,4:16384,4:32768," \
              "8:4096,8:16384,8:32768,16:4096,16:16384,16:32768,32:4096,32:16384,32:32768"


def configs(grid="bs1"):
    if grid == "batch":
        return [DEFAULT] + list(itertools.product([64], BATCH_CTAS, BATCH_CHUNKS))
    return list(itertools.product(TILES, CTAS, CHUNKS))


def tune_str(c):
    return ",".join(str(x) for x in c)


# ----------------------------------------------------------------------
# build
# ----------------------------------------------------------------------

def build_one(tune):
    """Compile the module for one tuning (runs in its own process)."""
    import torch
    from flashinfer.jit.attention.modules import gen_batch_attention_module
    from flashinfer.utils import PosEncodingMode

    if tune:
        os.environ["FLASHINFER_CASCADE_TUNE"] = tune
    else:
        os.environ.pop("FLASHINFER_CASCADE_TUNE", None)
    spec = gen_batch_attention_module(
        torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32,
        64, 64, PosEncodingMode["NONE"].value, False, False,
    )
    t0 = time.time()
    if not spec.is_compiled:
        spec.build(verbose=False)
    print(f"built {tune or 'default'} in {time.time() - t0:.0f}s -> {spec.name}", flush=True)


def cmd_build(args):
    env = dict(os.environ, FLASHINFER_CUDA_ARCH_LIST="8.6", CUDA_VISIBLE_DEVICES="",
               MAX_JOBS=str(args.ninja_jobs))
    todo = [None] + [tune_str(c) for c in configs(args.grid)]
    running = []
    for tune in todo:
        while len(running) >= args.jobs:
            running = [p for p in running if p.poll() is None]
            time.sleep(2)
        cmd = [sys.executable, __file__, "_build_one", tune or ""]
        running.append(subprocess.Popen(cmd, env=env))
    for p in running:
        p.wait()


# ----------------------------------------------------------------------
# bench
# ----------------------------------------------------------------------

def bench_one(cases, iters):
    """Time Ours for the current FLASHINFER_CASCADE_TUNE; print one JSON line.

    cases: list of (batch_size, prefix_len); every request has topk branches.
    """
    import torch
    from flashinfer.attention import CascadeBatchAttentionWrapper
    from flashinfer.cascade import MultiLevelCascadeAttentionWrapper

    sys.path.insert(0, os.path.dirname(__file__))
    from profile_draft_kernels import graph_time_us
    from profile_query_padding import host_indptrs, make_batch

    dtype = torch.bfloat16
    topk, D, hq, hkv, d, layers = 10, 6, 16, 4, 64, 16
    common = dict(num_qo_heads=hq, num_kv_heads=hkv, head_dim_qk=d, head_dim_vo=d,
                  page_size=1, causal=False, sm_scale=None, q_data_type=dtype,
                  kv_data_type=dtype)
    fi_ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    result = {"tune": os.environ.get("FLASHINFER_CASCADE_TUNE", tune_str(DEFAULT)),
              "attn_ms": {}, "max_err": 0.0}
    for bs, L in cases:
        shared, unique, kvs = make_batch(bs, L, topk, D, hkv, d, layers)
        nq = bs * topk
        q = torch.randn(nq, hq, d, device="cuda", dtype=dtype)
        ours = CascadeBatchAttentionWrapper(2, "NHD", "cuda")
        h = host_indptrs(bs, L, topk, D)
        for first in (True, False):
            ours.plan_for_draft(
                max_draft_depth=D, first_call=first,
                qo_indptr_host_arr=[h["qo0"], h["qo1"]],
                kv_indptr_host_arr=[h["kv0"], h["kv1"]],
                kv_indices_arr=[shared, unique.contiguous().view(-1)],
                kv_len_host_arr=[h["len0"], h["len1"]], **common)
        fi = MultiLevelCascadeAttentionWrapper(2, fi_ws, "NHD")
        total_us = 0.0
        for s in range(1, D + 1):
            ours.update_draft_step(step_kv_len=s + 1, step_kv_end=s)
            hs = host_indptrs(bs, L, topk, s)
            fi.plan([hs["qo0"], hs["qo1"]], [hs["kv0"], hs["kv1"]],
                    [shared, unique[:, :s].contiguous().view(-1)],
                    [torch.ones(bs, dtype=torch.int32), torch.ones(nq, dtype=torch.int32)],
                    num_qo_heads=hq, num_kv_heads=hkv, head_dim=d, page_size=1,
                    causal=False, q_data_type=dtype, kv_data_type=dtype)
            o_ours, _ = ours.run(q, kvs[0])
            err = (o_ours.float() - fi.run(q, kvs[0]).float()).abs().max().item()
            result["max_err"] = max(result["max_err"], err)
            total_us += graph_time_us(lambda: [ours.run(q, kv) for kv in kvs], iters)
        result["attn_ms"][str(L) if bs == 1 and cases_bs1_only(cases) else f"{bs}:{L}"] = total_us / 1e3  # per draft iteration, all layers
        del kvs, ours, fi
        torch.cuda.empty_cache()
    print("RESULT " + json.dumps(result), flush=True)


def cases_bs1_only(cases):
    """Keep the bs=1 sweep's original JSON keys (prefix length only)."""
    return all(bs == 1 for bs, _ in cases)


def parse_cases(text):
    """"16384,32768" (bs=1) or "bs:len,bs:len"."""
    return [tuple(map(int, c.split(":"))) if ":" in c else (1, int(c)) for c in text.split(",")]


def cmd_bench(args):
    rows = []
    cases = args.cases or (BATCH_CASES if args.grid == "batch" else args.prefix_lens)
    todo = [None] + [tune_str(c) for c in configs(args.grid) if c != DEFAULT]
    for tune in todo:
        env = dict(os.environ)
        if tune:
            env["FLASHINFER_CASCADE_TUNE"] = tune
        else:
            env.pop("FLASHINFER_CASCADE_TUNE", None)
        cmd = [sys.executable, __file__, "_bench_one", cases, str(args.iters)]
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        if out.returncode != 0 or not line:
            print(f"{tune or 'default'}: FAILED\n{out.stderr[-2000:]}", flush=True)
            continue
        r = json.loads(line[0][len("RESULT "):])
        rows.append(r)
        print(f"{r['tune']:>14}  " + "  ".join(f"{v:7.2f}" for v in r["attn_ms"].values())
              + f"  err {r['max_err']:.1e}", flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")

    if not rows:
        return
    lens = list(rows[0]["attn_ms"])
    base = next(r for r in rows if r["tune"] == tune_str(DEFAULT))
    print(f"\nAttention ms per draft iteration (16 layers x 6 steps); default = {tune_str(DEFAULT)}")
    print("ranked by geometric-mean speedup over default across cases")
    print(f"{'tune':>14} " + " ".join(f"{L:>8}" for L in lens) + f" {'geomean':>8}")
    gm = lambda r: math.exp(sum(math.log(base["attn_ms"][L] / r["attn_ms"][L]) for L in lens) / len(lens))
    for r in sorted(rows, key=gm, reverse=True):
        print(f"{r['tune']:>14} " + " ".join(f"{r['attn_ms'][L]:8.2f}" for L in lens)
              + f" {gm(r):7.2f}x")
    print("\nbest tune per case (speedup vs default):")
    for L in lens:
        b = min(rows, key=lambda r: r["attn_ms"][L])
        print(f"  {L:>9}: {b['tune']:>12}  {b['attn_ms'][L]:8.2f} ms  "
              f"{base['attn_ms'][L] / b['attn_ms'][L]:.2f}x")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_build_one":
        build_one(sys.argv[2] or None)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "_bench_one":
        bench_one(parse_cases(sys.argv[2]), int(sys.argv[3]))
        return
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--jobs", type=int, default=4, help="variants compiled concurrently")
    b.add_argument("--grid", choices=["bs1", "batch"], default="bs1")
    b.add_argument("--ninja-jobs", type=int, default=4, help="nvcc jobs per variant")
    r = sub.add_parser("bench")
    r.add_argument("--grid", choices=["bs1", "batch"], default="bs1")
    r.add_argument("--prefix-lens", default="16384,32768,52203,65536,97404")
    r.add_argument("--cases", default=None, help='"bs:len,..." (default depends on --grid)')
    r.add_argument("--iters", type=int, default=20)
    r.add_argument("--out", default="logs/tune_cascade_a6000.jsonl")
    args = ap.parse_args()
    {"build": cmd_build, "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    main()
