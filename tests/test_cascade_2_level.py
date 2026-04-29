"""
CUDA Graph benchmark: Multi-Level Cascade Attention vs Forced Two-Level Cascade
across draft tree topologies (best / worst / random) and depths.

Two approaches compared:
  - Multi-Level: dynamic num_levels based on tree branching structure
  - Forced Two-Level: always num_levels=2 (prompt + flattened draft nodes)

Additional modes (stem, stress) also compare against Fused (CascadeBatchAttentionWrapper).

Usage:
    python tests/test_cascade_2_level.py sweep --plot
    python tests/test_cascade_2_level.py stem --plot
    python tests/test_cascade_2_level.py stress --plot
    python tests/test_cascade_2_level.py batch --plot

"""

import argparse
import sys
import os

import torch

# Ensure tests/ is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from draft_tree_modes import (
    batch_sweep_test,
    run_benchmark,
    stress_test,
    sweep_stem_length,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Multi-Level vs Forced Two-Level cascade attention "
                    "for speculative decoding draft trees."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Sweep mode
    sweep_parser = subparsers.add_parser("sweep", help="Sweep depths × widths × topologies")
    sweep_parser.add_argument("--depths", type=int, nargs="+",
                              default=[10, 12, 14, 16, 32])
    sweep_parser.add_argument("--widths", type=int, nargs="+",
                              default=[4, 6, 8, 10, 12, 14, 16])
    sweep_parser.add_argument("--prompt_len", type=int, default=32768)
    sweep_parser.add_argument("--random_n", type=int, default=50)
    sweep_parser.add_argument("--warmup", type=int, default=50)
    sweep_parser.add_argument("--repeat", type=int, default=200)
    sweep_parser.add_argument("--plot", action="store_true")

    # Stem sweep mode
    stem_parser = subparsers.add_parser("stem", help="Sweep stem length to find Two-Level crossover")
    stem_parser.add_argument("--budgets", type=int, nargs="+",
                             default=[64, 128, 256],
                             help="Total node budgets to test")
    stem_parser.add_argument("--prompt_len", type=int, default=1024)
    stem_parser.add_argument("--warmup", type=int, default=50)
    stem_parser.add_argument("--repeat", type=int, default=200)
    stem_parser.add_argument("--plot", action="store_true")

    # Stress test mode
    stress_parser = subparsers.add_parser("stress",
        help="Stress test: push budget to extremes (3-way: Multi vs Two vs Fused)")
    stress_parser.add_argument("--budgets", type=int, nargs="+",
                               default=[128, 256, 512, 1024, 2048, 4096, 8192])
    stress_parser.add_argument("--page_sizes", type=int, nargs="+",
                               default=[1, 4, 16])
    stress_parser.add_argument("--prompt_len", type=int, default=1024)
    stress_parser.add_argument("--warmup", type=int, default=50)
    stress_parser.add_argument("--repeat", type=int, default=200)
    stress_parser.add_argument("--plot", action="store_true")

    # Batch sweep mode
    batch_parser = subparsers.add_parser("batch",
        help="Sweep batch size to test L2 cache pollution")
    batch_parser.add_argument("--batch_sizes", type=int, nargs="+",
                              default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    batch_parser.add_argument("--stem", type=int, default=64)
    batch_parser.add_argument("--budget", type=int, default=128)
    batch_parser.add_argument("--prompt_len", type=int, default=1024)
    batch_parser.add_argument("--page_size", type=int, default=1)
    batch_parser.add_argument("--warmup", type=int, default=50)
    batch_parser.add_argument("--repeat", type=int, default=200)
    batch_parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    # Default to sweep if no subcommand given
    if args.command is None or args.command == "sweep":
        run_benchmark(
            depths=getattr(args, "depths", [10, 12, 14, 16]),
            widths=getattr(args, "widths", [4, 6, 8, 10, 12, 14, 16]),
            prompt_len=getattr(args, "prompt_len", 4096),
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            page_size=1,
            dtype=torch.bfloat16,
            random_n=getattr(args, "random_n", 50),
            warmup=getattr(args, "warmup", 50),
            repeat=getattr(args, "repeat", 200),
            do_plot=getattr(args, "plot", False),
        )
    elif args.command == "stem":
        sweep_stem_length(
            budgets=args.budgets,
            prompt_len=args.prompt_len,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            page_size=1,
            dtype=torch.bfloat16,
            warmup=args.warmup,
            repeat=args.repeat,
            do_plot=args.plot,
        )
    elif args.command == "stress":
        stress_test(
            budgets=args.budgets,
            page_sizes=args.page_sizes,
            prompt_len=args.prompt_len,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            dtype=torch.bfloat16,
            warmup=args.warmup,
            repeat=args.repeat,
            do_plot=args.plot,
        )
    elif args.command == "batch":
        batch_sweep_test(
            batch_sizes=args.batch_sizes,
            stem_length=args.stem,
            budget=args.budget,
            prompt_len=args.prompt_len,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            page_size=args.page_size,
            dtype=torch.bfloat16,
            warmup=args.warmup,
            repeat=args.repeat,
            do_plot=args.plot,
        )
