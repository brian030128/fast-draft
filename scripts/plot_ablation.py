"""Draft-phase ablation figure (paper/ablation_study.svg).

NarrativeQA, batch size 4, Llama-3.1-8B target / Llama-3.2-1B draft, A6000 x2.
Mean draft time per prompt for each step, as draft speedup over SGLang:
  SGLang                       paged draft backend, CUDA graphs (paper's baseline run)
  + Our Kernel                 cascade_per_step_no_cg
  + Cuda Graph                 cascade_per_step
  + Reuse Plan (Ours)          cascade

    uv run python scripts/plot_ablation.py [--result-dir results/repro_newkernel]
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# SGLang draft time at B=4 (paper Table tab:batch-e2e, A6000 x2); not re-run.
SGLANG_DRAFT_S = 8.72
CSV = "A6000_standalone_{m}_Llama-3.1-8B_Llama-3.2-1B_top10_7_narrativeqa_temp0_bs4.csv"
STEPS = [
    ("SGLang", None),
    ("+ Our\nKernel", "cascade_per_step_no_cg"),
    ("+ Cuda\nGraph", "cascade_per_step"),
    ("+ Reuse\nPlan\n(Ours)", "cascade"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="results/repro_newkernel")
    ap.add_argument("--out", default="paper/ablation_study.svg")
    args = ap.parse_args()

    draft = [SGLANG_DRAFT_S] + [
        pd.read_csv(os.path.join(args.result_dir, CSV.format(m=m))).draft_time.mean()
        for _, m in STEPS[1:]
    ]
    speedup = [SGLANG_DRAFT_S / t for t in draft]
    for (label, _), t, s in zip(STEPS, draft, speedup):
        print(f"{label.replace(chr(10), ' '):24} draft {t:.2f} s  speedup {s:.2f}x")

    fig, ax = plt.subplots(figsize=(7.3, 2.89))
    y = list(range(len(STEPS)))[::-1]
    colors = plt.cm.Blues([0.3, 0.45, 0.6, 0.85])
    ax.barh(y, speedup, height=0.82, color=colors)
    for i in range(len(STEPS) - 1):
        x0, x1 = speedup[i], speedup[i + 1]
        yy = y[i] - 0.18
        ax.annotate("", xy=(x1, yy), xytext=(x0, yy),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color="black", mutation_scale=18))
        ax.text((x0 + x1) / 2, yy + 0.12, f"{x1 / x0:.2f}x", ha="center", va="bottom", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in STEPS], fontsize=13)
    ax.set_xlabel("Draft Speedup", fontsize=14)
    ax.set_xlim(0, max(speedup) * 1.05)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.5, len(STEPS) - 0.5)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
