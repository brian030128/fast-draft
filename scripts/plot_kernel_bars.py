"""Grouped bar chart: speedup over Paged attention baseline.

4 subplots (2 models x 2 topk), each full-width, stacked vertically.
Bars show speedup = flat_ms / method_ms for each method.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_MAP = {
    "ml_cascade_ms": "FlashInfer",
    "fasttree_ms": "FastTree",
    "deft_ms": "DeFT",
    "cascade_ms": "Ours",
}
METHODS = list(METHOD_MAP.keys())
LABELS = list(METHOD_MAP.values())

COLORS = {
    "FlashInfer": "#74a9cf",
    "FastTree": "#fd8d3c",
    "DeFT": "#78c679",
    "Ours": "#e41a1c",
}


def load_csv(path):
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def make_plot(csv_groups, output):
    """csv_groups: list of (label, [csv_path, ...])"""
    # Each group becomes a row of 2 subplots (topk=8, topk=16)
    n_rows = len(csv_groups)
    n_methods = len(METHODS)

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.2 * n_rows), squeeze=False)

    for row_idx, (group_label, csv_paths) in enumerate(csv_groups):
        all_rows = []
        for p in csv_paths:
            all_rows.extend(load_csv(p))

        topk_groups = {}
        for row in all_rows:
            tk = int(row["topk"])
            if tk not in topk_groups:
                topk_groups[tk] = []
            topk_groups[tk].append(row)

        for col_idx, tk in enumerate(sorted(topk_groups.keys())[:2]):
            ax = axes[row_idx][col_idx]
            rows = sorted(topk_groups[tk], key=lambda r: int(r["prefix_len"]))
            prefix_lens = [int(r["prefix_len"]) for r in rows]
            x_labels = [f"{pl // 1024}K" if pl >= 1024 else str(pl) for pl in prefix_lens]
            n_groups = len(rows)

            x = np.arange(n_groups)
            width = 0.8 / n_methods
            offsets = np.arange(n_methods) - (n_methods - 1) / 2

            for m_idx, (method_key, label) in enumerate(zip(METHODS, LABELS)):
                speedups = []
                for r in rows:
                    flat = float(r.get("flat_ms", 0) or 0)
                    val = float(r.get(method_key, 0) or 0)
                    speedups.append(flat / val if val > 0 else 0)
                bars = ax.bar(
                    x + offsets[m_idx] * width,
                    speedups,
                    width,
                    label=label,
                    color=COLORS[label],
                    edgecolor="white",
                    linewidth=0.5,
                )

            ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_xlabel("Prefix Length", fontsize=11)
            ax.set_ylabel("Speedup over Paged", fontsize=11)
            ax.set_title(f"{group_label},  Top-$k$ = {tk}", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=9, ncol=n_methods, loc="upper left")
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="figures/kernel_bars.pdf")
    parser.add_argument("--groups", nargs="+", action="append",
                        help="--groups 'Label' file1.csv file2.csv (repeat for each model)")
    args = parser.parse_args()

    csv_groups = []
    for g in args.groups:
        label = g[0]
        paths = g[1:]
        csv_groups.append((label, paths))

    make_plot(csv_groups, args.output)


if __name__ == "__main__":
    main()
