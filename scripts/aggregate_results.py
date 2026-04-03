"""Aggregate benchmark results across methods (AR, cascade, paged) and datasets."""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Known datasets (used to split dataset name from model name in AR filenames)
DATASETS = {"gov_report", "narrativeqa", "pg19"}

# Regex for speculative decoding filenames:
#   standalone_{method}_{target}_{draft}_top{k}_{steps}_{dataset}.csv
SPEC_RE = re.compile(
    r"^standalone_(cascade|paged)_(.+?)_(.+?)_top(\d+)_(\d+)_(.+)\.csv$"
)


def parse_filename(name: str) -> dict | None:
    """Return metadata dict or None if the file doesn't match any pattern."""
    m = SPEC_RE.match(name)
    if m:
        return {
            "method": m.group(1),
            "target_model": m.group(2),
            "draft_model": m.group(3),
            "topk": int(m.group(4)),
            "steps": int(m.group(5)),
            "dataset": m.group(6),
        }
    # AR pattern: {target_model}_{dataset}.csv
    stem = name.removesuffix(".csv")
    for ds in DATASETS:
        if stem.endswith(f"_{ds}"):
            target = stem.removesuffix(f"_{ds}")
            return {
                "method": "ar",
                "target_model": target,
                "draft_model": None,
                "topk": None,
                "steps": None,
                "dataset": ds,
            }
    return None


def load_results(results_dir: Path) -> dict[tuple[str, str, int | None], dict[str, pd.DataFrame]]:
    """Load CSVs grouped as {(dataset, target_model, topk): {method: DataFrame}}.

    AR results (topk=None) are duplicated into every topk group for the same
    (dataset, target_model) so that speedup-vs-AR comparisons work per topk.
    """
    grouped = defaultdict(dict)
    ar_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    topks_by_key: dict[tuple[str, str], set[int]] = defaultdict(set)
    draft_models: dict[tuple[str, str, int], str] = {}

    for f in sorted(results_dir.glob("*.csv")):
        meta = parse_filename(f.name)
        if meta is None:
            continue
        df = pd.read_csv(f)
        df["decode_time"] = df["e2e"] - df["ttft"]
        ds_target = (meta["dataset"], meta["target_model"])
        if meta["topk"] is None:
            ar_by_key[ds_target] = df
        else:
            key = (meta["dataset"], meta["target_model"], meta["topk"])
            grouped[key][meta["method"]] = df
            topks_by_key[ds_target].add(meta["topk"])
            if meta["draft_model"]:
                draft_models[(meta["dataset"], meta["target_model"], meta["topk"])] = meta["draft_model"]

    # Inject AR into each topk group
    for (dataset, target), df in ar_by_key.items():
        topks = topks_by_key.get((dataset, target))
        if topks:
            for topk in topks:
                grouped[(dataset, target, topk)]["ar"] = df
        else:
            # No spec results for this dataset+target, keep AR standalone
            grouped[(dataset, target, None)]["ar"] = df

    return dict(grouped), draft_models


def aggregate(df: pd.DataFrame) -> pd.Series:
    """Compute summary statistics for one method on one dataset."""
    stats = {
        "num_samples": len(df),
        "avg_prompt_len": df["prompt_len"].mean(),
        "avg_e2e": df["e2e"].mean(),
        "avg_ttft": df["ttft"].mean(),
        "avg_decode_time": df["decode_time"].mean(),
        "avg_draft_time": df["draft_time"].mean(),
        "avg_verify_time": df["verify_time"].mean(),
        "avg_throughput": df["throughput"].mean(),
        "avg_accept_length": df["accept_length"].mean(),
    }
    return pd.Series(stats)


def build_summary(grouped: dict[tuple[str, str, int | None], dict[str, pd.DataFrame]],
                   draft_models: dict) -> pd.DataFrame:
    """Build a summary DataFrame with one row per (dataset, target_model, topk, method)."""
    rows = []
    for (dataset, target_model, topk) in sorted(grouped, key=lambda x: (x[0], x[1], x[2] or 0)):
        for method in ["ar", "cascade", "paged"]:
            if method not in grouped[(dataset, target_model, topk)]:
                continue
            stats = aggregate(grouped[(dataset, target_model, topk)][method])
            stats["dataset"] = dataset
            stats["target_model"] = target_model
            stats["draft_model"] = draft_models.get((dataset, target_model, topk), "—")
            stats["topk"] = topk
            stats["method"] = method
            rows.append(stats)
    summary = pd.DataFrame(rows)
    # Reorder columns
    cols = ["dataset", "target_model", "draft_model", "topk", "method", "num_samples", "avg_prompt_len",
            "avg_e2e", "avg_ttft", "avg_decode_time",
            "avg_draft_time", "avg_verify_time",
            "avg_throughput", "avg_accept_length"]
    return summary[cols]


def add_speedups(summary: pd.DataFrame) -> pd.DataFrame:
    """Add speedup columns relative to AR and cascade-vs-paged."""
    rows = []
    for (dataset, target_model, topk), grp in summary.groupby(["dataset", "target_model", "topk"]):
        ar_row = grp[grp["method"] == "ar"]
        cascade_row = grp[grp["method"] == "cascade"]
        paged_row = grp[grp["method"] == "paged"]

        for _, row in grp.iterrows():
            r = row.copy()
            # Speedup vs AR (on e2e and decode_time)
            if len(ar_row) > 0:
                ar_e2e = ar_row["avg_e2e"].values[0]
                ar_dec = ar_row["avg_decode_time"].values[0]
                r["e2e_speedup_vs_ar"] = ar_e2e / r["avg_e2e"] if r["avg_e2e"] > 0 else float("nan")
                r["decode_speedup_vs_ar"] = ar_dec / r["avg_decode_time"] if r["avg_decode_time"] > 0 else float("nan")
            # Cascade vs paged (only for cascade row)
            if row["method"] == "cascade" and len(paged_row) > 0:
                paged_e2e = paged_row["avg_e2e"].values[0]
                paged_dec = paged_row["avg_decode_time"].values[0]
                paged_draft = paged_row["avg_draft_time"].values[0]
                r["e2e_speedup_vs_paged"] = paged_e2e / r["avg_e2e"] if r["avg_e2e"] > 0 else float("nan")
                r["decode_speedup_vs_paged"] = paged_dec / r["avg_decode_time"] if r["avg_decode_time"] > 0 else float("nan")
                r["draft_speedup_vs_paged"] = paged_draft / r["avg_draft_time"] if r["avg_draft_time"] > 0 else float("nan")
            rows.append(r)
    return pd.DataFrame(rows)


def print_dataset_table(dataset: str, target_model: str, draft_model: str,
                        topk: int | None, df: pd.DataFrame):
    """Pretty-print summary for one dataset + target_model + topk combination."""
    topk_str = f"  topk={topk}" if topk is not None else ""
    draft_str = f"  draft={draft_model}" if draft_model and draft_model != "—" else ""
    print(f"\n{'='*80}")
    print(f"  Dataset: {dataset}  target={target_model}{draft_str}{topk_str}  (n={int(df['num_samples'].iloc[0])})")
    print(f"{'='*80}")

    fmt = {
        "avg_prompt_len": ".0f",
        "avg_e2e": ".3f",
        "avg_ttft": ".3f",
        "avg_decode_time": ".3f",
        "avg_draft_time": ".4f",
        "avg_verify_time": ".4f",
        "avg_throughput": ".2f",
        "avg_accept_length": ".2f",
    }

    header = f"{'Method':<10} {'PromptLen':>9} {'E2E(s)':>8} {'TTFT(s)':>8} {'Decode(s)':>10} {'Draft(s)':>9} {'Verify(s)':>10} {'Tput(t/s)':>10} {'AcceptLen':>10}"
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        method = row["method"]
        line = (
            f"{method:<10} "
            f"{row['avg_prompt_len']:>9.0f} "
            f"{row['avg_e2e']:>8.3f} "
            f"{row['avg_ttft']:>8.3f} "
            f"{row['avg_decode_time']:>10.3f} "
            f"{row['avg_draft_time']:>9.4f} "
            f"{row['avg_verify_time']:>10.4f} "
            f"{row['avg_throughput']:>10.2f} "
            f"{row['avg_accept_length']:>10.2f}"
        )
        print(line)

    # Speedup summary
    cascade_row = df[df["method"] == "cascade"]
    paged_row = df[df["method"] == "paged"]
    ar_row = df[df["method"] == "ar"]

    print()
    if len(cascade_row) > 0 and len(ar_row) > 0:
        r = cascade_row.iloc[0]
        if "e2e_speedup_vs_ar" in r and pd.notna(r.get("e2e_speedup_vs_ar")):
            print(f"  Cascade vs AR:     e2e {r['e2e_speedup_vs_ar']:.3f}x  decode {r['decode_speedup_vs_ar']:.3f}x")
    if len(paged_row) > 0 and len(ar_row) > 0:
        r = paged_row.iloc[0]
        if "e2e_speedup_vs_ar" in r and pd.notna(r.get("e2e_speedup_vs_ar")):
            print(f"  Paged vs AR:       e2e {r['e2e_speedup_vs_ar']:.3f}x  decode {r['decode_speedup_vs_ar']:.3f}x")
    if len(cascade_row) > 0 and "e2e_speedup_vs_paged" in cascade_row.columns:
        r = cascade_row.iloc[0]
        if pd.notna(r.get("e2e_speedup_vs_paged")):
            print(f"  Cascade vs Paged:  e2e {r['e2e_speedup_vs_paged']:.3f}x  decode {r['decode_speedup_vs_paged']:.3f}x  draft {r['draft_speedup_vs_paged']:.3f}x")


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Directory containing result CSVs")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write summary CSV to this path")
    args = parser.parse_args()

    grouped, draft_models = load_results(args.results_dir)
    if not grouped:
        print(f"No result files found in {args.results_dir}")
        return

    summary = build_summary(grouped, draft_models)
    summary = add_speedups(summary)

    for (dataset, target_model, topk) in sorted(grouped, key=lambda x: (x[0], x[1], x[2] or 0)):
        if topk is not None:
            ds_df = summary[(summary["dataset"] == dataset) & (summary["target_model"] == target_model) & (summary["topk"] == topk)]
        else:
            ds_df = summary[(summary["dataset"] == dataset) & (summary["target_model"] == target_model) & (summary["topk"].isna())]
        draft_model = draft_models.get((dataset, target_model, topk), "—")
        print_dataset_table(dataset, target_model, draft_model, topk, ds_df)

    # Overall averages across datasets, per (target_model, topk)
    spec_subset = summary[summary["topk"].notna()]
    for (target_model, topk), subset in spec_subset.groupby(["target_model", "topk"]):
        topk = int(topk)
        print(f"\n{'='*80}")
        print(f"  Overall target={target_model}  topk={topk} (averaged across datasets)")
        print(f"{'='*80}")
        overall = subset.groupby("method")[
            ["avg_e2e", "avg_ttft", "avg_decode_time",
             "avg_draft_time", "avg_verify_time",
             "avg_throughput", "avg_accept_length"]
        ].mean()
        print(overall.to_string(float_format="%.3f"))

    # AR overall per target_model
    ar_subset = summary[summary["method"] == "ar"]
    for target_model, subset in ar_subset.groupby("target_model"):
        # Deduplicate AR rows (they're duplicated into each topk group)
        subset = subset.drop_duplicates(subset="dataset")
        print(f"\n{'='*80}")
        print(f"  Overall AR  target={target_model} (averaged across datasets)")
        print(f"{'='*80}")
        overall = subset[
            ["avg_e2e", "avg_ttft", "avg_decode_time",
             "avg_draft_time", "avg_verify_time",
             "avg_throughput", "avg_accept_length"]
        ].mean()
        print(overall.to_string(float_format="%.3f"))

    if args.output:
        summary.to_csv(args.output, index=False, float_format="%.4f")
        print(f"\nSummary written to {args.output}")


if __name__ == "__main__":
    main()
