"""Aggregate benchmark results across methods (AR, cascade, paged) and datasets."""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Known datasets (used to split dataset name from model name in AR filenames)
DATASETS = {"gov_report", "narrativeqa", "pg19"}

# Known GPU prefixes to strip from filenames
GPU_PREFIXES = ["H200_", "H100_", "A6000_", "PRO6000_", "PRO6000"]

# Regex for speculative decoding filenames:
#   [wo_graph_]standalone_{method}_{target}_{draft}_top{k}_{steps}_{dataset}[_temp{t}]_bs{bs}.csv
SPEC_RE = re.compile(
    r"^(wo_graph_)?standalone_(cascade_per_step_no_cg|cascade_per_step|cascade_no_cg|cascade|fasttree|paged_no_cg|paged|vllm)_(.+?)_(.+?)_top(\d+)_(\d+)_(.+?)(?:_(temp[^_]+))?_bs(\d+)\.csv$"
)


def strip_gpu_prefix(name: str) -> tuple[str, str]:
    """Strip a known GPU prefix from filename, return (gpu, stripped_name)."""
    for prefix in GPU_PREFIXES:
        if name.startswith(prefix):
            return prefix.rstrip("_"), name[len(prefix):]
    return "", name


AR_BS_RE = re.compile(r"(?:_(temp[^_]+))?_bs(\d+)$")


def parse_filename(name: str) -> dict | None:
    """Return metadata dict or None if the file doesn't match any pattern."""
    gpu, name = strip_gpu_prefix(name)
    m = SPEC_RE.match(name)
    if m:
        wo_graph = m.group(1) is not None
        method = m.group(2)
        if wo_graph:
            method = f"{method}_wo_graph"
        return {
            "gpu": gpu,
            "method": method,
            "target_model": m.group(3),
            "draft_model": m.group(4),
            "topk": int(m.group(5)),
            "steps": int(m.group(6)),
            "dataset": m.group(7),
            "temperature": m.group(8) or "temp0",
            "batch_size": int(m.group(9)),
        }
    # AR pattern: [wo_graph_]{target_model}_{dataset}[_temp{t}][_bs{N}].csv
    stem = name.removesuffix(".csv")
    bs_match = AR_BS_RE.search(stem)
    if bs_match:
        temperature = bs_match.group(1) or "temp0"
        batch_size = int(bs_match.group(2))
        stem = stem[: bs_match.start()]
    else:
        temperature = "temp0"
        batch_size = 1
    for ds in DATASETS:
        if stem.endswith(f"_{ds}"):
            target = stem.removesuffix(f"_{ds}")
            wo_graph = target.startswith("wo_graph_")
            if wo_graph:
                target = target.removeprefix("wo_graph_")
            method = "ar_wo_graph" if wo_graph else "ar"
            return {
                "gpu": gpu,
                "method": method,
                "target_model": target,
                "draft_model": None,
                "topk": None,
                "steps": None,
                "dataset": ds,
                "temperature": temperature,
                "batch_size": batch_size,
            }
    return None


def load_results(results_dir: Path):
    """Load CSVs grouped as {(gpu, dataset, target_model, topk, steps, batch_size, temperature): {method: DataFrame}}.

    AR results (topk=None, steps=None) are duplicated into every (topk, steps)
    group for the same (gpu, dataset, target_model, batch_size, temperature) so
    that speedup-vs-AR comparisons work per group.
    """
    grouped = defaultdict(dict)
    ar_by_key: dict[tuple[tuple[str, str, str, int, str], str], pd.DataFrame] = {}
    spec_keys_by_target: dict[tuple[str, str, str, int, str], set[tuple[int, int]]] = defaultdict(set)
    draft_models: dict[tuple[str, str, str, int, int, int, str], str] = {}

    for f in sorted(results_dir.glob("*.csv")):
        meta = parse_filename(f.name)
        if meta is None:
            continue
        df = pd.read_csv(f)
        df["decode_time"] = df["e2e"] - df["ttft"]
        bs = meta["batch_size"]
        temp = meta["temperature"]
        gpu_ds_target_bs = (meta["gpu"], meta["dataset"], meta["target_model"], bs, temp)
        if meta["topk"] is None:
            ar_by_key[(gpu_ds_target_bs, meta["method"])] = df
        else:
            key = (meta["gpu"], meta["dataset"], meta["target_model"], meta["topk"], meta["steps"], bs, temp)
            grouped[key][meta["method"]] = df
            spec_keys_by_target[gpu_ds_target_bs].add((meta["topk"], meta["steps"]))
            if meta["draft_model"]:
                draft_models[key] = meta["draft_model"]

    # Inject AR (and ar_wo_graph) into each (topk, steps) group for matching batch_size + temperature
    for ((gpu, dataset, target, bs, temp), method), df in ar_by_key.items():
        keys = spec_keys_by_target.get((gpu, dataset, target, bs, temp))
        if keys:
            for topk, steps in keys:
                grouped[(gpu, dataset, target, topk, steps, bs, temp)][method] = df
        else:
            # No spec results for this dataset+target+bs+temp, keep AR standalone
            grouped[(gpu, dataset, target, None, None, bs, temp)][method] = df

    return dict(grouped), draft_models


def aggregate(df: pd.DataFrame) -> pd.Series:
    """Compute summary statistics (mean and std) for one method on one dataset."""
    stats = {
        "num_samples": len(df),
        "avg_prompt_len": df["prompt_len"].mean(),
        "avg_e2e": df["e2e"].mean(),
        "std_e2e": df["e2e"].std(),
        "avg_ttft": df["ttft"].mean(),
        "std_ttft": df["ttft"].std(),
        "avg_decode_time": df["decode_time"].mean(),
        "std_decode_time": df["decode_time"].std(),
        "avg_draft_time": df["draft_time"].mean(),
        "std_draft_time": df["draft_time"].std(),
        "avg_verify_time": df["verify_time"].mean(),
        "std_verify_time": df["verify_time"].std(),
        "avg_throughput": df["throughput"].mean(),
        "std_throughput": df["throughput"].std(),
        "avg_accept_length": df["accept_length"].mean(),
        "std_accept_length": df["accept_length"].std(),
    }
    return pd.Series(stats)


def build_summary(grouped, draft_models: dict) -> pd.DataFrame:
    """Build a summary DataFrame with one row per (gpu, dataset, target_model, topk, steps, batch_size, temperature, method)."""
    rows = []
    for key in sorted(grouped, key=lambda x: (x[0], x[1], x[2], x[3] or 0, x[4] or 0, x[5], x[6])):
        gpu, dataset, target_model, topk, steps, batch_size, temperature = key
        for method in sorted(grouped[key].keys()):
            stats = aggregate(grouped[key][method])
            stats["gpu"] = gpu
            stats["dataset"] = dataset
            stats["target_model"] = target_model
            stats["draft_model"] = draft_models.get(key, "—")
            stats["topk"] = topk
            stats["steps"] = steps
            stats["batch_size"] = batch_size
            stats["temperature"] = temperature
            stats["method"] = method
            rows.append(stats)
    summary = pd.DataFrame(rows)
    # Reorder columns
    cols = ["gpu", "dataset", "target_model", "draft_model", "topk", "steps", "batch_size", "temperature",
            "method", "num_samples", "avg_prompt_len",
            "avg_e2e", "std_e2e", "avg_ttft", "std_ttft",
            "avg_decode_time", "std_decode_time",
            "avg_draft_time", "std_draft_time",
            "avg_verify_time", "std_verify_time",
            "avg_throughput", "std_throughput",
            "avg_accept_length", "std_accept_length"]
    return summary[cols]


def _per_sample_speedup_std(df_ref: pd.DataFrame, df_method: pd.DataFrame, col: str) -> float:
    """Compute std of per-sample speedup (ref[col] / method[col]) matched on prompt_idx."""
    merged = df_ref.merge(df_method, on="prompt_idx", suffixes=("_ref", "_m"))
    if len(merged) == 0:
        return float("nan")
    ratios = merged[f"{col}_ref"] / merged[f"{col}_m"]
    return ratios.std()


def add_speedups(summary: pd.DataFrame, grouped: dict) -> pd.DataFrame:
    """Add speedup columns (mean and std) relative to AR and cascade-vs-paged."""
    rows = []
    for (gpu, dataset, target_model, topk, steps, batch_size, temperature), grp in summary.groupby(
        ["gpu", "dataset", "target_model", "topk", "steps", "batch_size", "temperature"], dropna=False):
        # Construct key for raw DataFrame lookup (NaN → None)
        raw_key = (
            gpu, dataset, target_model,
            None if pd.isna(topk) else int(topk),
            None if pd.isna(steps) else int(steps),
            int(batch_size), temperature,
        )
        raw = grouped.get(raw_key, {})

        ar_row = grp[grp["method"] == "ar"]
        ar_wo_graph_row = grp[grp["method"] == "ar_wo_graph"]
        paged_row = grp[grp["method"] == "paged"]
        paged_wo_graph_row = grp[grp["method"] == "paged_wo_graph"]

        for _, row in grp.iterrows():
            r = row.copy()
            method = row["method"]
            is_wo_graph = method.endswith("_wo_graph")

            # Pick matching AR baseline (wo_graph methods compare against ar_wo_graph)
            ref_ar = ar_wo_graph_row if is_wo_graph else ar_row
            ref_ar_method = "ar_wo_graph" if is_wo_graph else "ar"
            if len(ref_ar) > 0:
                ar_e2e = ref_ar["avg_e2e"].values[0]
                ar_dec = ref_ar["avg_decode_time"].values[0]
                r["e2e_speedup_vs_ar"] = ar_e2e / r["avg_e2e"] if r["avg_e2e"] > 0 else float("nan")
                r["decode_speedup_vs_ar"] = ar_dec / r["avg_decode_time"] if r["avg_decode_time"] > 0 else float("nan")

                # Per-sample speedup std
                ar_df = raw.get(ref_ar_method)
                m_df = raw.get(method)
                if ar_df is not None and m_df is not None and method != ref_ar_method:
                    r["std_e2e_speedup_vs_ar"] = _per_sample_speedup_std(ar_df, m_df, "e2e")
                    r["std_decode_speedup_vs_ar"] = _per_sample_speedup_std(ar_df, m_df, "decode_time")

            # Cascade/FastTree/vLLM/paged_no_cg/cascade_per_step vs paged (within same graph/no-graph group)
            if method in ("cascade", "cascade_no_cg", "cascade_per_step", "cascade_per_step_no_cg",
                          "fasttree", "vllm", "paged_no_cg") and len(paged_row) > 0:
                ref_paged = paged_row
                ref_paged_method = "paged"
            elif method == "cascade_wo_graph" and len(paged_wo_graph_row) > 0:
                ref_paged = paged_wo_graph_row
                ref_paged_method = "paged_wo_graph"
            else:
                ref_paged = None
                ref_paged_method = None

            if ref_paged is not None:
                paged_e2e = ref_paged["avg_e2e"].values[0]
                paged_dec = ref_paged["avg_decode_time"].values[0]
                paged_draft = ref_paged["avg_draft_time"].values[0]
                r["e2e_speedup_vs_paged"] = paged_e2e / r["avg_e2e"] if r["avg_e2e"] > 0 else float("nan")
                r["decode_speedup_vs_paged"] = paged_dec / r["avg_decode_time"] if r["avg_decode_time"] > 0 else float("nan")
                r["draft_speedup_vs_paged"] = paged_draft / r["avg_draft_time"] if r["avg_draft_time"] > 0 else float("nan")

                # Per-sample speedup std
                p_df = raw.get(ref_paged_method)
                m_df = raw.get(method)
                if p_df is not None and m_df is not None:
                    r["std_e2e_speedup_vs_paged"] = _per_sample_speedup_std(p_df, m_df, "e2e")
                    r["std_decode_speedup_vs_paged"] = _per_sample_speedup_std(p_df, m_df, "decode_time")
                    r["std_draft_speedup_vs_paged"] = _per_sample_speedup_std(p_df, m_df, "draft_time")
            rows.append(r)
    return pd.DataFrame(rows)


def print_dataset_table(gpu: str, dataset: str, target_model: str, draft_model: str,
                        topk: int | None, steps: int | None, batch_size: int,
                        temperature: str, df: pd.DataFrame):
    """Pretty-print summary for one gpu + dataset + target_model + topk + steps + batch_size + temperature combination."""
    gpu_str = f"  gpu={gpu}" if gpu else ""
    topk_str = f"  topk={topk}" if topk is not None else ""
    steps_str = f"  steps={steps}" if steps is not None else ""
    bs_str = f"  bs={batch_size}"
    temp_str = f"  {temperature}"
    draft_str = f"  draft={draft_model}" if draft_model and draft_model != "—" else ""
    if df.empty:
        print(f"\n{'='*80}")
        print(f"  [SKIP] Empty dataframe for{gpu_str} dataset={dataset} target={target_model}"
              f" draft={draft_model} topk={topk} steps={steps} bs={batch_size} temp={temperature}")
        print(f"  Columns: {list(df.columns)}")
        print(f"{'='*80}")
        return
    print(f"\n{'='*80}")
    print(f"  Dataset: {dataset}  target={target_model}{draft_str}{topk_str}{steps_str}{bs_str}{temp_str}{gpu_str}  (n={int(df['num_samples'].iloc[0])})")
    print(f"{'='*80}")

    def ms(mean, std):
        """Format mean±std compactly."""
        return f"{mean:.3f}±{std:.3f}"

    method_w = max(10, max((len(str(m)) for m in df["method"]), default=10))
    header = (f"{'Method':<{method_w}} {'PromptLen':>9} {'E2E(s)':>15} {'TTFT(s)':>15} {'Decode(s)':>15} "
              f"{'Draft(s)':>15} {'Verify(s)':>15} {'Tput(t/s)':>15} {'AcceptLen':>11}")
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        method = row["method"]
        line = (
            f"{method:<{method_w}} "
            f"{row['avg_prompt_len']:>9.0f} "
            f"{ms(row['avg_e2e'], row['std_e2e']):>15} "
            f"{ms(row['avg_ttft'], row['std_ttft']):>15} "
            f"{ms(row['avg_decode_time'], row['std_decode_time']):>15} "
            f"{ms(row['avg_draft_time'], row['std_draft_time']):>15} "
            f"{ms(row['avg_verify_time'], row['std_verify_time']):>15} "
            f"{ms(row['avg_throughput'], row['std_throughput']):>15} "
            f"{row['avg_accept_length']:>6.2f}±{row['std_accept_length']:<4.2f}"
        )
        print(line)

    # Speedup summary
    print()
    # Print speedups for each cascade variant vs its matching paged/AR
    for suffix, label_prefix in [("", ""), ("_wo_graph", " (wo_graph)")]:
        cascade_row = df[df["method"] == f"cascade{suffix}"]
        paged_row = df[df["method"] == f"paged{suffix}"]
        ar_row = df[df["method"] == f"ar{suffix}"]

        if len(cascade_row) > 0 and len(ar_row) > 0:
            r = cascade_row.iloc[0]
            if "e2e_speedup_vs_ar" in r and pd.notna(r.get("e2e_speedup_vs_ar")):
                e2e_std = f"±{r['std_e2e_speedup_vs_ar']:.3f}" if pd.notna(r.get("std_e2e_speedup_vs_ar")) else ""
                dec_std = f"±{r['std_decode_speedup_vs_ar']:.3f}" if pd.notna(r.get("std_decode_speedup_vs_ar")) else ""
                print(f"  Cascade{label_prefix} vs AR:     e2e {r['e2e_speedup_vs_ar']:.3f}{e2e_std}x  decode {r['decode_speedup_vs_ar']:.3f}{dec_std}x")
        if len(paged_row) > 0 and len(ar_row) > 0:
            r = paged_row.iloc[0]
            if "e2e_speedup_vs_ar" in r and pd.notna(r.get("e2e_speedup_vs_ar")):
                e2e_std = f"±{r['std_e2e_speedup_vs_ar']:.3f}" if pd.notna(r.get("std_e2e_speedup_vs_ar")) else ""
                dec_std = f"±{r['std_decode_speedup_vs_ar']:.3f}" if pd.notna(r.get("std_decode_speedup_vs_ar")) else ""
                print(f"  Paged{label_prefix} vs AR:       e2e {r['e2e_speedup_vs_ar']:.3f}{e2e_std}x  decode {r['decode_speedup_vs_ar']:.3f}{dec_std}x")
        if len(cascade_row) > 0 and "e2e_speedup_vs_paged" in cascade_row.columns:
            r = cascade_row.iloc[0]
            if pd.notna(r.get("e2e_speedup_vs_paged")):
                e2e_std = f"±{r['std_e2e_speedup_vs_paged']:.3f}" if pd.notna(r.get("std_e2e_speedup_vs_paged")) else ""
                dec_std = f"±{r['std_decode_speedup_vs_paged']:.3f}" if pd.notna(r.get("std_decode_speedup_vs_paged")) else ""
                draft_std = f"±{r['std_draft_speedup_vs_paged']:.3f}" if pd.notna(r.get("std_draft_speedup_vs_paged")) else ""
                print(f"  Cascade{label_prefix} vs Paged:  e2e {r['e2e_speedup_vs_paged']:.3f}{e2e_std}x  decode {r['decode_speedup_vs_paged']:.3f}{dec_std}x  draft {r['draft_speedup_vs_paged']:.3f}{draft_std}x")

    # cascade_no_cg vs AR
    cascade_no_cg_row = df[df["method"] == "cascade_no_cg"]
    if len(cascade_no_cg_row) > 0:
        r = cascade_no_cg_row.iloc[0]
        if "e2e_speedup_vs_ar" in r and pd.notna(r.get("e2e_speedup_vs_ar")):
            e2e_std = f"±{r['std_e2e_speedup_vs_ar']:.3f}" if pd.notna(r.get("std_e2e_speedup_vs_ar")) else ""
            dec_std = f"±{r['std_decode_speedup_vs_ar']:.3f}" if pd.notna(r.get("std_decode_speedup_vs_ar")) else ""
            print(f"  Cascade (no_cg) vs AR:     e2e {r['e2e_speedup_vs_ar']:.3f}{e2e_std}x  decode {r['decode_speedup_vs_ar']:.3f}{dec_std}x")

    # cascade_per_step / cascade_per_step_no_cg vs AR / vs Paged
    for method_name, label in [("cascade_per_step", "Cascade (per-step)"),
                                ("cascade_per_step_no_cg", "Cascade (per-step, no_cg)")]:
        row = df[df["method"] == method_name]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        if "e2e_speedup_vs_ar" in r and pd.notna(r.get("e2e_speedup_vs_ar")):
            e2e_std = f"±{r['std_e2e_speedup_vs_ar']:.3f}" if pd.notna(r.get("std_e2e_speedup_vs_ar")) else ""
            dec_std = f"±{r['std_decode_speedup_vs_ar']:.3f}" if pd.notna(r.get("std_decode_speedup_vs_ar")) else ""
            print(f"  {label} vs AR:     e2e {r['e2e_speedup_vs_ar']:.3f}{e2e_std}x  decode {r['decode_speedup_vs_ar']:.3f}{dec_std}x")
        if "e2e_speedup_vs_paged" in r and pd.notna(r.get("e2e_speedup_vs_paged")):
            e2e_std = f"±{r['std_e2e_speedup_vs_paged']:.3f}" if pd.notna(r.get("std_e2e_speedup_vs_paged")) else ""
            dec_std = f"±{r['std_decode_speedup_vs_paged']:.3f}" if pd.notna(r.get("std_decode_speedup_vs_paged")) else ""
            draft_std = f"±{r['std_draft_speedup_vs_paged']:.3f}" if pd.notna(r.get("std_draft_speedup_vs_paged")) else ""
            print(f"  {label} vs Paged:  e2e {r['e2e_speedup_vs_paged']:.3f}{e2e_std}x  decode {r['decode_speedup_vs_paged']:.3f}{dec_std}x  draft {r['draft_speedup_vs_paged']:.3f}{draft_std}x")

    # FastTree vs AR
    fasttree_row = df[df["method"] == "fasttree"]
    if len(fasttree_row) > 0:
        r = fasttree_row.iloc[0]
        if "e2e_speedup_vs_ar" in r and pd.notna(r.get("e2e_speedup_vs_ar")):
            e2e_std = f"±{r['std_e2e_speedup_vs_ar']:.3f}" if pd.notna(r.get("std_e2e_speedup_vs_ar")) else ""
            dec_std = f"±{r['std_decode_speedup_vs_ar']:.3f}" if pd.notna(r.get("std_decode_speedup_vs_ar")) else ""
            print(f"  FastTree vs AR:    e2e {r['e2e_speedup_vs_ar']:.3f}{e2e_std}x  decode {r['decode_speedup_vs_ar']:.3f}{dec_std}x")

    # vLLM vs AR / vs Paged
    vllm_row = df[df["method"] == "vllm"]
    if len(vllm_row) > 0:
        r = vllm_row.iloc[0]
        if "e2e_speedup_vs_ar" in r and pd.notna(r.get("e2e_speedup_vs_ar")):
            e2e_std = f"±{r['std_e2e_speedup_vs_ar']:.3f}" if pd.notna(r.get("std_e2e_speedup_vs_ar")) else ""
            dec_std = f"±{r['std_decode_speedup_vs_ar']:.3f}" if pd.notna(r.get("std_decode_speedup_vs_ar")) else ""
            print(f"  vLLM vs AR:        e2e {r['e2e_speedup_vs_ar']:.3f}{e2e_std}x  decode {r['decode_speedup_vs_ar']:.3f}{dec_std}x")
        if "e2e_speedup_vs_paged" in r and pd.notna(r.get("e2e_speedup_vs_paged")):
            e2e_std = f"±{r['std_e2e_speedup_vs_paged']:.3f}" if pd.notna(r.get("std_e2e_speedup_vs_paged")) else ""
            dec_std = f"±{r['std_decode_speedup_vs_paged']:.3f}" if pd.notna(r.get("std_decode_speedup_vs_paged")) else ""
            print(f"  vLLM vs Paged:     e2e {r['e2e_speedup_vs_paged']:.3f}{e2e_std}x  decode {r['decode_speedup_vs_paged']:.3f}{dec_std}x")


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
    summary = add_speedups(summary, grouped)

    for key in sorted(grouped, key=lambda x: (x[0], x[1], x[2], x[3] or 0, x[4] or 0, x[5], x[6])):
        gpu, dataset, target_model, topk, steps, batch_size, temperature = key
        base_mask = (
            (summary["gpu"] == gpu)
            & (summary["dataset"] == dataset)
            & (summary["target_model"] == target_model)
            & (summary["batch_size"] == batch_size)
            & (summary["temperature"] == temperature)
        )
        if topk is not None:
            ds_df = summary[base_mask & (summary["topk"] == topk) & (summary["steps"] == steps)]
        else:
            ds_df = summary[base_mask & (summary["topk"].isna()) & (summary["steps"].isna())]
        draft_model = draft_models.get(key, "—")
        print_dataset_table(gpu, dataset, target_model, draft_model, topk, steps, batch_size, temperature, ds_df)

    if args.output:
        summary.to_csv(args.output, index=False, float_format="%.4f")
        print(f"\nSummary written to {args.output}")


if __name__ == "__main__":
    main()
