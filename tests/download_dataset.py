"""Download and preprocess datasets for benchmarking.

Supported datasets:
  - gov_report  (launch/gov_report) — long government reports
  - pg19        (emozilla/pg19-test) — Project Gutenberg books

Downloads samples via streaming (avoids full download), tokenizes with
Llama-3.1-8B tokenizer, and saves as JSONL with token_len and prompt.

Usage:
    uv run python tests/download_dataset.py --dataset pg19 --min-tokens 30000 --max-tokens 100000
    uv run python tests/download_dataset.py --dataset gov_report --min-tokens 30000
"""

import argparse
import json
import os

import requests
from huggingface_hub import hf_hub_url
from transformers import AutoTokenizer


TOKENIZER_MODEL = "meta-llama/Llama-3.1-8B"


# ---------------------------------------------------------------------------
# gov_report helpers
# ---------------------------------------------------------------------------

GOV_REPORT_FILES = [
    "data/crs_train.jsonl",
    "data/gao_train.jsonl",
]


def flatten_crs_report(node):
    """Recursively flatten CRS nested report structure into text parts."""
    parts = []
    if node.get("section_title"):
        parts.append(node["section_title"])
    for p in node.get("paragraphs", []):
        parts.append(p)
    for sub in node.get("subsections", []):
        parts.extend(flatten_crs_report(sub))
    return parts


def flatten_gao_report(sections):
    """Flatten GAO report (list of section dicts) into text parts."""
    parts = []
    for section in sections:
        if section.get("section_title"):
            parts.append(section["section_title"])
        for p in section.get("paragraphs", []):
            parts.append(p)
        for sub in section.get("subsections", []):
            parts.extend(flatten_gao_report([sub]) if isinstance(sub, dict) else [])
    return parts


def extract_gov_report_text(record, source_file):
    """Extract full report text from a gov_report record."""
    if "crs" in source_file:
        parts = flatten_crs_report(record["reports"])
    else:
        parts = flatten_gao_report(record["report"])
    return "\n\n".join(parts)


def stream_gov_report_records(filename, max_records):
    """Stream JSONL records from launch/gov_report via HTTP Range requests."""
    url = hf_hub_url("launch/gov_report", filename=filename, repo_type="dataset")
    chunk_size = 512 * 1024  # 512KB chunks
    offset = 0
    buffer = ""
    count = 0

    while count < max_records:
        end = offset + chunk_size - 1
        resp = requests.get(url, headers={"Range": f"bytes={offset}-{end}"})
        if resp.status_code not in (200, 206) or not resp.text:
            break

        buffer += resp.text
        lines = buffer.split("\n")
        buffer = lines[-1]

        for line in lines[:-1]:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                yield record
                count += 1
                if count >= max_records:
                    return
            except json.JSONDecodeError:
                continue

        offset += chunk_size


def stream_gov_report(max_per_file):
    """Yield (text,) tuples from gov_report."""
    for source_file in GOV_REPORT_FILES:
        for record in stream_gov_report_records(source_file, max_per_file):
            text = extract_gov_report_text(record, source_file)
            if text.strip():
                yield text


# ---------------------------------------------------------------------------
# PG-19 helpers
# ---------------------------------------------------------------------------

def stream_pg19():
    """Yield (text,) from PG-19 test set via HuggingFace datasets (streaming)."""
    from datasets import load_dataset

    ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
    for sample in ds:
        text = sample.get("text", "")
        if text.strip():
            yield text


# ---------------------------------------------------------------------------
# NarrativeQA helpers
# ---------------------------------------------------------------------------

def stream_narrativeqa():
    """Yield unique document texts from NarrativeQA (deduped by context)."""
    from datasets import load_dataset

    ds = load_dataset("meithnav/narrativeqa", split="train", streaming=True)
    seen = set()
    for sample in ds:
        ctx = sample.get("context", "")
        ctx_hash = hash(ctx[:500])
        if ctx_hash in seen:
            continue
        seen.add(ctx_hash)
        if ctx.strip():
            yield ctx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = {
    "gov_report": {
        "stream_fn": stream_gov_report,
        "needs_max_per_file": True,
    },
    "pg19": {
        "stream_fn": stream_pg19,
        "needs_max_per_file": False,
    },
    "narrativeqa": {
        "stream_fn": stream_narrativeqa,
        "needs_max_per_file": False,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Download dataset samples and save as JSONL with token counts"
    )
    parser.add_argument(
        "--dataset", default="pg19", choices=list(DATASETS.keys()),
        help="Dataset to download (default: pg19)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSONL path (default: data/{dataset}.jsonl)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=200,
        help="Number of samples to collect (default: 200)",
    )
    parser.add_argument(
        "--min-tokens", type=int, default=0,
        help="Skip prompts with fewer than this many tokens (default: 0)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Skip prompts with more than this many tokens (default: no limit)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/{args.dataset}.jsonl"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading tokenizer: {TOKENIZER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    ds_info = DATASETS[args.dataset]
    if ds_info["needs_max_per_file"]:
        text_stream = ds_info["stream_fn"](max_per_file=args.num_samples * 10)
    else:
        text_stream = ds_info["stream_fn"]()

    total = 0
    scanned = 0

    print(f"Streaming from {args.dataset} "
          f"(min_tokens={args.min_tokens}, max_tokens={args.max_tokens}) ...")

    with open(args.output, "w") as f:
        for text in text_stream:
            token_ids = tokenizer.encode(text)
            token_len = len(token_ids)
            scanned += 1
            if token_len < args.min_tokens:
                continue
            if args.max_tokens is not None and token_len > args.max_tokens:
                continue
            f.write(json.dumps({"token_len": token_len, "prompt": text}) + "\n")
            total += 1
            if total % 20 == 0:
                print(f"  {total}/{args.num_samples} samples collected "
                      f"({scanned} scanned)")
            if total >= args.num_samples:
                break

    print(f"\nDone: {total} samples saved to {args.output} "
          f"(scanned {scanned}, min_tokens={args.min_tokens})")

    if total == 0:
        print("  WARNING: no samples matched the token length filter!")
        return

    # Print summary stats
    token_lens = []
    with open(args.output) as f:
        for line in f:
            token_lens.append(json.loads(line)["token_len"])
    token_lens.sort()
    print(f"  Token length stats:")
    print(f"    min={token_lens[0]}, max={token_lens[-1]}, "
          f"median={token_lens[len(token_lens)//2]}, "
          f"mean={sum(token_lens)/len(token_lens):.0f}")


if __name__ == "__main__":
    main()
