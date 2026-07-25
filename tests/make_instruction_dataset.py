"""Build a Llama-3 instruction-formatted JSONL from a public instruction dataset.

Pulls from yahma/alpaca-cleaned (no auth required), wraps each instruction in
the Llama-3 chat template via tokenizer.apply_chat_template, and saves as
JSONL with {"token_len": int, "prompt": str}.

Usage:
    uv run python tests/make_instruction_dataset.py \
        --output data/alpaca_llama3.jsonl \
        --num-samples 30 \
        --min-tokens 200
"""
import argparse
import json
import os
from transformers import AutoTokenizer
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/alpaca_llama3.jsonl")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--min-tokens", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", default="yahma/alpaca-cleaned")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Streaming {args.dataset}")
    ds = load_dataset(args.dataset, split="train", streaming=True)

    saved = scanned = 0
    with open(args.output, "w") as f:
        for ex in ds:
            scanned += 1
            instr = ex.get("instruction", "").strip()
            inp = ex.get("input", "").strip()
            if not instr:
                continue
            user_msg = f"{instr}\n\n{inp}" if inp else instr
            # apply Llama-3 chat template with add_generation_prompt=True
            text = tok.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
            token_ids = tok.encode(text, add_special_tokens=False)
            n = len(token_ids)
            if n < args.min_tokens or n > args.max_tokens:
                continue
            f.write(json.dumps({"token_len": n, "prompt": text}) + "\n")
            saved += 1
            if saved >= args.num_samples:
                break

    print(f"\nDone: {saved}/{args.num_samples} saved to {args.output} "
          f"(scanned {scanned}, min={args.min_tokens})")

    if saved == 0:
        return
    lens = [json.loads(l)["token_len"] for l in open(args.output)]
    lens.sort()
    print(f"  Token-length stats: min={lens[0]} max={lens[-1]} "
          f"median={lens[len(lens)//2]} mean={sum(lens)/len(lens):.0f}")


if __name__ == "__main__":
    main()
