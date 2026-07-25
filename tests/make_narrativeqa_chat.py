"""Build a Llama-3 chat-formatted NarrativeQA JSONL with long contexts (>= min_tokens).

User message: "Here is a document:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
Assistant: (empty — generation prompt).
"""
import argparse
import json
import os
from transformers import AutoTokenizer
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/narrativeqa_chat.jsonl")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--min-tokens", type=int, default=50000)
    parser.add_argument("--max-tokens", type=int, default=60000)
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print("Streaming meithnav/narrativeqa")
    ds = load_dataset("meithnav/narrativeqa", split="train", streaming=True)

    seen = set()
    saved = scanned = 0
    with open(args.output, "w") as f:
        for ex in ds:
            scanned += 1
            ctx = (ex.get("context") or "").strip()
            q = (ex.get("question") or "").strip()
            if not ctx or not q:
                continue
            h = hash(ctx[:500])
            if h in seen:
                continue
            seen.add(h)
            user_msg = (
                f"Here is a document:\n\n{ctx}\n\n"
                f"Question: {q}\n\nAnswer:"
            )
            text = tok.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
            n = len(tok.encode(text, add_special_tokens=False))
            if n < args.min_tokens or n > args.max_tokens:
                continue
            f.write(json.dumps({"token_len": n, "prompt": text}) + "\n")
            saved += 1
            if saved % 2 == 0:
                print(f"  {saved}/{args.num_samples} (scanned {scanned})")
            if saved >= args.num_samples:
                break

    print(f"\nDone: {saved} saved to {args.output} (scanned {scanned})")

    if saved == 0:
        return
    lens = [json.loads(l)["token_len"] for l in open(args.output)]
    lens.sort()
    print(f"  Token-length stats: min={lens[0]} max={lens[-1]} "
          f"median={lens[len(lens)//2]} mean={sum(lens)/len(lens):.0f}")


if __name__ == "__main__":
    main()
