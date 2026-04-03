#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="results/sweep_draft_tokens_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

for NDT in $(seq 8 4 128); do
  echo "=== speculative-num-draft-tokens=$NDT ==="
  CUDA_VISIBLE_DEVICES=3,2 uv run python tests/bench_e2e.py \
    --model-path meta-llama/Llama-3.1-8B \
    --draft-model-path meta-llama/Llama-3.2-1B \
    --prompt-lengths 50000 \
    --eagle-topk 15 \
    --max-new-tokens 128 \
    --num-requests 2 \
    --context-length 51000 \
    --mem-fraction-static 0.40 \
    --batch-size 1 \
    --tp 2 \
    --speculative-num-steps 5 \
    --time-spec \
    --temperature 0 \
    --speculative-num-draft-tokens "$NDT" \
    --only cascade \
    --result-file "$RESULTS_DIR/ndt_${NDT}.json"
done

echo ""
echo "=== Verify Time Summary ==="
uv run python -c "
import json, glob, os, sys

results_dir = '$RESULTS_DIR'
files = sorted(glob.glob(os.path.join(results_dir, 'ndt_*.json')))
if not files:
    print('No result files found'); sys.exit(1)

print(f'{\"draft_tokens\":>12}  {\"mean_verify_s\":>13}  {\"mean_draft_s\":>12}')
print('-' * 42)
for f in sorted(files, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])):
    ndt = int(os.path.basename(f).split('_')[1].split('.')[0])
    with open(f) as fh:
        data = json.load(fh)
    verify_times = [r['verify_time_s'] for r in data if r.get('verify_time_s', 0) > 0]
    draft_times = [r['draft_time_s'] for r in data if r.get('draft_time_s', 0) > 0]
    mean_v = sum(verify_times) / len(verify_times) if verify_times else 0
    mean_d = sum(draft_times) / len(draft_times) if draft_times else 0
    print(f'{ndt:>12}  {mean_v:>13.4f}  {mean_d:>12.4f}')
"

echo ""
echo "Results saved to $RESULTS_DIR/"
