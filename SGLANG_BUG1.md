# SGLang Bug #1: STANDALONE spec decoding crashes when target/draft hidden sizes differ

## Background

SGLang's STANDALONE speculative decoding lets you pair an arbitrary draft model with an arbitrary target model. In our setup we use Llama-3.1-8B (target, hidden_size=4096) with Llama-3.2-1B (draft, hidden_size=2048). Standard EAGLE/EAGLE3 draft heads share the target's hidden size by construction; STANDALONE relaxes that.

Reproducing config:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-12.8 \
uv run python tests/bench_dataset.py \
  --dataset-path data/pg19.jsonl \
  --model-path meta-llama/Llama-3.1-8B \
  --draft-model-path meta-llama/Llama-3.2-1B \
  --eagle-topk 10 --speculative-num-steps 7 --speculative-num-draft-tokens 15 \
  --max-new-tokens 512 --context-length 110000 --mem-fraction-static 0.40 \
  --batch-size 4 --tp 1 --time-spec --result-prefix DBG_ --only flat
```

Failure (intermittent, mid-run):

```
File ".../scheduler.py", line 2080, in get_next_batch_to_run
    self.running_batch.merge_batch(self.last_batch)
File ".../schedule_batch.py", line 2234, in merge_batch
    self.spec_info.merge_batch(other.spec_info)
File ".../eagle_info.py", line 804, in merge_batch
    self.hidden_states = torch.cat(
RuntimeError: Sizes of tensors must match except in dimension 0.
Expected size 4096 but got size 2048 for tensor number 1 in the list.
```

Observed across 3 datasets (`narrativeqa`, `gov_report`, `pg19`):

- gov_report: all 5 phases passed.
- narrativeqa: `flat` and `fasttree` crashed with this error; others passed.
- pg19: `flat`, `cascade_no_cg`, `cascade` crashed in different runs.

The crash is timing-dependent on internal scheduler batch merges; whether it fires on a given run depends on prompt lengths, generation lengths, and how requests stagger.

## Why this happens

`EagleDraftInput.hidden_states` carries the **target** model's last-layer activations into the draft step. Every site that populates it from a real forward writes target hidden states, e.g. `eagle_worker.py:790`:

```python
spec_info.hidden_states = logits_output.hidden_states
```

where `logits_output` is the target model's output. Real `hidden_states` therefore have shape `(N, target_hidden_size)`.

The SGLang scheduler maintains a `running_batch` and a `last_batch`, and merges them every iteration via `running_batch.merge_batch(last_batch)` (`scheduler.py:2080`). That delegates to `EagleDraftInput.merge_batch` which does:

```python
self.hidden_states = torch.cat(
    [self.hidden_states, other.hidden_states], axis=0
)
```

Both sides must agree on the trailing dim.

Even one synchronous `engine.generate(...)` call with 4 prompts is *not* one forward — it produces dozens to thousands of internal scheduler iterations:

1. **Chunked prefill.** 110K-token prompts × 4 requests get sliced into many prefill chunks; each chunk is a new `ScheduleBatch` that merges into the running set.
2. **Prefill → decode handoff per request.** Requests transition out of prefill at different times, producing alternating prefill/decode batches that must be stitched back together.
3. **Idle slots.** When the running batch has no in-flight requests for a step (e.g. between scheduler decisions), an *idle* `ScheduleBatch` is materialized with a placeholder `EagleDraftInput`. When a new request becomes ready, the scheduler merges that idle placeholder back into a real batch.

The bug lives in how the idle `EagleDraftInput` placeholder is created. `EagleWorker._draft_preprocess_idle` (`eagle_worker.py:555-561`):

```python
batch.spec_info = EagleDraftInput.create_idle_input(
    device=self.device,
    hidden_size=self.model_config.hidden_size,   # ← draft's hsz
    dtype=self.model_config.dtype,
    topk=self.topk,
    capture_hidden_mode=CaptureHiddenMode.LAST,
)
```

`EagleWorker` inherits `TpModelWorker` with `is_draft_worker=True`, so `self.model_config` is the **draft** model's config. `hidden_size` here is 2048 (Llama-3.2-1B), not 4096.

A second site at `eagle_worker.py:962-977` (the draft-extend idle fallback for batches whose `verified_id` is empty) makes the same mistake.

A third site at `eagle_info.py:240` does it correctly — it uses `batch.model_config.hidden_size`, where `batch` is the target's `ScheduleBatch`.

So the failing cat is:

```
torch.cat([(N, 4096) real, (0, 2048) idle], axis=0)  ->  RuntimeError
```

Standard EAGLE / EAGLE3 don't trip on this because the draft's hidden size matches the target's by design. STANDALONE with mismatched hidden sizes is the exposing case.

## How to fix

At both buggy sites in `3rdparty/sglang/python/sglang/srt/speculative/eagle_worker.py`, swap `self.model_config` (draft's) for `self.target_worker.model_runner.model_config` (target's):

```python
# Site 1: _draft_preprocess_idle (line 555-561)
batch.spec_info = EagleDraftInput.create_idle_input(
    device=self.device,
-   hidden_size=self.model_config.hidden_size,
-   dtype=self.model_config.dtype,
+   hidden_size=self.target_worker.model_runner.model_config.hidden_size,
+   dtype=self.target_worker.model_runner.model_config.dtype,
    topk=self.topk,
    capture_hidden_mode=CaptureHiddenMode.LAST,
)
```

```python
# Site 2: draft-extend idle fallback (line 962-977)
if not input_is_idle and batch.spec_info.verified_id.numel() == 0:
    batch = batch.copy()
    batch.prepare_for_idle()
+   target_cfg = self.target_worker.model_runner.model_config
    hidden_size = (
-       self.model_config.hidden_size * 3
+       target_cfg.hidden_size * 3
        if self.speculative_algorithm.is_eagle3()
        and self.eagle_use_aux_hidden_state
-       else self.model_config.hidden_size
+       else target_cfg.hidden_size
    )
    batch.spec_info = EagleDraftInput.create_idle_input(
        device=self.device,
        hidden_size=hidden_size,
-       dtype=self.model_config.dtype,
+       dtype=target_cfg.dtype,
        topk=self.topk,
        capture_hidden_mode=CaptureHiddenMode.LAST,
    )
```

After the fix:

- Idle placeholder: `(0, 4096)` — empty along the batch dim, trailing dim matches target.
- Real batch: `(N, 4096)` from target output — same trailing dim.
- `torch.cat([(N, 4096), (0, 4096)], axis=0)` succeeds.

For standard EAGLE/EAGLE3 the change is a no-op because draft.hidden_size == target.hidden_size in those modes, so this does not regress them. The behavior only changes when draft and target genuinely differ — i.e. STANDALONE with mismatched hidden sizes.
