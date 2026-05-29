# Experiment Plan

## Global Rules

Use the remote project:

```bash
cd /home/app-ahr/Hetero-KVCache-Optimizer
```

Use target physical GPU1:

```bash
CUDA_VISIBLE_DEVICES=1
```

Inside the process, the selected GPU appears as:

```text
cuda:0
```

For HeteroKV acceptance tests, enforce:

```text
target_hbm_cap = 22 GiB
enable_triton = False
```

Do not run heavy tests if the target GPU has other users' processes. Full 128K full-KV baseline is deferred until the server is idle.

Authenticity rule:

- Report only real measured outcomes, never inferred success.
- Clearly label oracle and diagnostic experiments; they are for diagnosis, not
  final acceptance.
- Preserve failed ideas and negative results in the tracker/reports.
- Acceptance requires matching evidence from code path, mechanism logs, memory
  monitor, and final model output.
- A100 under 22 GiB cap is a memory-envelope result, not a native RTX 4090
  latency claim.

## Stage 0: Safety and Static Checks

Purpose:

- Confirm target GPU safety.
- Confirm static restrictions.

Command:

```bash
CUDA_VISIBLE_DEVICES=1 /home/app-ahr/miniconda3/bin/python scripts/validate_4090_24g_survival.py --stage safety --gpu-index 1
```

Static check:

```bash
CUDA_VISIBLE_DEVICES=1 /home/app-ahr/miniconda3/bin/python scripts/validate_4090_24g_survival.py --stage static --gpu-index 1 --allow-busy --output experiments/4090_24g_static_validation.json
```

Pass criteria:

- `enable_triton` default is false.
- old mean-K / cosine main-path terms are absent.
- GPU safety gate either passes or explicitly skips.

## Stage 1: Small-Tensor Mechanism Tests

Purpose:

- Prove core mechanisms without loading the 7B model.

Command:

```bash
/home/app-ahr/miniconda3/bin/python -m pytest -q tests/test_heterokv_stage1.py
```

Checks:

- Prefill returns short KV.
- Sink/Tail positions are preserved.
- DRAM compressed KV entry is created.
- Incremental prefill remains bounded.
- Dot-product retrieval hits an artificial target chunk.

Current result:

```text
3 passed
```

## Stage 2: Real Qwen Short-Context Smoke

Purpose:

- Verify real model generate compatibility before long tests.

Lengths:

- 2K
- 4K
- 8K

Command:

```bash
CUDA_VISIBLE_DEVICES=1 /home/app-ahr/miniconda3/bin/python scripts/run_stage2_smoke.py --gpu-index 1 --output experiments/stage2_smoke.json
```

Pass criteria:

- Model loads under 22 GiB cap.
- Attention patch applies to Qwen modules.
- generate returns non-empty text.
- No shape mismatch.
- No `position_ids`, `cache_position`, or `attention_mask` failure.
- Memory summary shows bounded active HBM KV.

If GPU1 has another user process, expected output is:

```text
skipped due to shared-server safety
```

## Stage 3: 16K/32K Ablations

Purpose:

- Prove each mechanism contributes.

Lengths:

- 16K
- 32K

Depths:

- 0%
- 25%
- 50%
- 75%
- 90%
- 99%

Configurations:

1. SinkTail only.
2. Sink + Tail + Heavy-Hitter, no retrieval.
3. Mean-K retrieval legacy baseline.
4. Query-Key dot-product retrieval.

Metrics:

- NIAH accuracy.
- retrieved chunk key.
- retrieval score.
- retrieved range.
- active HBM KV length.
- DRAM token length.
- peak allocated/reserved memory.

Expected result:

- Dot-product retrieval should outperform Mean-K retrieval, especially for middle-depth and late-depth needle placements.

## Stage 4: 64K/128K 4090-24G Survival

Purpose:

- Prove the primary project claim.

Configuration:

- Qwen2.5-7B-Instruct.
- 22 GiB cap.
- `enable_triton=False`.
- HeteroKV dot-product retrieval.

Lengths:

- 64K
- 128K

Pass criteria:

- 128K does not OOM.
- nvidia-smi process memory stays within the 24G envelope.
- `torch.cuda.max_memory_reserved()` stays within or close to the 22 GiB cap.
- active HBM KV length stays bounded.
- DRAM compressed KV length grows with context.
- generate produces output.

Required logs:

- Before generate memory.
- Per prefill chunk memory.
- Per eviction memory.
- Per retrieval memory.
- End memory.
- active HBM KV length.
- DRAM compressed KV length.
- actual DRAM bytes.

## Stage 5: Real NIAH

Purpose:

- Prove semantic recall under long context.

Model:

- Qwen2.5-7B-Instruct.

Length:

- 128K.

Depths:

- 0%
- 25%
- 50%
- 75%
- 90%
- 99%

Trials:

- At least 3 random codes per depth.
- Prefer 5 if runtime allows.

Pass criteria:

- Minimum: >= 95%.
- Excellent: 100%.

Required logs:

- target code.
- answer.
- correctness.
- retrieved chunk.
- retrieval score.
- retrieved range.

## Stage 6: WikiText-2 Real PPL

Purpose:

- Measure real semantic degradation, not MSE proxy.

Comparisons:

1. Full KV baseline at the largest length that safely runs.
2. HeteroKV approximate cache at matched length.
3. HeteroKV 128K survival run where full KV may OOM.

Pass criteria:

- HeteroKV PPL degradation <= 5% versus the valid full-KV baseline length.

If full KV 128K OOMs under 24G cap:

- Record OOM as baseline evidence.
- Do not treat full KV 128K OOM as HeteroKV failure.

## Stage 7: Latency Breakdown

Purpose:

- Explain runtime cost before low-level optimization.

Measure:

- prefill time.
- decode ms/token.
- retrieval scoring time.
- dequant / transfer time.
- end-to-end generate time.

Pass criteria:

- Target: <= 2x valid baseline.
- If > 2x but correctness and memory pass, stop and request Triton/CUDA optimization permission.

## Deferred Tests

### Full 128K Full-KV Baseline

Run only when the target GPU is idle.

Purpose:

- Provide OOM or uncapped A100 reference evidence.

Do not run while other users are active.

### Real RTX 4090 Retest

Required only if the final claim needs real 4090 latency, not just A100 under a 4090-like memory envelope.


## Round 24 Addendum: Current Strong NIAH Test Configuration

Use a safe physical GPU selected by current occupancy. If the user has approved switching GPUs, another physical GPU may be used only when the projected total memory is safe. Kill the current HeteroKV process group if it exceeds 30 GiB.

Current strongest NIAH configuration:

- `keep_tail=8192`
- `method_d_top_k=4`
- `score_reduce=query_top_r_mean`
- `query_history_tokens=64`
- `method_d_source_token_boost=2.5`
- `method_d_require_source_overlap=True`
- `method_d_source_cue_focus=True`
- `method_d_retrieve_focus_context_tokens=3`
- `method_d_source_fusion_alpha=0.65`
- `method_d_token_window=64`
- `method_d_reuse_ttl_tokens=6`
- `method_d_reuse_source_threshold=35`
- `method_d_reuse_kv_cache=True`

Optional exactness reranker:

- `method_d_source_copy_logit_boost=20`
- Report this as `experimental SourceCopy`, not as the pure retrieval result.
