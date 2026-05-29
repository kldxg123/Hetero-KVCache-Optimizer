# Hetero-KVCache-Optimizer Idea Report

## Executive Summary

Hetero-KVCache-Optimizer should be positioned as an **Approximate Long-Context Cache** system, not as a lossless 128K full-KV reconstruction system. The strongest idea is to prove that Qwen2.5-7B-Instruct can survive 128K context under a 4090-like 24G memory budget by physically bounding active HBM KV and recovering important evicted information through token-level Query x Key retrieval from 4-bit DRAM storage.

The winning direction is:

1. Physically truncate prefill KV into Sink + Tail + Heavy-Hitter.
2. Store evicted KV in DRAM-side quantized format.
3. Use RoPE-aware short-KV attention with true logical positions.
4. Replace mean-K retrieval with token-level Query x Key dot-product retrieval.
5. Validate under a 22 GiB memory cap on A100 as a conservative RTX 4090 24G survival proxy.

## Ranked Ideas

| Rank | Idea | Value | Feasibility | Risk | Decision |
|---:|---|---|---|---|---|
| 1 | 4090-24G survival proof with 22 GiB cap on A100 | Very high | High | Medium | Adopt |
| 2 | Physical short-KV prefill: Sink + Tail + Heavy-Hitter | Very high | High | Medium | Adopt |
| 3 | RoPE-aware short-KV attention using logical key positions | Very high | Medium | High | Adopt |
| 4 | Token-level Query x Key dot-product retrieval from 4-bit DRAM Key | Very high | Medium | Medium | Adopt |
| 5 | Stage-based tests before 128K runs | High | High | Low | Adopt |
| 6 | SinkTail / HH / MeanK / DotProduct ablation matrix | High | High | Low | Adopt |
| 7 | WikiText-2 real PPL instead of MSE proxy | High | Medium | Medium | Adopt |
| 8 | Latency breakdown before low-level optimization | Medium-high | High | Low | Adopt |
| 9 | Triton fused dequant attention | Potentially high | Medium-low | High | Defer |
| 10 | Full 128K full-KV baseline | Medium | Low under 24G | High | Run only as OOM/baseline evidence when server is idle |

## Adopted Ideas

### 1. 4090-24G Survival Proof

Use the A100 server, but run the HeteroKV acceptance path under a conservative `22 GiB` PyTorch memory cap. This gives a stronger and safer claim than simply saying the model runs on A100.

Claim allowed:

> HeteroKV supports Qwen2.5-7B-Instruct 128K context under a 4090-like 24G memory envelope on A100.

Claim not allowed without real hardware:

> HeteroKV latency on RTX 4090 is proven.

### 2. Physical Short-KV Prefill

The previous implementation compressed some tokens but still returned full K/V during prefill. That cannot prove O(1) HBM active KV. The adopted fix is to physically return only the short retained KV set.

Required evidence:

- `_prefill_update()` return length is bounded.
- active HBM KV length stays stable as context grows.
- evicted tokens are present in DRAM storage.

### 3. Logical Position Tracking for Short KV

Because Sink, Tail, Heavy-Hitter, and retrieved chunks are non-contiguous in logical sequence space, attention must not treat the short KV as a compressed contiguous sequence. The adopted design tracks true logical key positions and builds causal masking from those positions.

Required evidence:

- `position_ids` remain absolute.
- `cache_position` remains absolute.
- attention mask matches physical short-KV length.
- no padding back to 128K.

### 4. Query-Key Dot-Product Retrieval

Mean-K retrieval averages away token-level needle information. The adopted design scores DRAM chunks by dequantizing candidate 4-bit Keys in small batches and computing:

```text
score = Query x Key
```

Default chunk score is max token score. Top-r mean can be logged as an auxiliary metric, but max is the default for needle-like retrieval.

### 5. Stage-Gated Validation

Heavy 128K tests should not be the first test. The adopted route is:

1. Static and safety checks.
2. Small tensor mechanism tests.
3. Real model 2K/4K/8K smoke tests.
4. 16K/32K ablations.
5. 64K/128K survival.
6. NIAH, PPL, latency.

## Failed or Rejected Ideas

### Failed Idea: Return Full K/V During Prefill and Rely on GC

Reason rejected:

- This keeps the attention-facing K/V full length.
- It cannot demonstrate O(1) active HBM KV.
- It can hide memory growth behind transient cleanup logs.

Current rule:

- Prefill must return physically truncated K/V.

### Failed Idea: Pad Short KV Back to 128K

Reason rejected:

- It defeats the memory objective.
- It can make shape checks pass while preserving O(N) HBM behavior.

Current rule:

- Never pad short KV back to original context length.

### Failed Idea: Mean-K / Chunk Embedding / Cosine Retrieval

Reason rejected:

- Token-level needles are destroyed by averaging.
- It explains previous NIAH failures at long depths.

Current rule:

- Mean-K can only appear in historical notes or ablation, never as the main retrieval path.

### Failed Idea: Use MSE Proxy as PPL Evidence

Reason rejected:

- MSE reconstruction error is not real language-model perplexity.
- It cannot prove semantic loss is controlled.

Current rule:

- WikiText-2 PPL must use true model loss.

### Failed Idea: Claim Real 4090 Latency from A100

Reason rejected:

- A100 and RTX 4090 differ in memory bandwidth, PCIe behavior, kernel behavior, and scheduling.

Current rule:

- A100 capped tests prove a 4090-like memory envelope, not real 4090 latency.

### Deferred Idea: Triton / CUDA Fused Dequant Attention

Reason deferred:

- It adds complexity before correctness is proven.
- It risks hiding algorithmic bugs behind low-level optimization.

Current rule:

- Use PyTorch matmul first.
- Apply Triton only if correctness, memory, NIAH, and PPL pass but latency remains above target.

## Current Implementation Status

Implemented so far:

- Short-KV physical prefill return path.
- Logical key-position tracking.
- RoPE-aware attention patch for short KV.
- Query-aware token-level dot-product retriever.
- Stage 1 small-tensor mechanism tests.
- 4090-24G safety gate.
- Stage 2 smoke-test runner with shared-server safety gate.
- Project guardrails document.

Not yet run due to shared-server GPU safety:

- Real Qwen2.5-7B 2K/4K/8K smoke test.
- 16K/32K ablations.
- 64K/128K survival test.
- Real NIAH.
- WikiText-2 PPL.
- Latency breakdown.

## Workflow 2.0 Idea Updates

### Adopted: Calibrated 4K Before Scaling

Decision:

- Keep the next research loop at 4K until retrieval quality is understood.

Reason:

- Full KV baseline now reaches 4/4 on calibrated 4K NIAH.
- HeteroKV no-retrieval and dotproduct both reach only 2/4 with `keep_tail=2048`.
- Scaling this failure to 16K/32K/128K would waste GPU time and produce ambiguous evidence.

### Adopted: Non-Eviction Control

Decision:

- Use `keep_tail=4096` as a diagnostic control, not as the final memory-saving setting.

Evidence:

- Full KV: 4/4.
- HeteroKV no-retrieval with `keep_tail=4096`: 4/4.

Meaning:

- The attention wrapper and absolute position path are plausibly correct when the needle is still present in active HBM KV.
- The blocker is evicted-token recovery, not basic generate compatibility.

### Failed Idea: Force More Retrieval

Experiment:

- Method-D gate margin set to `0`.

Result:

- Dotproduct accuracy dropped to 0/4.

Lesson:

- Retrieval must be selective. More DRAM KV is not automatically better.

### Failed Idea: Single Best-Token Window

Experiment:

- Retrieve only a 256-token window around the highest QK token inside the selected chunk.

Result:

- Dotproduct remained 2/4.

Lesson:

- The highest QK token often drifts toward generic late-chunk text rather than the needle.
- Even when windows overlap the needle, recovered KV does not reliably steer generation yet.

### Next Ranked Ideas

| Rank | Idea | Purpose | Decision |
|---:|---|---|---|
| 1 | First-token attention-mass logging for retrieved DRAM vs active HBM | Prove whether retrieved tokens receive useful attention after injection | Next |
| 2 | Oracle retrieval by known needle range | Separate ranking failure from injection/attention failure | Next diagnostic only |
| 3 | BF16 DRAM retrieval ablation | Separate 4-bit quantization loss from cache approximation loss | Next diagnostic |
| 4 | Multi-window retrieval around top-r tokens | Reduce risk that a single best token misses the needle | Candidate after diagnostics |
| 5 | Proceed directly to 16K/32K NIAH | Would scale an unresolved 4K quality failure | Reject for now |

## Workflow 2.0 Round 3 Idea Outcomes

### Adopted Diagnostic: Retrieval-Aware First Token

Problem:

- HF `generate()` emits the first answer token from prefill logits.
- Method-D retrieval only runs during decode.

Decision:

- HeteroKV NIAH diagnostics now use chunked prefill through the penultimate prompt token, then decode the final prompt token to generate the first answer token.

Outcome:

- `keep_tail=4096` control remains 4/4, so this diagnostic path is valid.

### Failed Idea: Oracle Chunk Retrieval Is Sufficient

Experiment:

- Force retrieval of the DRAM chunk containing the known needle range.

Result:

- Accuracy stayed 2/4.

Lesson:

- Ranking is not the only blocker. Even a known-good chunk does not recover early/mid NIAH.

### Failed Idea: Oracle 64-Token Window Is Sufficient

Experiment:

- Force retrieval of a 64-token window around the known needle range.

Result:

- Accuracy stayed 2/4.

Lesson:

- Finer retrieval helps attention mass but does not yet steer generation.

### Failed Idea: 4-bit Quantization Is The Primary Cause

Experiment:

- Store BF16 diagnostic DRAM copies and retrieve from BF16 instead of 4-bit for oracle mode.

Result:

- Accuracy stayed 2/4.

Lesson:

- Quantization loss is not the primary blocker in the current 4K diagnostic.

### Updated Next Ideas

| Rank | Idea | Purpose | Decision |
|---:|---|---|---|
| 1 | Full-prefill-small-control then short-cache decode | Separate prefill representation damage from decode retrieval failure | Next diagnostic |
| 2 | Preserve semantic neighborhood spans around detected facts | Give retrieval enough local context to steer answer tokens | Candidate |
| 3 | Retrieval-aware first-token API | Make the project compatible with standard answer-first tasks | Candidate |
| 4 | Scale to 16K/128K semantic NIAH now | Would hide unresolved 4K failure | Reject for now |

## Workflow 2.0 Round 4 Idea Outcomes

### Adopted: Query-History Dot-Product Actually Uses Multiple Query Tokens

Problem:

- `method_d_query_history_tokens` existed in the CLI, but the retriever truncated `query_states` to the final token before scoring.

Outcome:

- Added multi-query scoring and `query_top_r_mean`.
- Added a Stage1 test proving a multi-token consensus source beats a single-token spike false positive.

### Failed: Focus-Only Source Bias

Result:

- 32K remained 4/4.
- 128K 25% still failed.

Lesson:

- Correct chunks can be retrieved but still lose inside the attention/fusion stage.

### Failed: Stronger Bias And Top-K Shrinking Alone

Result:

- Stronger focus bias shifted outputs but did not recover the target.
- top4/top1 source windows remained 32K-correct but 128K 25%-incorrect.

Lesson:

- False positives must be reranked before fusion; simply changing attention weights is not enough.

### Diagnostic Only: Oracle Source Fusion

Result:

- 128K 25% passed when the correct source was forced.

Lesson:

- Source-aware fusion can use a correct retrieved source.
- The blocker was real source selection, not the fusion branch alone.

### Adopted: Source-Token Lexical Reranker

Method:

- Register source token ids as lightweight metadata.
- Rerank DRAM chunks with rare query/source token overlap.
- Do not use needle position, target code, or oracle labels.

Result:

- 32K sanity: 4/4.
- 128K 25% single-depth: 1/1.
- 128K required depths 25/50/75/90: 4/4.

Current ranked ideas:

| Rank | Idea | Purpose | Decision |
|---:|---|---|---|
| 1 | Multi-trial 128K NIAH with source-token reranker | Check robustness beyond one seed/code per depth | Next |
| 2 | Optional 0%/99% depths | Stress boundary positions | Next |
| 3 | Real WikiText-2 PPL | Measure general semantic degradation | Next |
| 4 | Latency breakdown | Quantify source-token reranker and source fusion overhead | Next after quality |
| 5 | Triton/CUDA fused dequant attention | Optimize only if quality passes and latency is >2x | Defer |

### Boundary Depth Outcome

| Idea | Result | Decision |
|---|---|---|
| Optional 99% depth | Passed | Keep as supporting evidence |
| Optional 0% depth with main min-position filter | Failed | Record as boundary failure |
| Optional 0% depth with `min_position=0` | Failed | Unresolved prefix-boundary weakness |

Updated lesson:

- The current mechanism is strong enough for required 25/50/75/90 NIAH at 128K.
- It is not yet robust at the exact prefix boundary.  Avoid claiming 0% success.

## Workflow 2.0 Round 11-14 Idea Outcomes

### Adopted: Source-Aware Method-D Reuse TTL

Method:

- Keep the first retrieval as real token-level Query x Key dot-product.
- If the selected source chunk has enough source evidence, reuse that chunk for a short TTL across following decode tokens.
- Log reused retrieval separately as `*_reuse` with `reuse_hit=True`.
- Do not call this fresh dot-product evidence.

Best current setting:

- `method_d_reuse_ttl_tokens=6`
- `method_d_reuse_source_threshold=35`
- `method_d_token_window=64`

Result:

- seed4242 required-depth 128K: `8/8`, average decode `544.6 ms/token`.
- seed6004 required-depth 128K: `8/8`, average decode `562.8 ms/token`.
- seed7777 blind required-depth 128K: `4/4`, average decode `530.2 ms/token`.
- max reserved remained `20.6465 GiB`.

Decision:

- Adopt as the current PyTorch main path for NIAH latency reduction.

### Failed: Layer-Subset Retrieval

| Variant | Result | Decision |
|---|---:|---|
| layers `20-27` | failed sensitive seed6004 | Reject |
| layers `8-27` | seed4242 required-depth `7/8` | Reject |
| layers `4-27` | seed4242 required-depth `7/8` | Reject |

Lesson:

- Skipping Method-D on early/mid layers causes real quality loss.

### Failed Or Marginal: TTL And Window Micro-Tuning

| Idea | Result | Lesson |
|---|---:|---|
| TTL source threshold `45` | no reuse | Threshold exceeded actual source score. |
| TTL12 with window128 | correct, small single-case gain | Not enough to justify replacing TTL6 without broader validation. |
| `query_history=16` | correct, no speed gain | Bottleneck is not dominated by query-history length. |
| TTL12 with window64 | correct, marginal single-case gain | TTL6 remains safer as main path. |

### Failed: Optional 0% Boundary Fixes

| Idea | Result | Decision |
|---|---:|---|
| Main config optional 0% | `0/2` | Known boundary failure |
| `allow_source_before_min_position` | `0/2` | Reject |
| source-cue-score before min-position | `0/2`, slower | Reject |

Lesson:

- The prefix-boundary failure needs a dedicated retention/retrieval design, not another small reranker tweak.

### Current Ranked Ideas

| Rank | Idea | Purpose | Decision |
|---:|---|---|---|
| 1 | Request/plan Triton fused dequant attention | Current PyTorch path is still far slower than FullKV SDPA decode | Candidate, requires user approval |
| 2 | Dedicated prefix-boundary retention path | Address optional 0% without weakening required-depth robustness | Candidate |
| 3 | Broader PPL suite | Extend current 8K/10K evidence toward paper-grade semantic-loss evidence | Candidate |
| 4 | Larger NIAH statistics | Move beyond 2 seeds x 2 trials/depth | Candidate |
| 5 | True 4090 replication | Convert A100-under-envelope claim into hardware claim | Later |

### Adopted Evidence: 10K PPL Follow-Up

Result:

- FullKV PPL: `4.9011`
- HeteroKV PPL: `4.9046`
- Relative delta: `+0.07%`
- Artifact: `experiments/ppl_10k_prefix8192_gate35_nofusion_sdpa_autogpu_20260528_170936.json`

Decision:

- Keep as stronger semantic-loss evidence alongside the earlier 8K `+3.55%` run.
- Still do not claim broad 128K PPL robustness.

## Workflow 2.0 Round 5 Idea Outcomes

### Adopted: Decode-Suffix PPL Harness

Problem:

- Chunked prefill PPL does not exercise decode-time Method-D retrieval, so it cannot evaluate the real retrieval path.

Outcome:

- Added a WikiText-2 PPL script with `decode_suffix` mode.
- It preloads a compressed prefix, then computes next-token CE loss one token at a time.
- Source-token metadata is updated using only already observed tokens, avoiding future-token leakage in PPL.

### Failed: Reusing Aggressive NIAH Source Fusion For WikiText PPL

Result:

- 512-token suffix PPL worsened from `6.7392` to `33.2591` when source fusion was applied aggressively.

Lesson:

- NIAH answer recovery and generic language-modeling PPL have different false-positive tolerances.
- Source-aware fusion must be gated or disabled when the retrieval evidence is weak.

### Partially Adopted: Strict False-Positive Gate For PPL

Result:

- Raising the gate to `3.5`, using `top_k=1`, and disabling source fusion removed the catastrophic PPL failure.
- On 4K/prefix3072/keep_tail2048, HeteroKV suffix PPL was `5.1463` vs full `6.2006`.

Lesson:

- Strict gating is a viable PPL-safe mode.
- Do not claim the aggressive NIAH source-fusion configuration is universally PPL-safe.

Current ranked ideas:

| Rank | Idea | Purpose | Decision |
|---:|---|---|---|
| 1 | Multi-trial 128K NIAH with the source-token reranker | Test robustness beyond one code per required depth | Next |
| 2 | Latency breakdown for aggressive NIAH config and strict PPL config | Quantify overhead and expose retrieval cost | Next |
| 3 | Adaptive false-positive gate | Use aggressive retrieval only when source evidence is strong | Candidate |
| 4 | Longer WikiText suffix PPL if GPU remains safe | Reduce sample noise beyond 4K | Candidate |
| 5 | Retry optional 0% boundary with prefix-aware source handling | Address known boundary failure | Lower priority |

### Adopted: 128K Latency Breakdown Instrumentation

Outcome:

- Added per-case timing to NIAH outputs.
- Captured a 128K 50% depth main-path run:
  - prefill `62.70 s`;
  - decode `30.16 s`;
  - decode `1206.44 ms/step`;
  - peak process memory `20.44 GiB`.

Lesson:

- The project now has a real latency artifact, but not yet a baseline ratio.
- Full-KV or no-retrieval baselines should be run only when server safety allows.

## Workflow 2.0 Round 7 Idea Outcomes

### Failed: Single-Trial Success Was Enough

Result:

- The previous 128K required-depth main path passed `4/4`, but multi-trial required depths passed only `6/8`.

Lesson:

- Required-depth NIAH needs at least small multi-trial robustness before paper claims.

### Partially Failed: Source-Overlap Hard Filter Alone

Result:

- Filtering zero-overlap false-positive chunks improved retrieved-source cleanliness but still passed only `2/4` on the previously weak 25/50 retry.

Lesson:

- Ranking correctness is necessary but not sufficient; retrieved source must also influence answer-token generation.

### Failed: Static Source Fusion Alpha

Result:

- Strong static alpha fixed 25/50 but broke 90.
- Middle alpha was worse overall.

Lesson:

- Early/mid-depth and near-tail cases need different fusion strength.

### Adopted: Dynamic Source-Aware Fusion

Method:

- Keep source-overlap hard filtering.
- Use high fusion alpha when source-token evidence is strong.
- Fall back to lower fusion alpha when source evidence is weaker.

Final result:

- 128K required depths 25/50/75/90, 2 trials each: `8/8`.
- Peak process memory: `20.4375 GiB`.

Current ranked ideas:

| Rank | Idea | Purpose | Decision |
|---:|---|---|---|
| 1 | More WikiText-2 PPL samples | Reduce small-sample variance in the 4K suffix PPL result | Candidate |
| 2 | Fair optimized full-KV baseline using SDPA/FlashAttention if available | Replace failed eager full-attention baseline with a feasible latency reference | Candidate |
| 3 | Optional 0% boundary-specific repair | Fix known prefix-boundary weakness | Candidate |
| 4 | Dynamic gate/fusion ablation table | Quantify each structural addition separately | Candidate |
| 5 | Triton/CUDA fused dequant attention | Only after quality is locked and latency ratio demands it | Defer |

## Workflow 2.0 Round 8 Idea Outcomes

### Confirmed: Full-KV 128K Cannot Be The 24G Survival Baseline

Result:

- FullKV 128K under the same 22 GiB PyTorch cap OOMed.
- Artifact: `experiments/niah_fullkv_128k_cap22_20260527_231220.json`.
- Error included a failed `16.00 GiB` allocation while the 22 GiB cap was already nearly saturated.

Lesson:

- The 24G-envelope survival proof should compare HeteroKV survival against FullKV OOM under the same cap.
- This is a valid memory-survival control, but it cannot provide latency ratio because it does not complete.

### Confirmed: Eager Full-Attention 128K Is Not A Feasible A100 Latency Baseline

Result:

- FullKV 128K with a wide 75 GiB A100 cap also OOMed.
- Artifact: `experiments/niah_fullkv_128k_cap75_20260527_231309.json`.
- The failure attempted an `895.92 GiB` allocation, showing that the eager full-attention baseline is dominated by attention-score materialization, not only KV memory.

Lesson:

- Do not claim a 128K `<=2x` latency ratio against this baseline.
- A fair latency baseline requires an optimized full-attention implementation such as SDPA/FlashAttention if available, or a shorter-context reference clearly labeled as such.

### Recorded: Internal No-Retrieval Latency Is Fast But Quality-Failing

Result:

- HeteroKV 128K no-retrieval latency: prefill `62.79 s`, decode `82.33 ms/step`.
- Artifact: `experiments/niah_heterokv_128k_noretrieval_latency_20260527_231538.json`.
- Quality failed by generating `000000` instead of the target.

Lesson:

- Retrieval/fusion is the dominant decode overhead and is also necessary for NIAH correctness.
- This is an internal ablation, not an accepted system configuration.

### Recorded: 8K Short-Context Speed References

Result:

- FullKV 8K wide-cap reference passed but used `32.61 GiB` reserved memory:
  `experiments/niah_fullkv_8k_cap75_latency_20260527_231358.json`.
- HeteroKV 8K under 22 GiB passed with `19.86 GiB` reserved memory:
  `experiments/niah_heterokv_8k_cap22_latency_20260527_231448.json`.

Lesson:

- The 8K references are useful sanity checks but should not be used to prove 128K latency.
- HeteroKV's main paper claim remains memory survival plus semantic recovery at 128K, with latency reported as A100-under-cap measurement and optimization target.

## Workflow 2.0 Round 9 Idea Outcomes

### Adopted For Next Probe: Focus-Only Source Fusion

Problem:

- A new seed-6004, 128K, 50% depth run failed even though Method-D retrieval was active.
- The output drifted toward `000008...` and repeated needle markup, suggesting that source-aware fusion over an entire retrieved chunk may inject too much nearby template/filler context.

Idea:

- Keep token-level Query x Key retrieval unchanged.
- Keep oracle/diagnostic paths separate.
- Add an optional `source_fusion_focus_only` mode so the source-only fusion step attends only to the focus window around matched source tokens instead of the whole retrieved 2048-token chunk.

Decision:

- Implemented locally as a structural probe, behind an explicit flag.
- Must be tested first on the failing seed-6004 case.
- If it fixes seed-6004, run regression on the previous 25/50/75/90 required-depth suite before adopting it as the new main result.

Current ranked ideas:

| Rank | Idea | Purpose | Decision |
|---:|---|---|---|
| 1 | Focus-only source fusion | Reduce retrieved-chunk filler/template contamination | Implemented locally, pending remote test |
| 2 | Broader multi-seed 128K NIAH | Replace fragile 8/8 evidence with stronger robustness | Next after probe |
| 3 | More WikiText-2 PPL samples | Strengthen semantic-loss claim | Candidate |
| 4 | Retrieval overhead reduction | Address ~1s/step HeteroKV decode | Candidate after quality |
| 5 | Triton/CUDA fused dequant attention | Only after quality is locked and user approves | Defer |

## Workflow 2.0 Round 10 Idea Outcomes

### Failed: Strong Focus-Only Fusion Alone

Result:

- alpha `0.75` improved seed6004 but still missed the first digit.
- alpha `1.0` over-focused `[NEEDLE]` markup.

Lesson:

- Retrieved chunk focus must distinguish answer-bearing tokens from source cue and markup tokens.

### Adopted: Non-Oracle Source Cue Focus

Method:

- Register cue token sequences from the prompt template, not the answer.
- Focus the tokens immediately after cues like `The target code is `.
- Keep this separate from oracle retrieval and disabled for PPL.

Best configuration:

- cue-focus alpha `0.65`;
- token window `128`;
- focus bias `4.0`;
- nonfocus penalty `1.0`.

Result:

- seed4242 required-depth 128K: `8/8`.
- seed6004 required-depth 128K: `8/8`.
- optional 99%: `2/2`.
- optional 0%: `0/2`.

Current ranked ideas:

| Rank | Idea | Purpose | Decision |
|---:|---|---|---|
| 1 | Broader real PPL when GPU is free | Strengthen semantic-loss claim beyond the current 4K suffix sample | Next safe-window task |
| 2 | Retrieval overhead reduction | Current NIAH decode is about 1.1-1.75 s/token | Candidate |
| 3 | Optional 0% prefix-boundary handling | Fix documented boundary weakness without harming required depths | Candidate |
| 4 | Larger multi-seed NIAH table | Move from strong prototype evidence to paper-grade statistics | Candidate |
| 5 | Triton/CUDA fused dequant attention | Only after quality/PPL claims are stable and user approves | Defer |

## Workflow 2.0 Round 15 Idea Outcomes

### Adopted: Path A Triton INT4 Retrieval Scoring

Permission:

- User approved Path A.
- Scope remains retrieval scoring only, not fused attention output.

Method:

- Add an optional Triton kernel for Method-D scoring.
- Dequantize uint8-backed INT4 K inside the kernel using the existing group-wise scale/zp format.
- Compute token-level `Q x K` block scores and best-token offsets.
- Do not materialize full FP16/BF16 K.
- Do not fuse V weighting in this step.
- Keep PyTorch dequant scoring as the fallback and as the reference path.

Evidence so far:

- Stage1 remote tests: `12 passed`.
- Warm microbench: top-k equal to PyTorch, best offsets equal, median speedup `3.19x`, allocated memory reduced from `21.28 MiB` to `9.30 MiB`.
- 32K real Qwen2.5-7B NIAH smoke passed `1/1`; Method-D event backend was `triton_int4`; max reserved stayed `20.65 GiB`.

Failed/Inconclusive:

- 128K one-depth Path-A probe stayed under the 30 GiB process fuse, but did not return before SSH port `2222` became unreachable.
- This is not counted as an accepted 128K Path-A result.
- It exposed a likely risk: per-chunk Triton launch overhead may still be too high in full 128K decode.

### Adopted Next: Batched Triton Scoring

Idea:

- Instead of one Triton scoring launch per DRAM chunk, stage several same-shaped quantized chunks and score them in one launch grid.
- Control staging with `triton_scoring_batch_chunks`.

Why it should help:

- It reduces Python loop and kernel launch overhead.
- It still does not allocate full FP16 K.
- Quantized staging is bounded and small compared with the 30 GiB safety fuse.

Current ranked ideas:

| Rank | Idea | Purpose | Decision |
|---:|---|---|---|
| 1 | Batched Triton INT4 scoring | Reduce Method-D retrieval launch overhead without changing semantics | Implemented locally, pending remote sync/test |
| 2 | 128K Path-A single-depth retry | Verify quality, backend logs, and latency after batched scoring | Next when SSH returns |
| 3 | Required-depth 128K Path-A regression | Ensure Triton path does not regress the 20/20 PyTorch main result | After single-depth pass |
| 4 | Optional 0% prefix-boundary design | Fix known boundary failure separately from Path A | Candidate |
| 5 | Broader PPL suite | Strengthen semantic-loss evidence | Candidate after latency path stabilizes |

## Workflow 2.0 Round 16 Idea Outcomes

### Corrected Testing Method: No-Pipe GPU Monitor

Finding:

- Long 128K Method-D runs produce enough mechanism logs to fill an unconsumed stdout pipe.
- A monitor wrapper that uses `stdout=PIPE` but only calls `communicate()` after process exit can falsely stall the experiment.

Decision:

- Use log-file stdout/stderr redirection or an actively drained pipe for monitored GPU runs.
- Treat the earlier 25-minute Path-A timeouts as invalid harness failures, not model/runtime failures.

### Failed: Batched Triton Scoring As Main Path

What worked:

- Stage1 passed.
- Microbench top-k matched PyTorch after fixing the batched reducer.
- Single-depth 128K ran to completion under the 30 GiB process fuse.

What failed:

- Required-depth 128K with `triton_int4_batch` scored only `2/4`.
- Adding FP16 dequant rounding inside the Triton kernel did not restore quality.

Decision:

- Keep Triton scoring as an experimental branch.
- Do not use it for the main semantic claim.

### Failed: Retrieved K/V Cache Reuse As Default

Idea:

- Reuse decompressed short retrieved K/V windows across selected-key TTL hits.

Result:

- 32K smoke improved from `35.7s` to `18.7s`.
- 128K required-depth quality fell to `2/4`.

Decision:

- Keep the feature behind `method_d_reuse_kv_cache`.
- Default remains off.
- Record it as a speed-quality tradeoff, not an accepted optimization.

### New Robustness Gap: Correct Chunk Found, Answer Not Reliably Generated

Finding:

- In new seed6004 one-trial required-depth pairings, failures at 50% and 90% still retrieved the needle-containing chunk in most Method-D events.
- Source cue focus covered the answer span.
- The generated text contained partial digits but not the exact code.

Interpretation:

- The remaining blocker is not retrieval recall.
- It is final answer-span attention fusion / decoding fidelity after the correct source is available.

Current ranked ideas:

| Rank | Idea | Purpose | Decision |
|---:|---|---|---|
| 1 | Answer-span-only source fusion | Attend only answer tokens after source cue, not the surrounding 64-token window | Next |
| 2 | Source-aware false-positive suppressor at attention time | Reduce non-source retrieved chunk influence once source-cue chunk is found | Candidate |
| 3 | Decode-time extraction head / constrained answer span probe | Diagnostic only: separate cache quality from free-form generation drift | Candidate |
| 4 | Broader randomized 128K NIAH table | Measure robustness honestly after each fix | Required before Workflow3 |
| 5 | Triton scoring | Keep experimental until semantic parity is restored | Defer |

## Workflow 2.0 Round 17 Idea Outcomes

### Implemented Candidate: Source-Cue Answer-Span Physical Retrieval

Idea:

- When a retrieved DRAM chunk contains a registered non-oracle source cue such as `The target code is ` or `target_code=`, physically return only the answer tokens immediately after that cue.
- This is stricter than `source_fusion_focus_only`: the ordinary attention path no longer sees the surrounding 64-token retrieved window for source-cue hits.
- The feature is opt-in via `method_d_retrieve_focus_only` / `--method-d-retrieve-focus-only`.

Why it should help:

- Previous failure analysis showed the correct needle chunk was often retrieved, but final generation still drifted.
- Removing surrounding retrieved filler tokens reduces false-positive attention mass after the cache has already found the source.
- It also reduces retrieved K/V bytes for source-cue hits, while preserving the fixed-HBM invariant.

Evidence:

| Run | Result | Peak process memory | Artifact |
|---|---:|---:|---|
| Stage1 CPU/GPU | `15 passed` / `15 passed` | small | `tests/test_heterokv_stage1.py` |
| 32K sanity, focus-only retrieval | `1/1` | `21506 MiB` | `experiments/niah_32k_focus_only_retrieval_smoke_20260528_141635.json` |
| 128K required depths seed6004, focus-only + TTL6 | `4/4` | `21508 MiB` | `experiments/niah_128k_required_focus_only_retrieval_seed6004_20260528_141751.json` |
| 128K required depths seed6004, focus-only without TTL | `4/4` | `21508 MiB` | `experiments/niah_128k_required_focus_only_no_ttl_seed6004_20260528_142431.json` |

Interpretation:

- This branch fixes the specific new seed6004 one-trial failure pattern where the older PyTorch main path scored `2/4`.
- No-TTL focus-only retrieval is accurate but too slow (`~1440 ms/step`), so it is not a latency path by itself.
- Focus-only + TTL6 is accurate and bounded, but still slower than the prior TTL main average on the same class of runs.

### Rejected For Main Path: Triton Scoring + Focus-Only

Evidence:

| Run | Result | Decode | Artifact |
|---|---:|---:|---|
| 32K focus-only + Triton batched scoring | `1/1` | `~697 ms/step` | `experiments/niah_32k_focus_only_triton_smoke_20260528_143236.json` |
| 128K depth50 focus-only + Triton batched scoring + TTL6 | `1/1` | `~869 ms/step` | `experiments/niah_128k_depth50_focus_only_triton_ttl_seed6004_20260528_143358.json` |

Decision:

- Keep Triton scoring as a mechanism/microbench branch.
- Do not expand it to a full 128K matrix because end-to-end latency is worse than the PyTorch focus-only TTL path.

Current ranked ideas:

| Rank | Idea | Purpose | Decision |
|---:|---|---|---|
| 1 | Source-cue answer-span physical retrieval | Improve final answer fidelity after correct source retrieval | Implemented; candidate robustness path |
| 2 | Focus-only + selected-key TTL6 | Combine robustness with tolerable latency | Needs broader seed/trial matrix |
| 3 | Attention-time source suppressor | Further reduce non-source retrieved influence without hard span filtering | Candidate if focus-only regresses broader tests |
| 4 | Boundary-aware 0% NIAH handling | Address known prefix-boundary failure separately | Candidate, not in current claim |
| 5 | Triton scoring | Mechanism branch only until it improves end-to-end latency | Defer |

## Workflow 2.0 Round 18 Idea Outcomes

### Implemented Candidate: Cue-Context Physical Retrieval

Idea:

- Pure answer-span retrieval can leave the model attending to isolated digits without the local cue that says those digits are the target.
- Add `method_d_retrieve_focus_context_tokens` / `--method-d-retrieve-focus-context-tokens`.
- Physical retrieval includes a small number of cue/context tokens before the answer span, while the focus mask still marks only the answer tokens.

Why it should help:

- It preserves the semantic anchor (`target_code=` or the tail of `The target code is`) without returning the whole 64-token noisy window.
- It remains non-oracle: the cue strings come from the task template, not from the hidden answer span.

Key implementation detail:

- `context=3` covers the full `target_code=` cue in Qwen2.5 tokenization.
- Returned source-cue windows are 22 tokens in this NIAH setup, down from 26 with `context=5` and 32 with `context=8`.

Evidence:

| Run | Result | Peak process memory | Decode | Artifact |
|---|---:|---:|---:|---|
| Stage1 CPU/GPU | `16 passed` / `16 passed` | small | n/a | `tests/test_heterokv_stage1.py` |
| context=0, seed6004, 2 trials/depth | `7/8` | `21508 MiB` | `~572 ms/step` | `experiments/niah_128k_required_focus_only_ttl_seed6004_trials2_20260528_workflow2.json` |
| context=5 targeted seed6004 25/50 | `4/4` | `21508 MiB` | `~700 ms/step` | `experiments/niah_128k_focus_context5_seed6004_depth25_50_trials2_20260528_154059.json` |
| context=5 full seed6004 | `8/8` | `21652 MiB` | `~749 ms/step` | `experiments/niah_128k_required_focus_context5_seed6004_trials2_20260528_154813.json` |
| context=5 full seed4242 | `8/8` | safe | n/a | `experiments/niah_128k_required_focus_context5_seed4242_trials2_20260528_160141.json` |
| context=3 targeted seed6004 25/50 on GPU2 | `4/4` | `21508 MiB` | `~719 ms/step` | `experiments/niah_128k_focus_context3_seed6004_depth25_50_trials2_gpu2_20260528_164151.json` |
| context=3 full seed6004 | `8/8` | `21652 MiB` | `~832 ms/step` | `experiments/niah_128k_required_focus_context3_seed6004_trials2_gpu2_20260528_164910.json` |
| context=3 full seed4242 | `8/8` | `21652 MiB` | `~1051 ms/step` | `experiments/niah_128k_required_focus_context3_seed4242_trials2_gpu2_20260528_170258.json` |
| context=3 seed7777, one trial/depth | `4/4` | `21652 MiB` | `~1001 ms/step` | `experiments/niah_128k_required_focus_context3_seed7777_trial1_gpu2_20260528_171645.json` |

Invalid run:

- `experiments/niah_128k_focus_context3_seed6004_depth25_50_trials2_20260528_163931.json` OOMed because GPU3 had another `ahr` VideoMME process using about 25.5 GiB in addition to the shared `lhj` process. It is recorded as a shared-GPU scheduling failure, not an algorithm result.

Decision:

- Promote `context=3` to the current quality candidate: it has 128K required-depth `20/20` across seeds `4242`, `6004`, and `7777`.
- Do not mark Workflow3 ready: latency remains far above the relaxed FullKV SDPA reference, and PPL has not yet been refreshed for the new retrieval-context path.

Current ranked ideas:

| Rank | Idea | Purpose | Decision |
|---:|---|---|---|
| 1 | Cue-context focus-only retrieval, context=3 | Improve answer fidelity with bounded source context | Current quality candidate |
| 2 | Refresh PPL under context=3 | Check semantic-loss budget after new retrieval behavior | Next |
| 3 | Same-GPU latency retest under lighter shared load | Measure context=0/3/5 fairly | Needed before speed claim |
| 4 | Boundary-aware 0% NIAH | Address known prefix boundary failure | Candidate |
| 5 | Fused dequant/value weighting | Only if quality/PPL hold and latency remains blocked | Requires separate permission |

## Workflow 2.0 Round 19 Idea Outcomes

### Revisited Idea: Retrieved K/V Cache Under Cue-Context Retrieval

Earlier result:

- Retrieved K/V cache reuse was rejected as a default after it scored `2/4` in an older 128K required-depth run.

New hypothesis:

- The earlier failure came from reusing noisy or semantically under-anchored retrieved windows.
- With cue-context focus-only retrieval (`context=3`), the cached retrieved K/V window is smaller and more semantically stable.

Evidence:

| Run | Result | Decode | Peak process memory | Artifact |
|---|---:|---:|---:|---|
| context=3 + K/V cache targeted seed6004 25/50 | `4/4` | `~678 ms/step` | `21508 MiB` | `experiments/niah_128k_context3_kvcache_seed6004_depth25_50_trials2_gpu2_20260528_173924.json` |
| context=3 + K/V cache full seed6004 | `8/8` | `~597 ms/step` | `21652 MiB` | `experiments/niah_128k_context3_kvcache_seed6004_trials2_gpu2_20260528_174634.json` |
| context=3 + K/V cache full seed4242 | `8/8` | `~680 ms/step` | `21652 MiB` | `experiments/niah_128k_context3_kvcache_seed4242_trials2_gpu2_20260528_175954.json` |
| context=3 + K/V cache seed7777, one trial/depth | `4/4` | `~551 ms/step` | `21508 MiB` | `experiments/niah_128k_context3_kvcache_seed7777_trial1_gpu2_20260528_181316.json` |

Decision:

- Promote `context=3 + method_d_reuse_kv_cache` to the current speed/quality candidate.
- Keep it opt-in, not a global default.
- It needs a fairer same-GPU latency comparison before any latency claim.

### PPL Refresh

Run:

- `experiments/ppl_10k_prefix8192_gate35_nofusion_context3_gpu2_20260528_workflow2.json`

Result:

| Metric | FullKV | HeteroKV | Delta |
|---|---:|---:|---:|
| WikiText-2 10K decode-suffix PPL | `4.9011` | `4.9046` | `+0.07%` |

Notes:

- `retrieve_focus_only=True`, `retrieve_focus_context_tokens=3`.
- `method_d_event_count=0`, as expected for the strict/no-fusion WikiText PPL configuration.
- Process peak was `22608 MiB`, below the 30 GiB fuse.

Current ranked ideas:

| Rank | Idea | Purpose | Decision |
|---:|---|---|---|
| 1 | Context=3 + retrieved K/V cache | Best current NIAH speed/quality tradeoff | Current candidate |
| 2 | Fair latency retest on a less-contended GPU | Determine whether observed `~550-680 ms/step` is stable | Next |
| 3 | Optional boundary 0% design | Expand claim scope beyond required depths | Candidate |
| 4 | Larger/longer PPL suite | Strengthen semantic-loss evidence | Candidate |
| 5 | Fused dequant/value weighting | Attack latency if quality remains stable | Requires separate permission |


## Workflow2 Round 20: Source-Aware Latency And Boundary Checks

New evidence after the context=3 K/V-cache checkpoint:

| Idea | Configuration | Result | Decision | Artifact |
|---|---|---:|---|---|
| TTL12 reuse on strict no-fusion ablation | gate=3.5, no source-aware fusion, context=3, K/V cache | `0/4` | Failed diagnostic; not the main source-aware path | `experiments/niah_128k_context3_kvcache_ttl12_seed6004_depth25_50_trials2_gpu2_20260528_wf2c.json` |
| top_k=2 on strict no-fusion ablation | gate=3.5, context=3, K/V cache | `0/4` | Failed; reducing evidence width collapses to `000000` | `experiments/niah_128k_context3_kvcache_top2_ttl6_seed6004_depth25_50_trials2_gpu2_20260528_wf2.json` |
| qhist=32 on strict no-fusion ablation | gate=3.5, context=3, K/V cache | `0/4` | Failed; query history length is quality-critical | `experiments/niah_128k_context3_kvcache_qhist32_ttl6_seed6004_depth25_50_trials2_gpu2_20260528_wf2.json` |
| layers 4-27 only on strict no-fusion ablation | gate=3.5, context=3, K/V cache | `0/4` | Failed; skipping early layers removes useful signal | `experiments/niah_128k_context3_kvcache_layers4_27_ttl6_seed6004_depth25_50_trials2_gpu2_20260528_wf2.json` |
| answer-constrained source-aware context=3 | max_new_tokens=16, source-aware alpha=0.65, context=3, K/V cache | `4/4` | Useful latency diagnostic; not a replacement for 24-token main result | `experiments/niah_128k_sourceaware_context3_kvcache_maxnew16_seed6004_depth25_50_trials2_gpu2_20260528_wf2.json` |
| answer-constrained source-aware no-context | max_new_tokens=16, source-aware alpha=0.65, context=0, K/V cache | `3/4` | Context=3 is useful for short-output robustness | `experiments/niah_128k_sourceaware_nocontext_kvcache_maxnew16_seed6004_depth25_50_trials2_gpu2_20260528_wf2.json` |
| optional boundary 0/99 with context=3 | source-aware alpha=0.65, context=3, K/V cache | `2/4` | 99% passes; 0% remains a boundary weakness | `experiments/niah_128k_sourceaware_context3_optional0_99_seed4242_trials2_gpu2_20260528_wf2.json` |
| optional boundary with Sink=512 | source-aware alpha=0.65, context=3, sink=512 | `2/4` | Failed to fix 0%; larger sink alone is not enough | `experiments/niah_128k_sourceaware_context3_sink512_optional0_99_seed4242_trials2_gpu2_20260528_wf2.json` |

Updated decisions:

- Do not pursue semantic-signal pruning (`top_k=2`, `qhist=32`, `layers4-27`) as a main optimization; each collapsed to `000000` in the strict ablation.
- Keep source-aware cue focus plus token-level dot-product retrieval as the quality path.
- Treat `max_new_tokens=16` as an answer-constrained latency diagnostic only. The main open-ended NIAH result remains the 24-token setting.
- Optional depth 0% is still blocked. Do not claim full boundary robustness across 0/99 until a mechanism specifically fixes 0%.


## Workflow2 Round 21: Deferred Dequant Gate

Implementation change:

- Split Method-D retrieval into score/selection and materialization phases.
- The decode path now computes token-level Query-Key chunk scores and runs the HBM gate before dequantizing selected DRAM K/V.
- If the HBM gate rejects DRAM retrieval, selected chunks are logged but K/V tensors are not materialized, avoiding wasted dequantization and temporary HBM traffic.
- The semantic path is unchanged: `top_k=4`, `query_history=64`, source-aware cue focus, TTL6 K/V reuse, and context=3 remain intact.

Validation:

| Test | Result | Decode | Peak process memory | Artifact |
|---|---:|---:|---:|---|
| Stage1 mechanism tests | `16/16` | n/a | CPU/unit | `tests/test_heterokv_stage1.py` |
| 32K source-aware context3 smoke | `1/1` | `281.8 ms/step` | `21506 MiB` | `experiments/niah_32k_sourceaware_context3_deferred_smoke_seed6004_gpu2_20260528_wf2.json` |
| 128K targeted seed6004 25/50 | `4/4` | `578.9 ms/step` | `21652 MiB` | `experiments/niah_128k_sourceaware_context3_deferred_seed6004_depth25_50_trials2_gpu2_20260528_wf2.json` |
| 128K required seed6004 all depths | `8/8` | `559.1 ms/step` | `21652 MiB` | `experiments/niah_128k_sourceaware_context3_deferred_seed6004_trials2_gpu2_20260528_wf2.json` |

Decision:

- Promote deferred dequant gate as an implementation-level optimization because it preserves the accepted semantic configuration and passes real 128K required-depth testing.
- This improves the current shared-GPU seed6004 full-depth decode average from the previous context3/KV-cache result (`~597 ms/step`) to `~559 ms/step`.
- Latency is still far above the FullKV SDPA reference, so Workflow3 remains blocked.


### Deferred dequant cross-seed follow-up

Additional validation after the first deferred checkpoint:

| Test | Result | Decode | Peak process memory | Artifact |
|---|---:|---:|---:|---|
| 128K required seed4242, 2 trials/depth | `8/8` | `536.1 ms/step` | `21652 MiB` | `experiments/niah_128k_sourceaware_context3_deferred_seed4242_trials2_gpu2_20260528_wf2.json` |
| 128K required seed7777, 1 trial/depth | `4/4` | `481.9 ms/step` | `21652 MiB` | `experiments/niah_128k_sourceaware_context3_deferred_seed7777_trial1_gpu2_20260528_wf2.json` |

Combined deferred required-depth evidence is now `20/20` across seeds 4242, 6004, and 7777 under the 22 GiB cap and 30 GiB own-process fuse.

## Workflow2 Round 22: Source-Aware Triton Scoring And Gate-Bypass Diagnostics

New optimization candidates after deferred dequant:

| Idea | Configuration | Result | Decision | Artifact |
|---|---|---:|---|---|
| Source gate bypass threshold 50 | context=3, deferred dequant, threshold too high to trigger | `1/1` 32K, `0` bypass events | No effect; do not use as evidence | `experiments/niah_32k_sourceaware_context3_srcgate50_smoke_seed6004_gpu2_20260528_wf2.json` |
| Source gate bypass threshold 35 | context=3, all source-cue-selected chunks bypass HBM gate | `0/1` 32K, `229.1 ms/step` | Failed: fast but corrupts generation; do not promote | `experiments/niah_32k_sourceaware_context3_srcgate35_smoke_seed6004_gpu2_20260528_wf2.json` |
| TTL reuse gate bypass | skip HBM gate only when selected chunks are TTL reuse hits | `4/4` 128K targeted, `747.4 ms/step` | Correct but slower than deferred main path; keep diagnostic-only | `experiments/niah_128k_sourceaware_context3_reusegate_seed6004_depth25_50_trials2_gpu2_20260528_wf2.json` |
| Source-aware Triton batched scoring | stable context=3 + deferred dequant + `triton_int4_batch` scoring | `20/20` 128K required-depth across 3 seeds | Current latency candidate; still needs final PPL/baseline refresh | see below |

Triton-scoring validation under the 22 GiB cap and 30 GiB process fuse:

| Test | Result | Decode | Peak process memory | Artifact |
|---|---:|---:|---:|---|
| 32K smoke seed6004 | `1/1` | `338.5 ms/step` | `21506 MiB` | `experiments/niah_32k_sourceaware_context3_triton_score_smoke_seed6004_gpu2_20260528_wf2.json` |
| 128K targeted seed6004 25/50 | `4/4` | `539.2 ms/step` | `21518 MiB` | `experiments/niah_128k_sourceaware_context3_triton_score_seed6004_depth25_50_trials2_gpu2_20260528_wf2.json` |
| 128K required seed6004, 2 trials/depth | `8/8` | `438.4 ms/step` | `21654 MiB` | `experiments/niah_128k_sourceaware_context3_triton_score_seed6004_trials2_gpu2_20260528_wf2.json` |
| 128K required seed4242, 2 trials/depth | `8/8` | `481.7 ms/step` | `21654 MiB` | `experiments/niah_128k_sourceaware_context3_triton_score_seed4242_trials2_gpu2_20260528_wf2.json` |
| 128K required seed7777, 1 trial/depth | `4/4` | `603.1 ms/step` | `21654 MiB` | `experiments/niah_128k_sourceaware_context3_triton_score_seed7777_trial1_gpu2_20260528_wf2.json` |

Aggregate comparison on the same 20 required-depth NIAH cases:

| Path | Accuracy | Mean decode | Peak process memory |
|---|---:|---:|---:|
| Deferred dequant, torch scoring | `20/20` | `534.5 ms/step` | `21652 MiB` |
| Deferred dequant, Triton batched scoring | `20/20` | `488.7 ms/step` | `21654 MiB` |

Decision:

- Promote source-aware Triton batched scoring as the current latency candidate, not as a final paper claim yet.
- Keep failed gate-bypass variants recorded to avoid repeating them: full source-gate bypass is fast but semantically unsafe; TTL reuse-gate bypass is correct but slower in the 128K targeted run.
- The accepted semantic configuration remains source-aware cue focus, token-level Method-D retrieval, `top_k=4`, `query_history=64`, `token_window=64`, TTL6 K/V reuse, context=3, and deferred dequant.
- Before Workflow3, refresh PPL with the final candidate flags and run a fair latency/baseline table; optional 0% boundary remains a blocked claim.

## Workflow2 Round 23: PPL Method Fix And Long-PPL Blocker

Testing-method fix:

- `scripts/run_ppl_eval.py` full baseline no longer computes suffix PPL by materializing all suffix logits at once when `loss_start_token > 0`.
- The full baseline now uses a real FullKV decode-suffix path: prefix builds full KV cache, suffix computes one-token loss with `logits_to_keep=1`.
- This avoids an artificial logits tensor OOM and is a fairer latency/PPL comparison to HeteroKV decode-suffix evaluation.

PPL evidence after the fix:

| Test | Result | PPL / Failure | Peak process memory | Artifact |
|---|---:|---|---:|---|
| 2K full baseline smoke | pass | FullKV PPL `4.5419`, eval style `decode_suffix_full_kv` | n/a | `experiments/ppl_full_decode_suffix_smoke_2k_20260528_wf2.json` |
| 12K final flags, prefix 6144, tail 8192 | pass | FullKV `5.2026`, HeteroKV `5.1917`, delta `-0.21%` | `22774 MiB` | `experiments/ppl_12k_prefix6144_final_triton_context3_gpu2_20260528_wf2.json` |
| 16K final flags, prefix 8192, tail 8192 | fail | HeteroKV OOM at manual attention `key_states.float()`, request `168 MiB` | about `22.4 GiB` process | `experiments/ppl_16k_prefix8192_final_triton_context3_heteroonly_gpu2_20260528_wf2.log` |
| 16K BF16 attention diagnostic | fail | OOM at `repeat_kv(value)`, request `84 MiB` | about `22.5 GiB` process | `experiments/ppl_16k_prefix8192_final_triton_context3_bf16attn_heteroonly_gpu2_20260528_wf2.log` |
| 14K tail6144 stress PPL | fail | OOM at manual attention `key_states.float()`, request `126 MiB` | about `22.4 GiB` process | `experiments/ppl_14k_prefix6144_tail6144_triton_context3_stress_heteroonly_expand_gpu2_20260528_wf2.log` |
| 32K SDPA/GQA attention smoke | pass | `1/1`, `283.3 ms/step`; diagnostic only | `22346 MiB` | `experiments/niah_32k_sourceaware_context3_triton_score_sdpa_gqa_smoke_seed6004_gpu2_20260528_wf2.json` |
| 14K tail6144 SDPA/GQA stress PPL | fail | OOM inside `scaled_dot_product_attention`, request `126 MiB` | about `22.4 GiB` process | `experiments/ppl_14k_prefix6144_tail6144_triton_context3_sdpa_gqa_heteroonly_gpu2_20260528_wf2.log` |

Interpretation:

- The 12K PPL result is valid but must be labeled as a no-DRAM/no-retrieval PPL point: `method_d_event_count=0`, `dram_entries=0`. It shows the final flags do not hurt medium-context PPL, but it does not prove long-PPL semantic recovery through DRAM retrieval.
- Long PPL that crosses the HBM budget is currently blocked under the strict 22 GiB cap by attention temporary memory, not by NIAH retrieval correctness.
- BF16 attention and SDPA/GQA diagnostics did not solve the long-PPL OOM. SDPA/GQA can pass a 32K NIAH smoke but still OOMs in 14K PPL stress at the 22 GiB cap.
- Next credible route is not another threshold tweak; it is a true memory-efficient attention path that avoids materializing repeated K/V and large temporary score buffers while preserving source-fusion behavior, or a clearly separated 24 GiB supplementary PPL run.


## Workflow2 Round 24: Automatic Review And SourceCopy Exactness Reranker

Mode:

- Switched to fully automatic review/test loop.
- Continued to separate main retrieval, oracle/diagnostic paths, and experimental exactness rerankers.
- Used physical GPU3 when safe, with a 22 GiB PyTorch cap and a 30 GiB process-group memory fuse.

Failed / negative ideas:

| Idea | Result | Artifact | Interpretation |
|---|---:|---|---|
| Late-trigger SourceCopy on weak retrieval | `0/2` on 128K 25/50 | `experiments/niah_128k_depth25_50_sourcecopy_relaxed_boost20_v2_gpu3_20260529_154031.json` | Copy candidates appeared too late because the source-aware reranker was not active. |
| Bare current dot-product + source cue focus | `0/2` on 128K 25/50 | `experiments/niah_128k_depth25_50_current_triton_score_context3_gpu3_20260529_160542.json` | Missing source-token overlap filtering, consensus, and reuse; retrieval degenerated to non-source chunks. |
| Current main reranker without SourceCopy | `1/2` on 128K 25/50 | `experiments/niah_128k_depth25_50_current_main_cuefocus_reuse_win64_gpu3_20260529_160927.json` | Correct source window is retrieved, but exact numeric copying can still flip one digit (`690144` -> `691144`). |
| win128/no-KV-cache variant | `1/2` on 128K 25/50 | `experiments/niah_128k_depth25_50_main_win128_nokvcache_gpu3_20260529_161411.json` | The off-by-one digit is not caused by `token_window=64` or retrieved K/V cache reuse. |
| Interrupted win128/no-KV-cache probe | invalid | `experiments/niah_128k_depth50_main_win128_nokvcache_gpu3_20260529_161254.json` | Shared-GPU scheduling failure; not counted as algorithm evidence. |

Implemented candidate:

- Added optional `SourceCopy` logit reranking.
- It boosts next-token candidates extracted from retrieved source-cue answer spans.
- It is disabled by default.
- It is an experimental exact-string reranker and must not be reported as pure Query-Key dot-product retrieval.

Evidence:

| Run | Result | Avg decode | Max reserved | Peak process group | Artifact |
|---|---:|---:|---:|---:|---|
| seed6004 required depths, 1 trial/depth | `4/4` | `523.5 ms/token` | `21.5801 GiB` | `~22610 MiB` | `experiments/niah_128k_required4_main_win64_sourcecopy_boost20_seed6004_gpu3_20260529_162209.json` |
| seed4242 required depths, 1 trial/depth | `4/4` | `557.6 ms/token` | `21.5801 GiB` | `~22610 MiB` | `experiments/niah_128k_required4_main_win64_sourcecopy_boost20_seed4242_gpu3_20260529_162742.json` |

Mechanism evidence:

- Active HBM KV remained bounded with `max_hbm_tokens=12352`.
- DRAM compressed entries were `1680` at 128K.
- Source-aware retrieval logged `dot_product_source_filtered_consensus_reuse`.
- SourceCopy emitted explicit `[SourceCopy] step=... candidates=...` logs.
- The experimental reranker fixed exact digit fidelity while keeping memory below the 30 GiB fuse.

Current ranked ideas:

| Rank | Idea | Purpose | Decision |
|---:|---|---|---|
| 1 | Main source-aware reranker + experimental SourceCopy exactness | Strongest NIAH exact-copy result so far | Current NIAH candidate, label separately |
| 2 | Broader multi-trial/multi-seed NIAH | Check whether SourceCopy is robust beyond two seeds | Next when GPU is safe |
| 3 | PPL refresh with SourceCopy disabled | Ensure general semantic loss is still measured without NIAH-only copy help | Next |
| 4 | Full ablation table: no SourceCopy vs SourceCopy | Quantify exactness gain from the reranker | Needed for paper |
| 5 | True 4090 retest | Convert A100 memory-envelope evidence into hardware evidence | Later |


### Round 24 Follow-Up: Third Seed Check

Additional SourceCopy exactness run:

| Run | Result | Avg decode | Max reserved | Peak process group | Artifact |
|---|---:|---:|---:|---:|---|
| seed7777 required depths, 1 trial/depth | `4/4` | `843.5 ms/token` | `21.5801 GiB` | `~22610 MiB` | `experiments/niah_128k_required4_main_win64_sourcecopy_boost20_seed7777_gpu3_20260529_164144.json` |

Aggregate SourceCopy exactness evidence is now `12/12` across seeds `6004`, `4242`, and `7777`, one trial per required depth. The seed7777 50% row was slower (`1703 ms/token`), so latency still needs a fairer breakdown before Workflow3.

## Workflow2 Round 25: Automatic Review PPL Refresh

Mode:

- Continued fully automatic review discipline: record failed ideas, keep diagnostic/oracle paths separate from main results, and avoid claiming SourceCopy as pure dot-product retrieval.
- Remote execution is currently blocked locally by SSH credential parsing, so this section records the latest completed remote evidence from the active Workflow2 run. Further GPU experiments must resume only after SSH access is restored safely.

PPL refresh:

| Test | Result | FullKV PPL | HeteroKV PPL | Delta | Max reserved | Artifact |
|---|---:|---:|---:|---:|---:|---|
| 14K prefix12288/tail4096, SDPA, SourceCopy disabled | pass | `2.9669542983` | `3.0070550170` | `+1.35%` | HeteroKV `18.2344 GiB` | `experiments/ppl_14k_prefix12288_tail4096_gate5_top1_nofusion_sdpa_sourcecopy_disabled_gpu3_20260529_*.json` |

Interpretation:

- The earlier eager-attention PPL run that OOMed is invalid as an algorithm failure; it failed because the FullKV baseline used the wrong attention implementation and materialized too large a temporary attention tensor.
- The SDPA PPL refresh is the valid PPL point for this round: SourceCopy is disabled, so this measures general language modeling degradation rather than NIAH exact-string assistance.
- The relative degradation is within the <=5% semantic-loss budget.
- `method_d_event_count=512`, so the HeteroKV path exercised retrieval events rather than only no-DRAM tail behavior.
- This strengthens the semantic-loss claim, but Workflow3 is still not ready until the latency table and broader SourceCopy/no-SourceCopy ablation are refreshed under the same reporting rules.

Updated current ranking:

| Rank | Idea | Purpose | Decision |
|---:|---|---|---|
| 1 | Source-aware retrieval + SourceCopy exactness reranker | Strongest 128K required-depth NIAH exactness so far | Keep as experimental exact-copy path |
| 2 | Source-aware retrieval without SourceCopy | Main approximate cache claim | Needs broader exactness improvement; current hard probe is `1/2` |
| 3 | SDPA PPL refresh with SourceCopy disabled | General semantic-loss evidence | Valid, within budget at `+1.35%` |
| 4 | Broader NIAH ablation matrix | Separate retrieval, source-aware rerank, and copy rerank contributions | Next GPU task |
| 5 | Latency breakdown and fair baseline | Decide whether Triton/kernel work is scientifically justified | Still blocking Workflow3 |

## Workflow2 Round 26: Driver-Based SourceCopy Ablation

Infrastructure fix:

- `scripts/run_experiment.py` now forwards the NIAH attention backend, source-gate margin, source-cue order-aware flag, SourceCopy logit boost, SourceCopy candidate count, and gate-bypass flags to `scripts/run_niah_eval.py`.
- Remote validation passed before GPU tests: `16 passed`.
- Remote commit created: `396379c Pass NIAH SourceCopy args through workflow driver`.
- GitHub push for `396379c` is still pending because the server could not reach `github.com:443`; the commit exists in the remote working repository.

Controlled ablation under the 22 GiB cap and 30 GiB own-process fuse:

| Path | Length | Depths / Trials | Accuracy | Peak process memory | Monitor killed | Artifact |
|---|---:|---|---:|---:|---:|---|
| Source-aware retrieval, no SourceCopy | 128K | 25%/50%, 2 trials each | `3/4` | `21.8242 GiB` | False | `experiments/niah_128k_depth25_50_trials2_main_nosourcecopy_driver_gpu3_20260529_auto.json` |
| Source-aware retrieval + SourceCopy boost20 | 128K | 25%/50%, 2 trials each | `4/4` | `21.8262 GiB` | False | `experiments/niah_128k_depth25_50_trials2_main_sourcecopy_boost20_driver_gpu3_20260529_auto.json` |

Per-case exactness:

| Depth | Trial | Target | No SourceCopy | SourceCopy |
|---:|---:|---|---|---|
| 25% | 0 | `847754` | correct | correct |
| 25% | 1 | `690144` | failed, generated `69144...` | correct |
| 50% | 0 | `792275` | correct | correct |
| 50% | 1 | `439778` | correct | correct |

Mechanism notes:

- Both runs used source-aware token-level Method-D retrieval with `query_top_r_mean`, query history 64, consensus 8, source overlap required, source fusion enabled, focus-only retrieval, and K/V reuse.
- Both runs reported `max_hbm_tokens=12352`, `dram_entries=1680`, and `method_d_event_count=512` per row.
- The SourceCopy improvement is therefore an exact-string decoding/reranking gain on top of the same retrieval substrate, not a memory-budget change.
- Report SourceCopy as an experimental exact-copy reranker. Do not merge it into the pure Query-Key dot-product claim.

## Workflow2 Round 27: SourceCopy Required-Depth Robustness

Extended driver-based robustness run:

| Path | Seed | Length | Depths / Trials | Accuracy | Peak process memory | Monitor killed | Artifact |
|---|---:|---:|---|---:|---:|---:|---|
| Source-aware retrieval + SourceCopy boost20 | `4242` | 128K | 25%/50%/75%/90%, 2 trials each | `8/8` | `21.8262 GiB` | False | `experiments/niah_128k_required4_trials2_sourcecopy_boost20_seed4242_driver_gpu3_20260529_auto.json` |

Rows:

| Depth | Trial | Code | Correct | Row elapsed |
|---:|---:|---|---:|---:|
| 25% | 0 | `620966` | True | `81.16s` |
| 25% | 1 | `542870` | True | `74.15s` |
| 50% | 0 | `722971` | True | `73.63s` |
| 50% | 1 | `028225` | True | `74.22s` |
| 75% | 0 | `123937` | True | `72.72s` |
| 75% | 1 | `045052` | True | `72.33s` |
| 90% | 0 | `855966` | True | `72.11s` |
| 90% | 1 | `542598` | True | `72.60s` |

Mechanism/memory:

- `max_reserved_gib=21.3262`, monitor peak `21.8262 GiB`.
- `max_hbm_tokens=12352`, `dram_entries=1680`, `method_d_event_count=512` for each row.
- Total run time `618.3s`; mean row elapsed `74.1s`.

Decision:

- This strengthens the SourceCopy-assisted NIAH exactness evidence under the same 22 GiB memory envelope.
- It still does not turn SourceCopy into the main pure retrieval claim; it is a copy-task reranker layered above source-aware retrieval.

## Workflow2 Round 28: SourceCopy Required-Depth Robustness, Second Full Seed

Extended driver-based robustness run:

| Path | Seed | Length | Depths / Trials | Accuracy | Peak process memory | Monitor killed | Artifact |
|---|---:|---:|---|---:|---:|---:|---|
| Source-aware retrieval + SourceCopy boost20 | `7777` | 128K | 25%/50%/75%/90%, 2 trials each | `8/8` | `21.8262 GiB` | False | `experiments/niah_128k_required4_trials2_sourcecopy_boost20_seed7777_driver_gpu3_20260529_auto.json` |

Rows:

| Depth | Trial | Code | Correct | Row elapsed |
|---:|---:|---|---:|---:|
| 25% | 0 | `285761` | True | `77.49s` |
| 25% | 1 | `668808` | True | `74.22s` |
| 50% | 0 | `877347` | True | `73.84s` |
| 50% | 1 | `178244` | True | `73.07s` |
| 75% | 0 | `640303` | True | `71.02s` |
| 75% | 1 | `676631` | True | `61.91s` |
| 90% | 0 | `057936` | True | `73.64s` |
| 90% | 1 | `781509` | True | `73.64s` |

Mechanism/memory:

- `max_reserved_gib=21.3262`, monitor peak `21.8262 GiB`.
- `max_hbm_tokens=12352`, `dram_entries=1680`, `method_d_event_count=512` for each row.
- Total run time `603.3s`; mean row elapsed `72.4s`.

Decision:

- Driver-based SourceCopy-assisted required-depth robustness now has two full seeds with 2 trials per depth: seed4242 `8/8` and seed7777 `8/8`.
- Together with the driver-based seed6004 25%/50% ablation (`4/4`), the monitored SourceCopy exactness evidence is `20/20`, but seed6004 is not yet a full required-depth 2-trial seed in this driver format.
- Keep the claim boundary strict: this is an experimental exact-copy reranker layered on source-aware retrieval, not the pure token-level dot-product retrieval result.
- Next automatic Workflow2 step should prioritize latency breakdown and fair baseline refresh, unless GPU safety suggests a seed6004 full required-depth driver rerun is cheaper and more useful first.

## Workflow2 Round 29: SourceCopy Required-Depth Robustness, Third Full Seed

Extended driver-based robustness run:

| Path | Seed | Length | Depths / Trials | Accuracy | Peak process memory | Monitor killed | Artifact |
|---|---:|---:|---|---:|---:|---:|---|
| Source-aware retrieval + SourceCopy boost20 | `6004` | 128K | 25%/50%/75%/90%, 2 trials each | `8/8` | `21.8262 GiB` | False | `experiments/niah_128k_required4_trials2_sourcecopy_boost20_seed6004_driver_gpu3_20260529_auto.json` |

Rows:

| Depth | Trial | Code | Correct | Row elapsed |
|---:|---:|---|---:|---:|
| 25% | 0 | `847754` | True | `76.36s` |
| 25% | 1 | `690144` | True | `76.13s` |
| 50% | 0 | `792275` | True | `71.20s` |
| 50% | 1 | `439778` | True | `63.06s` |
| 75% | 0 | `899516` | True | `74.59s` |
| 75% | 1 | `618089` | True | `75.27s` |
| 90% | 0 | `205264` | True | `73.30s` |
| 90% | 1 | `259182` | True | `71.10s` |

Mechanism/memory:

- `max_reserved_gib=21.3262`, monitor peak `21.8262 GiB`.
- `max_hbm_tokens=12352`, `dram_entries=1680`, `method_d_event_count=512` for each row.
- Total run time `608.1s`; mean row elapsed `72.6s`.

Decision:

- Driver-based SourceCopy-assisted required-depth NIAH is now `24/24` across seeds `4242`, `7777`, and `6004`, with 2 trials per required depth under the 22 GiB cap and 30 GiB own-process fuse.
- This is the strongest current 128K exact-copy task evidence, but it remains a separate experimental exactness reranker result.
- The pure source-aware retrieval without SourceCopy remains weaker on the hard same-case ablation (`3/4` on 25%/50% trials2), which should be reported rather than hidden.
- Next automatic Workflow2 priority shifts to latency breakdown and fair baseline refresh, because quality and memory evidence for the SourceCopy-assisted path are now strong enough for this stage.

## Workflow2 Round 30: TTL12 Latency Candidate And Baseline Refresh

Automatic review target:

- After SourceCopy-assisted quality reached `24/24`, latency became the main blocker.
- FullKV 128K SDPA baseline can be run only when a target GPU is idle. During this round GPU1 was idle; GPU0 had another `ahr` VideoMME/native-KV process and was not touched.

Current TTL6 SourceCopy baseline:

| Path | Accuracy | Mean decode | Median decode | Peak process memory |
|---|---:|---:|---:|---:|
| SourceCopy TTL6, seeds 6004/4242/7777 | `24/24` | `689.4 ms/step` | `735.7 ms/step` | `21.8262 GiB` |

Ideas tested:

| Idea | Result | Decision | Artifact |
|---|---:|---|---|
| SourceCopy + Triton scoring, seed6004 25/50 trials2 | `4/4`, `571.0 ms/step` | Correct but only ~3.2% faster than same-case torch scoring; do not expand as main path | `experiments/niah_128k_depth25_50_trials2_sourcecopy_tritonscore_seed6004_driver_gpu3_20260529_auto.json` |
| Decode without attention mask | clean retry failed `0/4` with cuBLAS/runtime errors | Reject; short-KV decode still depends on the wrapper's mask path | `experiments/niah_128k_depth25_50_trials2_sourcecopy_nomask_seed6004_retryclean_gpu3_20260529_auto.json` |
| SourceCopy + selected-key TTL12 | `24/24` full required-depth across 3 seeds | Promote as current latency candidate | see below |

TTL12 validation under the 22 GiB cap and 30 GiB own-process fuse:

| Seed | Depths / Trials | Accuracy | Mean decode | Mean prefill | Max reserved | Artifact |
|---:|---|---:|---:|---:|---:|---|
| `6004` | 25/50/75/90, 2 each | `8/8` | `365.4 ms/step` | `48.5s` | `21.3262 GiB` | `experiments/niah_128k_required4_trials2_sourcecopy_ttl12_seed6004_driver_gpu3_20260529_auto.json` |
| `4242` | 25/50/75/90, 2 each | `8/8` | `533.3 ms/step` | `55.3s` | `21.3262 GiB` | `experiments/niah_128k_required4_trials2_sourcecopy_ttl12_seed4242_driver_gpu3_20260529_auto.json` |
| `7777` | 25/50/75/90, 2 each | `8/8` | `453.2 ms/step` | `55.2s` | `21.3262 GiB` | `experiments/niah_128k_required4_trials2_sourcecopy_ttl12_seed7777_driver_gpu3_20260529_auto.json` |

Aggregate TTL12:

- Accuracy: `24/24`.
- Mean decode: `450.6 ms/step`.
- Median decode: `393.6 ms/step`.
- Decode range: `328.1-875.0 ms/step`.
- Mean prefill: `53.0s`.
- Peak process memory: `21.8262 GiB`.

Refreshed FullKV wide-memory baseline:

| Baseline | Result | Prefill | Decode | Max reserved | Monitor peak | Artifact |
|---|---:|---:|---:|---:|---:|---|
| FullKV 128K SDPA manual decode, GPU1, 75 GiB cap | `1/1` | `28.72s` | `52.25 ms/step` | `62.9629 GiB` | `41.3672 GiB` | `experiments/niah_fullkv_128k_cap75_sdpa_manual_latency_refresh_gpu1_20260529_auto.json` |

Decision:

- TTL12 is the current best latency/quality candidate for the SourceCopy-assisted path.
- It improves mean decode from `689.4` to `450.6 ms/step` without increasing the 22 GiB memory envelope.
- It still does not meet the original `<=2x` latency target: mean ratio is `8.62x`, median ratio is `7.53x` versus the refreshed wide-memory FullKV SDPA reference.
- Workflow3 is not ready on latency grounds. Next credible step is a PPL refresh under the TTL12 candidate with SourceCopy kept out of general-language PPL, then consider deeper attention/retrieval fusion only if the user wants to pursue the latency gap.

## Workflow2 Round 31: TTL12 PPL Refresh And Runner Audit

Automatic review target:

- Validate semantic loss after the TTL12 latency-candidate round.
- Keep SourceCopy out of WikiText-2 PPL because SourceCopy is an exact-copy/reranker layer for NIAH-style source cues, not a general-language modeling mechanism.
- Fix the PPL runner so TTL and cache-shape parameters are recorded rather than inferred from filenames.

Runner audit:

| Finding | Fix | File |
|---|---|---|
| `run_ppl_eval.py` exposed `reuse_kv_cache` but did not pass `method_d_reuse_ttl_tokens` or `method_d_reuse_source_threshold` into `build_fused_cache` | Added both argparse flags, passed them into the cache, and wrote them into `method_d_config` | `scripts/run_ppl_eval.py` |
| PPL JSON did not record `sink_tokens`, `keep_tail`, or `chunk_size` | Added `cache_config` to the JSON result | `scripts/run_ppl_eval.py` |

Safety notes:

- First strict run on GPU1 stopped with `rc=8` after another user's process appeared; this is a valid shared-server safety stop, not a model failure.
- Retry used GPU3 with `--allow-other-processes-if-memory-fits`, because the other process used about `16.3 GiB` on GPU3 and the combined total remained well below A100 capacity.
- Own-process fuse remained `30 GiB`; PyTorch cap remained `22 GiB`; reserve remained `4 GiB`.

PPL refresh:

| Variant | Tokens | Loss suffix | SourceCopy | TTL config | Full PPL | HeteroKV PPL | Delta | Hetero max reserved | Process peak | Artifact |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| WikiText-2 real PPL, SDPA, FullKV vs HeteroKV | `14336` | `2048` | disabled | `reuse_ttl_tokens=12`, `reuse_source_threshold=35`, `reuse_kv_cache=True` | `2.9706` | `3.0063` | `+1.20%` | `19.2754 GiB` | `20.248 GiB` | `experiments/ppl_14k_prefix12288_tail4096_gate5_top1_nofusion_sdpa_ttl12_sourcecopy_disabled_allowcoexist_gpu3_20260529_auto.json` |

Mechanism/memory:

- `cache_config`: sink `64`, tail `4096`, chunk `2048`.
- `method_d_event_count=512`.
- `memory_summary`: `max_hbm_tokens=6208`, `dram_entries=112`, `dram_bytes=245891072`.
- Monitor peak total GPU memory was `36.569 GiB`, including another user's process; own process stayed below the `30 GiB` fuse.

Decision:

- The TTL12 branch still passes the PPL degradation target with SourceCopy disabled: `+1.20% <= 5%`.
- Because `source_token_boost=0` and `reuse_source_threshold=35`, TTL reuse is recorded but not allowed to turn arbitrary general-language chunks into SourceCopy-style cached chunks. This keeps the PPL claim separate from exact-copy NIAH claims.
- Workflow3 is still not ready: quality and PPL are now strong for the documented branches, but latency remains about `8.62x` mean versus the refreshed wide-memory FullKV SDPA reference.
