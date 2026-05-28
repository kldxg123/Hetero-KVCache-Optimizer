# Workflow 2.0 Review Log

Date: 2026-05-27

Primary tracker: `experiments/experiment_tracker_workflow2_round1.json`

## Round 1 Review

Reviewer score: 4/10

Verdict: not ready

Core finding:

The project has reached a runtime integration milestone, not a research-claim milestone.

Supported claim:

- Qwen2.5-7B-Instruct real `generate()` path can run short-context smoke tests under a 22 GiB PyTorch cap and 30 GiB debug fuse.

Unsupported claims:

- 128K 4090-24G survival.
- NIAH acceptance accuracy.
- WikiText-2 PPL degradation.
- Latency target.

Ranked weaknesses:

1. No 128K survival evidence.
2. Stage2 quality regression at 4K/8K.
3. NIAH is not acceptance-grade.
4. No PPL / latency evidence.
5. Raw artifacts must remain primary evidence.
6. 30 GiB monitor is a debug fuse, not an acceptance memory proof.
7. Ablation matrix was absent.

## Actions Taken After Round 1

### A/B Ablation Runner

Implemented: `scripts/run_stage2_ablation.py`

It compares:

- `full_kv_baseline`
- `heterokv_no_retrieval`
- `heterokv_dotproduct`

under the same prompt, seed, 22 GiB cap, and 30 GiB debug fuse.

### NIAH Smoke Runner

Implemented: `scripts/run_niah_eval.py`

Default smoke:

- lengths: 4096, 8192
- depths: 25%, 50%, 75%, 90%
- trials: 1

This is not the final NIAH acceptance test. It is a Workflow 2.0 diagnostic test.

### Method-D Gate

Problem found:

`heterokv_dotproduct` failed at 4K while both `full_kv_baseline` and `heterokv_no_retrieval` passed.

Diagnosis:

Method-D was retrieving DRAM chunks too eagerly during decode, injecting unrelated historical tokens into tail-instruction prompts.

Fix:

Added an HBM-vs-DRAM QK gate:

- DRAM retrieval still uses token-level Query x 4-bit Key dot-product.
- A retrieved DRAM chunk is only spliced into active KV if its best score beats the active HBM best score by `method_d_gate_margin`.
- Default margin: `1.10`.

## Round 2.0 Late-Stage Review

Reviewer score: 6/10

Verdict: memory-survival claim is supported; 128K semantic acceptance is not yet supported.

What is real and supported:

- 128K HeteroKV under 22 GiB cap does not OOM with `keep_tail=8192`.
- `torch.cuda.max_reserved` stayed around 20.65 GiB in the successful 128K survival runs.
- `nvidia-smi` process memory stayed below the 30 GiB debug fuse.
- DRAM compressed KV grows with context while active HBM KV remains bounded.
- FP32 QK scoring in short-KV manual attention is a real correctness fix.
- 16K, 32K, and 64K NIAH smoke tests have passing configurations.

What remains unsupported:

- Real 128K dot-product retrieval accuracy >=95%.
- WikiText-2 real PPL degradation <=5%.
- Native RTX 4090 latency.

Critical 128K finding:

Oracle retrieval plus a small attention bias reaches 4/4, while real
dot-product retrieval remains at 0/4 to 1/4 depending on top-k/bias.  Therefore
the current blocker is not physical cache survival; it is retrieval ranking and
false-positive chunk suppression at 128K.

Updated after rerank experiments:

- Real dot-product retrieval with range consensus and moderate retrieval bias
  reached 3/4 on the 128K four-depth smoke matrix.
- The remaining failed case is 25% depth, which repeatedly generates `000000`.
- Single-depth 25% diagnostics show that increasing top-k, lowering consensus,
  and adding a tail guard do not fix the failure.
- This is a meaningful improvement over the earlier 0/4 to 1/4 real dot-product
  runs, but it is still not a 95% acceptance result.

Do not overclaim:

- The oracle path is diagnostic only.
- Retrieval bias is diagnostic until validated with real dot-product retrieval.
- A100/22 GiB cap is a memory-envelope result, not a real 4090 latency result.

## Post-Fix Results

### Stage2 Ablation

Command:

```bash
CUDA_VISIBLE_DEVICES=1 ./run-experiment --stage gpu --suite ablation --gpu-index 1 --cap-gib 22 --max-vram-gib 30 --monitor-interval-sec 5 --timeout 1800 --tracker experiments/experiment_tracker_workflow2_round1.json
```

Result:

| Target | full KV | HeteroKV no retrieval | HeteroKV dotproduct |
| --- | --- | --- | --- |
| 4096 | pass | pass | pass |
| 8192 | OOM under 22 GiB cap | near-pass but exact mismatch | near-pass but exact mismatch |

Monitor:

- max process memory: 19.14 GiB.
- 30 GiB debug fuse not triggered.

Interpretation:

- The 4K dotproduct-specific regression is fixed.
- Full KV OOM under the 22 GiB cap at 8192 target smoke is useful survival contrast evidence.
- 8192 exact mismatch remains, but no-retrieval and dotproduct now behave similarly, so the remaining issue is not specifically Method-D retrieval.

### NIAH Smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=1 ./run-experiment --stage gpu --suite niah --gpu-index 1 --cap-gib 22 --max-vram-gib 30 --monitor-interval-sec 5 --timeout 1800 --tracker experiments/experiment_tracker_workflow2_round1.json
```

Result:

- accuracy: 2/8 = 25%.
- previous smoke accuracy before Method-D gate: 1/8 = 12.5%.
- max process memory: 15.21 GiB.
- 30 GiB debug fuse not triggered.

Interpretation:

- The gate improved small NIAH smoke but is far below acceptance.
- The project must not claim NIAH success yet.
- Next work should focus on real retrieval effectiveness, not memory survival alone.

## Current Verdict After Fixes

Status: still not ready for final research claim.

Improved:

- Runtime path is stable.
- 4K Method-D quality regression was fixed.
- Ablation runner now exists.
- NIAH diagnostic runner now exists.

Still blocked:

- NIAH accuracy is far below 95%.
- No 64K/128K survival runner yet.
- No WikiText-2 PPL runner yet.
- No latency breakdown runner yet.

## Next Minimum Fix

Before 128K:

1. Improve NIAH retrieval quality at 4K/8K.
2. Add retrieval evidence per row: selected chunks, ranges, scores, gate decision.
3. Add baseline rows for no-retrieval NIAH and dotproduct NIAH.
4. Only then scale to 16K/32K and finally 64K/128K survival.

## Round 2 Review And Actions

Reviewer score: 5/10

Verdict: still not ready

Core findings:

1. The NIAH orchestrator must not mark low accuracy as success.
2. NIAH rows need attribution evidence, not just aggregate accuracy.
3. Method-D gate and retrieval effects must be separable from baseline/no-retrieval behavior.
4. Do not spend 128K GPU time until a calibrated 4K attribution test is meaningful.

Implemented after review:

- `scripts/run_niah_eval.py` now supports multi-mode rows.
- `scripts/run_experiment.py --suite niah` passes mode and primary-mode settings.
- `quality_failed` now returns a non-zero child code and makes the tracker fail instead of silently passing.
- Method-D events include selected chunks, chunk ranges, scores, gate decisions, and optional retrieved token windows.
- NIAH prompt was recalibrated with Qwen chat template and 6-digit numeric codes.

## Round 2 Experiments

### Calibrated 4K Attribution

Command shape:

```bash
CUDA_VISIBLE_DEVICES=1 ./run-experiment --stage gpu --suite niah \
  --gpu-index 1 --cap-gib 22 --max-vram-gib 30 \
  --niah-lengths 4096 --niah-depths 0.25 0.5 0.75 0.9 \
  --niah-trials 1 \
  --niah-modes full_kv_baseline heterokv_no_retrieval heterokv_dotproduct \
  --niah-primary-mode heterokv_dotproduct
```

Results:

| Mode | Accuracy |
| --- | ---: |
| full KV baseline | 4/4 |
| HeteroKV no retrieval, keep_tail=2048 | 2/4 |
| HeteroKV dotproduct, keep_tail=2048 | 2/4 |

Monitor:

- max process memory: 20.18 GiB.
- 30 GiB debug fuse not triggered.

Interpretation:

- The calibrated prompt is valid because full KV reaches 100% on the same rows.
- HeteroKV passes near-tail cases and fails early/mid-depth cases.
- Dot-product retrieval currently does not add measurable NIAH benefit over no-retrieval at 4K.

### Failed Idea: Force Retrieval

Change:

- `--niah-method-d-gate-margin 0`

Result:

| Mode | Accuracy |
| --- | ---: |
| full KV baseline | 4/4 |
| HeteroKV no retrieval | 2/4 |
| HeteroKV dotproduct, gate=0 | 0/4 |

Interpretation:

- Over-injecting DRAM chunks is harmful.
- The answer is not simply "retrieve more".

### Failed Idea: Token-Window Retrieval

Change:

- `--niah-method-d-token-window 256`

Result:

| Mode | Accuracy |
| --- | ---: |
| full KV baseline | 4/4 |
| HeteroKV no retrieval | 2/4 |
| HeteroKV dotproduct, 256-token window | 2/4 |

Interpretation:

- Narrowing the retrieved span does not yet recover early/mid-depth needles.
- For depth 25%, most selected token windows drift toward the end of the evicted chunk and miss the needle.
- For depth 50%, many windows overlap the needle but generation still fails, so injection/attention influence remains suspect.

### Control: No Early Eviction

Change:

- `--niah-keep-tail 4096`
- modes: `full_kv_baseline`, `heterokv_no_retrieval`
- primary: `heterokv_no_retrieval`

Result:

| Mode | Accuracy |
| --- | ---: |
| full KV baseline | 4/4 |
| HeteroKV no retrieval, keep_tail=4096 | 4/4 |

Monitor:

- max process memory: 21.93 GiB.
- 30 GiB debug fuse not triggered.

Interpretation:

- The patched attention wrapper and absolute-position mask are not the primary failure in non-eviction 4K.
- The current blocker is DRAM retrieval quality after eviction.

## Current Round 2 Verdict

Do not proceed to 16K/32K acceptance yet.

The next minimum fix should stay at calibrated 4K and answer this specific question:

> When the selected DRAM chunk contains the needle, why does the retrieved KV fail to steer the first generated answer token?

Candidate next tests:

1. Record attention mass assigned to retrieved DRAM tokens vs active HBM tokens on the first generated token.
2. Compare BF16 DRAM retrieval against quantized 4-bit retrieval to isolate quantization error.
3. Add an oracle retrieval mode that retrieves the chunk containing the known needle range, without using it for final claims, to separate retrieval ranking from KV injection quality.

## Round 3 Diagnostics

Cross-model review recommendation was followed:

- oracle retrieval mode was added.
- first-token attention-mass probes were added.
- BF16 DRAM diagnostic retrieval was added.
- HeteroKV NIAH was corrected to use chunked prefill plus decode-first-token instead of relying on HF `generate()` prefill logits.

### Why The Test Path Changed

HF `generate()` produces the first new token from the prefill forward pass. Method-D retrieval is a decode-time mechanism, so it cannot rescue the first answer token if that token is selected from prefill logits. The diagnostic NIAH path now prefills through all but the last prompt token, then decodes the final prompt token with the HeteroKV cache so retrieval is active for the first answer token.

### Results

| Experiment | Result | Peak process memory | Verdict |
| --- | ---: | ---: | --- |
| chunked keep_tail=4096 control | HeteroKV no retrieval 4/4 | 18.89 GiB | Test path valid |
| chunked dotproduct keep_tail=2048 | dotproduct 2/4, no-retrieval 2/4 | 15.02 GiB | No retrieval gain |
| oracle full chunk | oracle 2/4 | 21.93 GiB | Retrieved chunk is not enough |
| oracle 64-token window | oracle 2/4 | 20.18 GiB | More precise range still not enough |
| BF16 oracle 64-token window | oracle 2/4 | 20.18 GiB | 4-bit quantization not primary blocker |

Attention-mass finding:

- Full oracle chunk retrieval can give retrieved tokens almost all attention mass, but the actual needle tokens receive very little mass.
- Oracle 64-token retrieval increases needle mass, but the answer still fails in early/mid-depth cases.
- Near-tail cases pass because the needle remains in active HBM, with much higher needle mass.

Round 3 conclusion:

- The failure is not simply ranking, not simply gate threshold, and not primarily 4-bit quantization.
- The current HeteroKV approximation loses semantic recoverability for early/mid-depth NIAH once the needle has been evicted.
- Continue at calibrated 4K. Do not claim NIAH success or scale semantic acceptance to 16K/128K yet.

Next ranked technical directions:

1. Preserve richer semantic neighborhoods around candidate needles, not only raw token windows selected by max QK.
2. Investigate prefill representation damage caused by short-KV attention during prompt encoding.
3. Add a controlled mode that keeps prefill attention full for small 4K only, then shortens cache before decode, to isolate prefill damage from decode retrieval.
4. Explore retrieval-aware first-token generation as a formal API path rather than relying on standard HF `generate()` semantics.

## Round 4 Structural Reranker And Source Fusion

Outcome:

- 128K required-depth NIAH passed on the real `heterokv_dotproduct` path.
- Oracle/diagnostic experiments were kept separate and used only to localize failure modes.

What changed:

- Query-history retrieval was fixed so `method_d_query_history_tokens` no longer gets truncated to one token.
- A query-history reducer, `query_top_r_mean`, was added to reduce single-token false positives.
- Source-aware attention fusion was added as an optional retrieved-only attention branch controlled by `method_d_source_fusion_alpha`.
- A source-token lexical reranker was added, controlled by `method_d_source_token_boost`; it uses source/query token overlap only and does not use the needle range.

Key diagnostic sequence:

| Step | Result | Interpretation |
| --- | --- | --- |
| focus-only top8 | 32K 4/4, 128K 25% failed | Retrieval hit the right chunk often, but wrong sources still dominated. |
| stronger focus bias | 128K 25% failed | Bias alone amplifies false positives. |
| top4/top1 source fusion | 32K 4/4, 128K 25% failed | Fewer chunks still did not identify the right source reliably. |
| oracle source fusion | 128K 25% passed | Fusion can recover when the source is correct; ranking was the blocker. |
| source-token reranker + source fusion | 128K 25/50/75/90 passed | False-positive reranking was sufficiently improved. |

Main result configuration:

`heterokv_dotproduct`, 128K, depths `25/50/75/90`, `keep_tail=8192`, `top_k=4`, `query_top_r_mean`, `query_history=64`, `source_token_boost=2.0`, `source_fusion_alpha=0.35`, 22 GiB cap, 30 GiB fuse.

Main result:

| Depth | Correct |
| ---: | --- |
| 25% | True |
| 50% | True |
| 75% | True |
| 90% | True |

Accuracy: `4/4 = 100%`.

Memory:

- max allocated: about `19.6688 GiB`.
- max reserved: about `20.6465 GiB`.
- process peak: about `21.5 GiB`.

Remaining work before paper-writing:

1. Run optional 0% and 99% NIAH depths.
2. Run multi-trial NIAH, not only one seed/code per depth.
3. Run real WikiText-2 PPL against a feasible full-KV baseline length.
4. Run latency breakdown and compare against an uncapped A100 baseline only when safe.
5. Clearly describe source-token reranking as an added metadata mechanism.

Boundary-depth follow-up:

- Optional 99% depth passed on the same source-aware main path.
- Optional 0% depth failed with both `method_d_min_position=4096` and `method_d_min_position=0`.
- The current claim is therefore limited to the required 25/50/75/90 depths.  Do not claim full boundary-depth robustness yet.

## Round 5 PPL And Reranker Review

What changed:

- Added a real WikiText-2 PPL harness using CE loss.
- Added `decode_suffix` PPL to exercise compressed-prefix decode and Method-D retrieval.
- Added dynamic source-token metadata for PPL so the reranker only sees tokens already observed by the model.
- Added GPU1 safety checks for 30 GiB fuse and other-process detection.

Findings:

| Test | Result | Review |
| --- | --- | --- |
| Aggressive source fusion on 512-token PPL | Full 6.7392 vs HeteroKV 33.2591 | Failed; source fusion causes unacceptable false positives on generic text. |
| Strict gate/no fusion on 512-token PPL | Full 6.7392 vs HeteroKV 7.5251 | Better but still +11.7% under a tiny active window. |
| Strict gate/no fusion on 4K suffix PPL | Full 6.2006 vs HeteroKV 5.1463 | Passed this sample; no observed degradation, with 28 DRAM entries. |

Scientific boundary:

- The PPL result is real loss, but only a 4K feasible full-baseline suffix under the current safety envelope.
- The negative PPL delta is not evidence of superior modeling; it should be reported as "no degradation observed on this sample".
- The PPL-safe strict gate configuration is separate from the NIAH-optimized source-fusion configuration.
- The failed aggressive PPL result must stay in the report to prevent overclaiming.

Next workflow step:

1. Run multi-trial 128K NIAH to test robustness of the main result.
2. Run latency breakdown for the accepted 128K NIAH configuration.
3. Keep full 128K full-KV baseline or uncapped latency baseline for an idle-server window only.

## Round 6 Latency Review

128K single-depth latency was captured after adding per-case timing fields.

Result:

- Correct at 50% depth.
- Total elapsed: `92.86 s`.
- Chunked prefill: `62.70 s`.
- Decode: `30.16 s` for 25 steps.
- Decode: `1206.44 ms/step`.
- Monitor peak process memory: `20.44 GiB`.
- Torch max reserved: `19.94 GiB`.

Review:

- The latency evidence is now real and separated from correctness.
- The number is only "A100 under 22 GiB memory envelope"; it is not a 4090 latency proof.
- A full-KV or no-retrieval latency baseline is still needed before claiming `<=2x`.
- The 30 GiB safety fuse was not triggered.

## Round 7 Robustness Review

Multi-trial NIAH exposed a real weakness:

- Previous source-aware main path: `6/8` on 128K required-depth, 2 trials each.
- Failures were at 25% and 50%.
- The failure was not OOM and not missing DRAM storage; it was false-positive/retrieved-source influence.

Structural changes reviewed:

- Added source-overlap hard filtering.
- Added dynamic source-fusion alpha:
  - strong alpha for high source-score chunks;
  - lower alpha for weaker near-tail source evidence.
- Cleared stale retrieval state per decode step.

Outcome:

- Stage1: `10 passed`.
- Sensitive retry 25/50/90 with dynamic alpha: `6/6`.
- Full required-depth multi-trial 128K: `8/8`.
- Dynamic-alpha log-check case after the logging-order fix: passed, with event tail alpha values including `0.5`.
- Monitor peak process memory: `20.4375 GiB`.

Review verdict:

- The current required-depth 128K NIAH evidence is much stronger than the earlier single-trial result.
- Optional 0% boundary remains unresolved and should stay out of the main claim.
- Latency baseline and broader PPL sampling remain before paper-ready status.

## Round 8 Baseline And Latency Boundary Review

Idle-server work completed:

- FullKV 128K was run under the same 22 GiB cap used for the HeteroKV 4090-envelope claim.
- FullKV 128K was also attempted with a wide 75 GiB A100 cap.
- Short-context 8K references were run for FullKV and HeteroKV.
- A 128K HeteroKV no-retrieval latency ablation was run.

Findings:

| Test | Outcome | Review |
| --- | --- | --- |
| FullKV 128K, 22 GiB cap | OOM | Valid control showing full KV does not survive the 4090-like memory envelope. |
| FullKV 128K, 75 GiB cap | OOM | Eager full attention attempted an `895.92 GiB` allocation, so it cannot be used as a latency baseline. |
| FullKV 8K, 75 GiB cap | Correct, `115.74 ms/step`, `32.61 GiB` reserved | Useful short-context reference, but it already exceeds the 24G target. |
| HeteroKV 8K, 22 GiB cap | Correct, `63.27 ms/step`, `19.86 GiB` reserved | Sanity check that capped HeteroKV path works at short length. |
| HeteroKV 128K no retrieval | `82.33 ms/step`, quality failed | Shows retrieval/fusion is the main decode cost and is required for answer recovery. |

Scientific conclusions:

- The memory-survival story is now stronger: accepted HeteroKV 128K runs stay near `20.44 GiB` process memory, while FullKV 128K OOMs under the same 22 GiB cap.
- The 128K latency ratio remains unresolved because the complete full-KV eager baseline cannot finish.
- The no-retrieval ablation is not an accepted system; it is diagnostic evidence that quality depends on retrieval.
- The next latency direction should be an optimized baseline and/or retrieval-overhead reduction, not a misleading ratio against a failed run.

Updated paper-readiness:

- Ready for a truthful 128K survival + NIAH required-depth mechanism section.
- Ready for an honest PPL subsection with the current 4K suffix result and the failed aggressive-PPL result.
- Not ready for a strong `<=2x 128K full-KV latency` claim.
- Remaining optional improvements: broader PPL sampling, optimized SDPA/FlashAttention baseline, optional 0% boundary repair, and retrieval scoring optimization.

## Round 9 ARIS-Style Review State

ARIS-style loop rules were adopted locally in:

- `review-stage/ARIS_WORKFLOW2_ADAPTATION.md`
- `review-stage/REVIEW_STATE.json`

New baseline evidence:

| Test | Outcome | Review |
| --- | --- | --- |
| FullKV 8K, SDPA, 75 GiB cap | Correct, `16.06 GiB` reserved | SDPA is a fairer baseline backend than eager. |
| FullKV 128K, SDPA, 22 GiB cap | OOM near cap | Stronger evidence that full KV does not survive the 24G envelope. |
| FullKV 128K, SDPA, 75 GiB cap | Correct, `40.88 GiB` reserved via generate | FullKV can run on A100 when memory is relaxed. |
| FullKV 128K, SDPA, manual timing | Correct, prefill `28.82s`, decode `39.42 ms/step`, `62.96 GiB` reserved | Gives a real A100 full-KV speed reference, but not a 24G survival baseline. |

New negative HeteroKV evidence:

- Artifact: `experiments/niah_heterokv_128k_seed6004_max24_retry_20260528_073824.json`.
- Target: `847754`.
- Generated: `000008...`.
- Result: quality failed.
- HeteroKV latency on this failed case: prefill `62.52s`, decode `992.05 ms/step`, max reserved `20.65 GiB`.

Review verdict:

- The memory-survival claim remains strong.
- The previous 8/8 required-depth result is not yet enough for a top-conference robustness claim because seed 6004 at 50% depth failed.
- The failure appears to be after retrieval, not simply missing the source chunk: the system retrieved source-adjacent chunks but generation still drifted.
- The next workflow2 experiment should probe stronger or more selective source-aware fusion/focus on the failed seed, then regression-test against the previously passing required-depth suite.
- Workflow3 is not ready yet.

## Round 10 Cue-Focus Review

Problem found:

- The failed seed6004 case was not a missing-retrieval failure.
- Method-D selected the correct source chunk, but dot-product best-token offsets often landed on `[NEEDLE]` tags or cue text rather than the answer digits.
- Strong focus-only source fusion over the retrieved chunk fixed some cases but over-focused markup/filler and failed required-depth regression.

Structural change:

- Added optional source cue focus:
  - no oracle answer span is passed;
  - NIAH runner registers cue token sequences such as `The target code is ` and `target_code=`;
  - the manager focuses the tokens immediately after the cue inside the retrieved DRAM chunk;
  - oracle/diagnostic paths remain separate.

Negative results:

| Config | Result | Lesson |
| --- | ---: | --- |
| focus-only source fusion, alpha `0.75` | seed6004 still failed, generated `047754...` | Better but first digit still unstable. |
| focus-only source fusion, alpha `1.0`, token window `128` | seed6004 failed toward `[NEEDLE]` markup | Strong fusion can overfit cue/markup tokens. |
| cue-focus alpha `0.85` | seed6004 fixed, but required-depth regression only `6/8` | Too strong; repeats retrieved span and can corrupt digits. |

Current best NIAH configuration:

- cue-focus enabled;
- `source_fusion_alpha=0.65`;
- `source_fusion_low_alpha=0.35`;
- `token_window=128`;
- `focus_bias=4.0`;
- `nonfocus_penalty=1.0`;
- `source_token_boost=2.5`;
- `query_history=64`;
- `keep_tail=8192`;
- 22 GiB cap.

Positive results:

| Test | Result | Artifact |
| --- | ---: | --- |
| seed6004 known failure, 50% depth | passed | `experiments/niah_cuefocus_alpha065_seed6004_20260528_133540.json` |
| seed4242 required depths 25/50/75/90, 2 trials each | `8/8` | `experiments/niah_128k_required_depths_cuefocus_alpha065_regression_20260528_133741.json` |
| seed6004 required depths 25/50/75/90, 2 trials each | `8/8` | `experiments/niah_128k_required_depths_cuefocus_alpha065_seed6004_regression_20260528_135221.json` |

Boundary result:

- Optional 99% depth passed `2/2`.
- Optional 0% depth still failed `0/2`.
- Artifact: `experiments/niah_128k_optional_depths_cuefocus_alpha065_20260528_140819.json`.

Review verdict:

- Required-depth NIAH evidence is now much stronger: two seeds, 16/16 cases at 128K under the 22 GiB envelope.
- Optional 0% remains a documented boundary weakness.
- The stronger NIAH method is task/cue-aware and must be presented as source-aware extraction support, not generic PPL-safe retrieval.
- Workflow3 is still not ready because broader PPL and latency-overhead evidence remain incomplete.

Server safety note:

- An attempted 8K WikiText-2 PPL run was skipped because another user's process `979528` occupied GPU1 and other GPUs.
- No process was killed or modified.

## Round 11-14 Review: PPL, TTL Reuse, And Boundary Failure

What is now stronger:

- Required-depth 128K NIAH is no longer a single-seed result. The current main path passes `16/16` across seeds `4242` and `6004`.
- Real 8K WikiText-2 PPL is within target: FullKV `7.2859`, HeteroKV `7.5443`, `+3.55%`.
- A PyTorch-only latency optimization was added without touching Triton/CUDA: source-aware Method-D reuse TTL.

Adopted mechanism:

- `method_d_reuse_ttl_tokens=6`
- `method_d_reuse_source_threshold=35`
- `method_d_token_window=64`

The reuse path is explicitly labeled in logs as reuse, not fresh dot-product retrieval. This keeps academic separation between first-hit Query x Key evidence and subsequent source-aware cached reuse.

Main current evidence:

| Test | Result | Decode | Artifact |
| --- | ---: | ---: | --- |
| seed4242 required depths, 2 trials/depth | `8/8` | `544.6 ms/token` avg | `experiments/niah_128k_required_depths_cuefocus_alpha065_reuse_ttl6_win64_thr35_seed4242_20260528_160930.json` |
| seed6004 required depths, 2 trials/depth | `8/8` | `562.8 ms/token` avg | `experiments/niah_128k_required_depths_cuefocus_alpha065_reuse_ttl6_win64_thr35_seed6004_20260528_162233.json` |
| seed7777 required depths, 1 trial/depth | `4/4` | `530.2 ms/token` avg | `experiments/niah_128k_required_depths_cuefocus_alpha065_reuse_ttl6_win64_thr35_seed7777_20260528_171805.json` |
| 8K WikiText-2 PPL | `+3.55%` vs FullKV | n/a | `experiments/ppl_8k_prefix6144_gate35_nofusion_sdpa_autogpu_20260528_142558.json` |
| 10K WikiText-2 PPL | `+0.07%` vs FullKV | n/a | `experiments/ppl_10k_prefix8192_gate35_nofusion_sdpa_autogpu_20260528_170936.json` |

Failed optimization ideas:

| Idea | Result | Lesson |
| --- | ---: | --- |
| Method-D layer subset `20-27` | failed sensitive seed6004 | Too few layers preserve the retrieval signal. |
| Method-D layer subset `8-27` | seed4242 required-depth `7/8` | Latency reduction is not worth quality loss. |
| Method-D layer subset `4-27` | seed4242 required-depth `7/8` | Even mild layer skipping is unstable. |
| TTL threshold `45` | no reuse triggered | Source score in the passing case was about `41.47`. |
| `query_history=16` | correct but not faster | Query history is not the dominant bottleneck. |
| TTL12 + window64 | correct but only marginally faster in single probe | TTL6 is the safer main setting. |

Boundary review:

- Optional 99% depth passes.
- Optional 0% depth still fails.
- `allow_source_before_min_position` and source-cue-score diagnostics did not fix 0%; selected chunks still did not carry useful source scores for the prefix case and the branch slowed down.

Current verdict:

- Workflow3 is not ready.
- Memory survival, required-depth semantic recovery (`20/20`), and 8K/10K PPL are now credible under the A100 22 GiB envelope.
- Latency remains the main blocker. The PyTorch path improved from about `1.1s/token` to about `0.55s/token`, but FullKV SDPA manual decode on wide A100 is about `39ms/token`, so a `<=2x` claim is not supported.
- The next decision is whether to request Triton/CUDA fused dequant attention permission or continue with a dedicated prefix-boundary mechanism.
