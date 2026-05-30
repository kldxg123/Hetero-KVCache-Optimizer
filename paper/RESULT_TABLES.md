# Workflow3 Result Tables

All tables in this file summarize experiments that were actually run. Artifact
paths are relative to `/home/app-ahr/Hetero-KVCache-Optimizer`.

## Main 128K Source-Aware NIAH Result

Configuration:

- Model: Qwen2.5-7B-Instruct.
- Context: 128K.
- Cache: HeteroKV.
- Retrieval: source-prefiltered token-level path.
- TTL: 24.
- Active retrieval layers: 22-27.
- Required depths: 25%, 50%, 75%, 90%.
- Trials: 2 per depth per seed.
- Seeds: 6004, 4242, 7777.
- Memory policy: 22 GiB PyTorch cap, 30 GiB own-process fuse.
- Hardware: A100 server, used as a 4090-like memory-envelope testbed.

| Seed | GPU | Depths | Trials | Correct | Mean Decode | Median Decode | Ratio vs FullKV | Monitor Peak | Artifact |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 6004 | 3 | 25/50/75/90 | 2 each | 8/8 | 97.85 ms/step | 97.80 ms/step | 1.87x | 22348 MB | `experiments/niah_128k_required4_trials2_sourceprefilter_ttl24_layers22_27_seed6004_gpu3_20260529_auto.json` |
| 4242 | 2 | 25/50/75/90 | 2 each | 8/8 | 98.45 ms/step | 98.46 ms/step | 1.88x | 22348 MB | `experiments/niah_128k_required4_trials2_sourceprefilter_ttl24_layers22_27_seed4242_gpu2_20260529_auto.json` |
| 7777 | 3 | 25/50/75/90 | 2 each | 8/8 | 98.07 ms/step | 97.93 ms/step | 1.88x | 22348 MB | `experiments/niah_128k_required4_trials2_sourceprefilter_ttl24_layers22_27_seed7777_gpu3_20260529_auto.json` |

Aggregate:

| Metric | Value |
| --- | ---: |
| Accuracy | 24/24 |
| Depth 25% | 6/6 |
| Depth 50% | 6/6 |
| Depth 75% | 6/6 |
| Depth 90% | 6/6 |
| Mean decode | 98.12 ms/step |
| Median decode | 97.98 ms/step |
| Decode std | 0.85 ms/step |
| Mean prefill | 48.95 s |
| Mean elapsed | 51.41 s |
| Ratio vs wide-memory FullKV A100 reference | 1.88x |

Mechanism evidence:

| Evidence | Value |
| --- | --- |
| Retrieval active layers | 22-27 |
| Method-D events | 150 per row |
| Source prefilter tail sample | 1 of 60 DRAM chunks |
| Safety fuse | No 30 GiB trigger |

## Latency Reference

| Variant | Context | Result | Prefill | Decode | Max Reserved | Monitor Peak | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FullKV SDPA manual decode, wide-memory A100 reference | 128K | 1/1 | 28.72 s | 52.25 ms/step | 62.9629 GiB | 42362 MB | `experiments/niah_fullkv_128k_cap75_sdpa_manual_latency_refresh_gpu1_20260529_auto.json` |

Interpretation:

- The FullKV reference is a speed/quality reference on wide-memory A100.
- It is not a 24G-survival baseline.
- FullKV reserved roughly 62.96 GiB in this reference setup, so it is not a
  4090-24G survival path.

## FullKV 22 GiB-Cap Negative Control

This control uses the same 128K Qwen2.5-7B-Instruct NIAH setting but runs the
FullKV baseline under the 22 GiB PyTorch cap. It is expected to fail and is
used only to verify that the full-cache path is not a 24G survival baseline.

| Variant | Context | Cap | Depth / Trial | Outcome | Max Allocated | Max Reserved | Monitor Peak | Monitor Killed | Artifact |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| FullKV baseline, 22 GiB cap | 128K | 22 GiB | 25%, 1 trial | CUDA OOM | 19.7629 GiB | 20.6191 GiB | 13.5801 GiB | False | `experiments/niah_fullkv_128k_cap22_expected_oom_seed6004_gpu1_20260530_auto.json` |

Key error text:

- PyTorch reports that the process had 20.90 GiB in use against a 22.00 GiB
  allowance and failed when trying to allocate another 1.75 GiB.
- The external monitor did not kill the run; no 30 GiB fuse trigger occurred.
- The monitor sampled 13.58 GiB because the OOM happened between 5-second
  monitor samples. Use the PyTorch OOM accounting for the immediate failure
  point and the monitor field for safety-fuse evidence.

Interpretation:

- This directly supports the claim that 128K FullKV is not viable under the
  22 GiB memory envelope used for the HeteroKV survival proof.
- This is not an accuracy result and should not be mixed into NIAH quality
  averages.

## 128K Memory Curve Evidence

Source:

- `experiments/niah_128k_required4_trials2_sourceprefilter_ttl24_layers22_27_seed6004_gpu3_20260529_auto.log`
- Parsed into `paper/data/workflow3_summary.json`.

| Metric | Value |
| --- | ---: |
| Parsed prefill chunks | 64 |
| First chunk | [0:2048] |
| Last chunk | [129024:131066] |
| Max active HBM KV length | 8192 tokens |
| Final DRAM compressed KV length | 122880 tokens |
| Max torch reserved | 21.33 GiB |
| Max nvidia-smi process memory | 21.82 GiB |

Interpretation:

- The curve directly supports the O(1) active-HBM claim for the promoted
  seed6004 128K run.
- DRAM-side compressed KV grows with context, as expected.
- This is a real log-derived curve, not a reconstructed schematic.

## WikiText-2 PPL

Configuration:

- Dataset: WikiText-2.
- Metric: real PPL from model loss.
- SourceCopy: disabled.
- Tokens: 14336.
- Loss suffix: 2048.
- Hetero cache config: sink 64, tail 4096, chunk 2048.
- Retrieval config recorded: TTL12, reuse source threshold 35.

| Variant | FullKV PPL | HeteroKV PPL | Delta | Hetero Max Reserved | Own Process Peak | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| WikiText-2 real PPL, SDPA, SourceCopy-disabled | 2.9706 | 3.0063 | +1.20% | 19.2754 GiB | 20.248 GiB | `experiments/ppl_14k_prefix12288_tail4096_gate5_top1_nofusion_sdpa_ttl12_sourcecopy_disabled_allowcoexist_gpu3_20260529_auto.json` |
| WikiText-2 real PPL 16K, SDPA, SourceCopy-disabled | 4.9896 | 5.0723 | +1.66% | 19.2754 GiB | 18.0605 GiB | `experiments/ppl_16k_prefix14336_tail4096_gate5_top1_nofusion_sdpa_ttl12_sourcecopy_disabled_gpu1_20260530_auto.json` |
| WikiText-2 real PPL 16K offset32768, SDPA, SourceCopy-disabled | 6.2955 | 6.3237 | +0.45% | 19.2754 GiB | 18.8984 GiB | `experiments/ppl_16k_offset32768_prefix14336_tail4096_gate5_top1_nofusion_sdpa_ttl12_sourcecopy_disabled_gpu1_20260530_auto.json` |

Mechanism and memory:

| Metric | Value |
| --- | ---: |
| Method-D event count | 512 |
| Max active HBM tokens | 6208 |
| DRAM entries | 112 |
| DRAM bytes | 245891072 |

Claim boundary:

- This supports controlled semantic loss on the tested 14K PPL setup.
- This supports controlled semantic loss on three tested PPL suffix setups
  (14K, 16K from the start of WikiText-2, and 16K from token offset 32768),
  all with SourceCopy disabled.
- This is not a 128K PPL claim.
- This does not validate SourceCopy/source-prefilter for general-language PPL.

## Source-Aware Versus Exact-Copy Reranker Ablation

This table is included to prevent overclaiming the source-aware NIAH result as
pure dot-product retrieval. Both rows use the same 128K memory envelope and are
copy-task variants, but the SourceCopy row adds an exact-string reranker on top
of the retrieval substrate.

| Variant | Length | Depths / Trials | Accuracy | Peak Process Memory | Max Reserved | Active HBM Tokens | DRAM Entries | Artifact |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Source-aware retrieval, SourceCopy disabled | 128K | 25/50, 2 each | 3/4 | 21.8242 GiB | 21.3262 GiB | 12352 | 1680 | `experiments/niah_128k_depth25_50_trials2_main_nosourcecopy_driver_gpu3_20260529_auto.json` |
| Source-aware retrieval + SourceCopy boost20 | 128K | 25/50, 2 each | 4/4 | 21.8262 GiB | 21.3262 GiB | 12352 | 1680 | `experiments/niah_128k_depth25_50_trials2_main_sourcecopy_boost20_driver_gpu3_20260529_auto.json` |

Interpretation:

- Source-aware retrieval without SourceCopy is a real non-oracle ablation but
  is weaker on this hard same-case 25/50 setting.
- SourceCopy improves exact string output without changing the memory envelope.
- This supports a two-layer method description: retrieval locates relevant
  source spans, while SourceCopy is a task-specific exact-string reranker.
- This table must not be used to claim pure KV-only dot-product retrieval
  solves 128K NIAH.

## Earlier Pure Dot-Product 128K Attempts

These runs are retained as negative evidence. They are useful reviewer-facing
evidence that the final source-aware path was not chosen by hiding failed pure
retrieval attempts.

| Variant | Result | Mean Decode | Monitor Peak | Monitor Killed | Treatment |
| --- | ---: | ---: | ---: | ---: | --- |
| clean current top8, qhist64, no source features | 0/4 | 1005.04 ms/step | 21.8242 GiB | False | Current clean negative control |
| keep_tail8192, token window 64, top2 | 0/4 | n/a | 21.0039 GiB | False | Failed pure dot-product attempt |
| keep_tail8192, token window 64, top8 | 1/4 | n/a | 21.1445 GiB | False | Failed pure dot-product attempt |
| keep_tail8192, token window 64, top2, qhist64 | 0/4 | n/a | 21.0039 GiB | False | Failed pure dot-product attempt |
| keep_tail16384, token window 64, top2, qhist64 | 0/4 | n/a | 21.0000 GiB | False | Failed pure dot-product attempt |

Artifacts:

- `experiments/experiment_tracker_workflow2_128k_keep8192_fp32qk_dot_top2_win64_20260527_210444.json`
- `experiments/experiment_tracker_workflow2_128k_keep8192_fp32qk_dot_top8_win64_20260527_211805.json`
- `experiments/experiment_tracker_workflow2_128k_keep8192_fp32qk_dot_top2_win64_qhist64_20260527_225330.json`
- `experiments/experiment_tracker_workflow2_128k_keep16384_fp32qk_dot_top2_win64_qhist64_20260527_231620.json`
- `experiments/niah_128k_depth25_50_trials2_pure_dotproduct_clean_seed6004_gpu1_20260530_auto.json`
- `experiments/experiment_tracker_niah_128k_depth25_50_trials2_pure_dotproduct_clean_seed6004_gpu1_20260530_auto.json`

Interpretation:

- These runs stayed below the 30 GiB fuse but failed quality.
- The clean current top8/qhist64 run produced `000000` on all four rows.
- They support the current claim boundary: pure token-level dot-product was not
  enough in these earlier 128K configurations.
- The clean current run fixes the old shared-child-output concern for this
  specific top8/qhist64 negative control.

## Optional Edge Depths

| Variant | Seeds | Depths | Trials | Correct | Depth 0% | Depth 99% | Peak | Artifact |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| HeteroKV source-prefilter 22-27 | 6004/4242/7777 | 0/99 | 2 each | 6/12 | 0/6 | 6/6 | 22348 MB | `experiments/niah_128k_optional0_99_trials2_sourceprefilter_ttl24_layers22_27_seed*_gpu*_20260529_auto.json` |
| FullKV wide-memory discriminativeness check | 6004 | 0/99 | 2 each | 2/4 | 0/2 | 2/2 | 42362 MB | `experiments/niah_128k_optional0_99_trials2_fullkv_cap75_sdpa_manual_seed6004_gpu1_20260529_auto.json` |

Interpretation:

- 99% is a valid optional edge-depth pass.
- 0% is non-discriminative under the current template because FullKV also
  fails it.

## Generate Compatibility

| Target Tokens | Actual Input Tokens | Output Check | Elapsed | Max Allocated | Max Reserved | Max HBM Tokens | DRAM Entries | DRAM Bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 1585 | True | 1.14 s | 15.10 GiB | 15.36 GiB | 4160 | 0 | 0 |
| 4096 | 3144 | True | 1.72 s | 16.24 GiB | 17.07 GiB | 4160 | 28 | 32897536 |
| 8192 | 6273 | True | 1.79 s | 18.06 GiB | 20.01 GiB | 4160 | 28 | 126817600 |

Artifact:

- Tracker: `experiments/experiment_tracker_stage2_generate_smoke_2k4k8k_after_fix_20260529_auto.json`.
- Child output: `experiments/stage2_smoke.json`.
- External monitor peak: 15.18 GiB.
- Status: ok.

Interpretation:

- HF `generate()` compatibility is validated for 2K/4K/8K smoke contexts.
- This is an API compatibility test, not the main 128K latency result.

## Failed Or Rejected Ideas

| Idea or Run | Failure Mode | Decision |
| --- | --- | --- |
| Parallel seed4242 and seed7777 prefilter run before output-path fix | Child output path clobbered one result | Excluded; sequential reruns are valid |
| First direct seed7777 wrapper | Missing `CUDA_VISIBLE_DEVICES`, exited before GPU use | Wrapper failure only |
| Sink1024 optional 0/99 diagnostic | 0/4, broke 99% too | Reject |
| Purely treating optional 0% as a HeteroKV failure | FullKV also failed 0% | Mark non-discriminative |
| Reporting source-prefilter NIAH as pure dot-product retrieval | Mechanism uses source-aware filtering | Disallowed |
