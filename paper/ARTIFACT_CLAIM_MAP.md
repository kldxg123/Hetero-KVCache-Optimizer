# Artifact Claim Map

This file maps every promoted Workflow3 claim to concrete artifacts. A claim
must not be used in the paper unless it has a row here.

## Promoted Claims

| Claim | Status | Primary artifact | Supporting artifact | Boundary |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct can run 128K NIAH with HeteroKV under a 4090-like memory envelope on the A100 server | Supported | `experiments/niah_128k_required4_trials2_sourceprefilter_ttl24_layers22_27_seed6004_gpu3_20260529_auto.json` | seed4242 and seed7777 repeats | This is A100 under a 22 GiB PyTorch cap and 30 GiB own-process fuse, not native RTX 4090 hardware |
| Required-depth 128K NIAH succeeds at 25%, 50%, 75%, and 90% | Supported | three required-depth JSON files for seeds 6004, 4242, 7777 | `paper/data/workflow3_summary.json` | Uses the source-aware promoted path |
| Active HBM KV length stays approximately O(1) during 128K prefill | Supported | `experiments/niah_128k_required4_trials2_sourceprefilter_ttl24_layers22_27_seed6004_gpu3_20260529_auto.log` | `paper/figures/memory_curve_tokens.svg` | Curve is from the promoted seed6004 run; peak memory is consistent with the multi-seed table |
| DRAM compressed KV grows while active HBM KV is bounded | Supported | seed6004 required-depth log | `paper/figures/memory_curve_tokens.svg` | DRAM growth is expected and should not be described as O(1) |
| Promoted source-aware path meets the 2x A100 reference latency target | Supported | required-depth JSON files | FullKV reference JSON | FullKV reference is wide-memory A100, not 24G survival |
| FullKV 128K is not a 24G survival baseline | Supported | `experiments/niah_fullkv_128k_cap75_sdpa_manual_latency_refresh_gpu1_20260529_auto.json` | `paper/RESULT_TABLES.md` | Use as speed/reference quality only |
| FullKV 128K fails under the 22 GiB memory envelope | Supported | `experiments/niah_fullkv_128k_cap22_expected_oom_seed6004_gpu1_20260530_auto.json` | tracker `experiments/experiment_tracker_niah_fullkv_128k_cap22_expected_oom_seed6004_gpu1_20260530_auto.json` | Negative survival control, not a quality result |
| SourceCopy-disabled PPL degradation is controlled on tested suffix setups | Supported | WikiText-2 14K/16K/16K-offset/32K and IMDb 16K PPL JSON artifacts | `paper/figures/ppl_relative_delta_by_context.svg` | This is not a 128K PPL claim |
| SourceCopy improves exact-string NIAH output without changing the memory envelope | Supported | SourceCopy-disabled and SourceCopy-boost20 128K ablation JSON files | `paper/figures/sourcecopy_ablation_accuracy.svg` | This is a copy-task reranker result, not general-language PPL evidence |
| Pure token-level dot-product retrieval alone has not solved the 128K NIAH setting | Supported as negative evidence | clean current pure-dot JSON and tracker | older pure-dot trackers | These are negative controls, not final-method failures |
| HF `generate()` compatibility holds for smoke contexts | Supported | `experiments/experiment_tracker_stage2_generate_smoke_2k4k8k_after_fix_20260529_auto.json` | child `stage2_smoke.json` | Smoke test only, not the 128K benchmark |

## Claims That Must Not Be Made Yet

| Forbidden or not-yet-supported claim | Reason |
| --- | --- |
| Native RTX 4090 latency is validated | No real 4090 run has been performed |
| The method is a lossless 128K full-KV replica | Project is an approximate long-context cache |
| Token-level logits match full attention | Not a goal and not tested |
| Pure Query-Key dot-product retrieval is the promoted 128K success path | Clean current pure-dot control is 0/4 |
| The current 0% NIAH result is a valid HeteroKV failure | FullKV also fails the current 0% template |
| 128K WikiText-2 PPL is validated | Current PPL evidence is 14K, 16K, and 32K suffix evaluation |
| Source-aware retrieval uses answer labels | Current method must be described as using source/query signals only; any stronger claim would be incorrect |

## Minimum Evidence Bundle For A Paper Draft

The draft should cite these files together:

- `paper/RESULT_TABLES.md`
- `paper/CLAIM_BOUNDARY.md`
- `paper/REVIEWER_RISK_REGISTER.md`
- `paper/CAPTION_BANK.md`
- `paper/data/workflow3_summary.json`
- `paper/figures/memory_curve_tokens.svg`
- `paper/figures/memory_curve_gib.svg`
- `paper/figures/niah_required_accuracy.svg`
- `paper/figures/latency_ratio.svg`
- `paper/figures/ppl_relative_delta_by_context.svg`
- `paper/APPENDIX_FAILED_IDEAS.md`
