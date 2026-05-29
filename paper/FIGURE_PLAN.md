# Figure Plan

## Figure 1: System Overview

Purpose:

- Show the Approximate Long-Context Cache design.
- Make clear that HBM stores only active short KV.
- Show DRAM-side compressed KV growth.

Elements:

- Input chunks.
- Sink tokens.
- Tail tokens.
- Heavy-hitter retained tokens.
- Evicted tokens to compressed DRAM storage.
- Query-aware retrieval path back into active cache.

Claim supported:

- HeteroKV is a bounded-HBM approximate cache, not a lossless full-KV replay.

## Figure 2: Memory Survival Curve

Purpose:

- Show active HBM KV and process memory across 128K prefill.

Curves:

- Active HBM KV tokens.
- DRAM compressed KV tokens or bytes.
- `torch.cuda.max_memory_reserved`.
- external monitor process memory.

Required source data:

- Promoted 128K source-prefilter artifacts.
- Memory logs from chunked prefill and retrieval.

Claim supported:

- Active HBM KV remains bounded while DRAM-side storage grows.

## Figure 3: Required-Depth NIAH Accuracy

Purpose:

- Show multi-depth recall at 128K.

Bars:

- 25%: 6/6.
- 50%: 6/6.
- 75%: 6/6.
- 90%: 6/6.
- Optional 99%: 6/6, visually separated.
- Optional 0%: mark as non-discriminative because FullKV also fails.

Claim supported:

- The promoted source-aware path solves the required NIAH depths.

## Figure 4: Latency Breakdown

Purpose:

- Explain where time is spent.

Bars:

- FullKV A100 wide-memory reference decode: 52.25 ms/step.
- HeteroKV promoted path decode: 98.12 ms/step.
- Ratio: 1.88x.

Optional sub-bars if raw logs expose them:

- prefill time;
- retrieval scoring;
- dequant/transfer;
- decode compute.

Claim supported:

- Latency growth is bounded and explainable for the validated path.

## Figure 5: PPL Semantic Loss

Purpose:

- Show controlled semantic loss on WikiText-2.

Bars:

- FullKV PPL: 2.9706.
- HeteroKV PPL: 3.0063.
- Delta: +1.20%.

Required label:

- SourceCopy disabled.
- 14K-token PPL setup.

Claim supported:

- Controlled semantic loss in the tested PPL setting.

## Figure 6: Ablation Ladder

Purpose:

- Show why late-layer retrieval was selected.

Rows:

- layers 12-27: 4/4, 131.2 ms/step, 2.51x.
- layers 16-27: 4/4, 118.5 ms/step, 2.27x.
- layers 20-27: 4/4, 105.2 ms/step, 2.01x.
- layers 21-27: 4/4, 104.9 ms/step, 2.01x.
- layers 22-27: 4/4, 101.0 ms/step, 1.93x.

Claim supported:

- Late-layer source-aware retrieval preserves NIAH accuracy while reducing
  latency.

## Required Styling Notes

- Every plot must include exact run identifiers in the caption or appendix.
- Diagnostic/oracle runs must use a different color or label from real method
  runs.
- A100-under-cap results must not be labeled as RTX 4090 measured latency.
