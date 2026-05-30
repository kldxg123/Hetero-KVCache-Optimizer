# Caption Bank

Use these captions as paper-ready starting points. Every caption includes the
claim boundary that reviewers should see.

## Figure: System Overview

HeteroKV is an approximate long-context cache rather than a lossless full-KV
reconstruction. During prefill, only a bounded active KV set remains in GPU HBM
(sink, tail, heavy-hitter, and retrieved tokens), while evicted tokens are
stored in compressed DRAM-side records and can be retrieved by query-aware
mechanisms during decode.

## Figure: 128K NIAH Accuracy

Required-depth 128K NIAH accuracy for Qwen2.5-7B-Instruct under the promoted
source-aware HeteroKV configuration. The run uses an A100 server constrained to
a 4090-like memory envelope with a 22 GiB PyTorch cap and a 30 GiB own-process
fuse. Across seeds 6004, 4242, and 7777, the method reaches 24/24 accuracy on
depths 25%, 50%, 75%, and 90%.

## Figure: Memory Curve

Log-derived 128K prefill memory curve from the promoted seed6004 run. Active
HBM KV reaches 8192 tokens and stays bounded, while compressed DRAM-side KV
grows to 122880 tokens. The same log records max torch reserved memory of
21.33 GiB and max nvidia-smi process memory of 21.82 GiB.

## Figure: 22GiB-Cap Survival Outcome

128K survival outcome under the 22 GiB PyTorch memory cap. The promoted
HeteroKV source-aware path completes the required-depth 128K NIAH suite without
triggering the 30 GiB fuse, while the FullKV 128K baseline fails with CUDA OOM
when PyTorch reports 20.90 GiB in use against the 22.00 GiB allowance. This is
a survival-control figure, not an accuracy comparison.

## Figure: Latency

Decode latency comparison between the promoted source-aware HeteroKV path and
the wide-memory A100 FullKV reference. HeteroKV averages 98.12 ms/step and the
FullKV reference averages 52.25 ms/step, giving a 1.88x ratio in the A100
memory-envelope setting. This is not a native RTX 4090 latency measurement.

## Figure: PPL

SourceCopy-disabled WikiText-2 PPL comparison. HeteroKV shows +1.20% relative
PPL delta on the 14K suffix setup, +1.66% on the 16K suffix setup, and +0.45%
on a 16K setup starting at WikiText-2 token offset 32768. A 32K suffix setup
has +3.14% relative PPL delta. An IMDb 16K suffix setup has +1.09% relative
PPL delta. All are below the 5% semantic-loss budget. These PPL runs do not
use the source-aware exact-copy NIAH path.

## Figure: SourceCopy Ablation

Controlled 128K exact-copy ablation on the 25%/50% NIAH subset. Source-aware
retrieval without SourceCopy reaches 3/4, while SourceCopy boost20 reaches 4/4
under the same memory envelope. This supports describing SourceCopy as a
task-specific exact-string reranker layered on top of retrieval.

## Figure: Pure Dot-Product Negative Control

Pure Query-Key dot-product negative controls at 128K. The clean current top8
qhist64 configuration scores 0/4, stays below the 30 GiB fuse, and outputs
`000000` on all rows. Earlier pure-dot variants also remain at 0/4 to 1/4.
These negative results motivate the source-aware method boundary.

## Figure: Pure Dot-Product Scaling Diagnostic

Pure Query-Key dot-product retrieval without source-aware filtering,
SourceCopy, oracle ranges, or Triton kernels. Under the same 22 GiB cap, it
scores 4/8 at 16K, 5/8 at 32K, and 2/8 at 64K across required depths
25%, 50%, 75%, and 90%. The result stays inside the memory envelope but fails
quality, supporting the boundary that the promoted path is source-aware.
