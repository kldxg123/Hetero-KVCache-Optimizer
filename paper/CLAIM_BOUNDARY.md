# Claim Boundary

This document defines what can and cannot be claimed from the current evidence.
It must be checked before writing any paper, report, abstract, slide, or README
claim.

## Allowed Claims

### Memory Survival

Allowed:

- HeteroKV supports 128K Qwen2.5-7B-Instruct NIAH generation under an A100 run
  constrained to a 4090-like memory envelope.
- The validated runs used a 22 GiB PyTorch memory cap and a 30 GiB own-process
  safety fuse.
- The promoted source-aware path completed required-depth 128K NIAH without
  triggering the 30 GiB fuse.
- Active HBM KV stays bounded by the configured cache policy while compressed
  DRAM-side KV grows with context length.
- A FullKV 128K negative control under the same 22 GiB PyTorch cap fails with
  CUDA OOM, while the monitor does not trigger the 30 GiB safety fuse.

Do not overstate:

- Do not claim this is native RTX 4090 latency unless it is rerun on a real
  RTX 4090.
- Do not claim all CUDA allocations are physically capped by PyTorch alone.
  The cap is a PyTorch allocator cap, not a hardware isolation mechanism.
- Do not treat the FullKV 22 GiB-cap OOM row as a quality or accuracy result;
  it is a survival negative control.

### NIAH Accuracy

Allowed:

- The promoted source-aware path achieved 24/24 on required NIAH depths
  25%, 50%, 75%, and 90%, using three seeds and two trials per depth.
- The optional 99% depth passed 6/6.
- The optional 0% depth is non-discriminative under the current NIAH template,
  because the FullKV wide-memory baseline also failed it.

Do not overstate:

- Do not report the 0% failures as a HeteroKV-specific model failure.
- Do not claim the source-aware NIAH path is pure dot-product retrieval.

### PPL

Allowed:

- The current real WikiText-2 PPL evidence is SourceCopy-disabled.
- The measured PPL delta is +1.20% relative to the FullKV baseline on the
  validated 14K-token decode-suffix setup.
- A second 16K-token SourceCopy-disabled setup has +1.66% relative PPL delta.
- A third 16K-token SourceCopy-disabled setup at WikiText-2 token offset 32768
  has +0.45% relative PPL delta.
- A fourth 32K-token SourceCopy-disabled WikiText-2 setup has +3.14% relative
  PPL delta.
- A fifth 16K-token SourceCopy-disabled IMDb setup has +1.09% relative PPL
  delta.
- This supports a controlled semantic-loss claim for the tested PPL setups.

Do not overstate:

- Do not claim 128K WikiText-2 PPL unless a true 128K PPL experiment is run.
- Do not claim the source-aware exact-copy NIAH mechanism improves general PPL
  unless it is tested separately.
- Do not hide that PPL evidence is still suffix-style evaluation rather than
  true 128K PPL.

### Latency

Allowed:

- The promoted 128K source-aware NIAH path achieved 98.12 ms/step mean decode.
- The wide-memory A100 FullKV reference was 52.25 ms/step.
- The validated path is 1.88x the wide-memory A100 reference for this setting.

Do not overstate:

- Do not call this a true RTX 4090 latency result.
- State the result as A100 under a 4090-like memory envelope.

## Disallowed Claims

The following claims are not supported by the current evidence:

- HeteroKV exactly reproduces native 128K full attention.
- HeteroKV is token-logit equivalent to FullKV.
- Pure token-level dot-product retrieval alone solves all 128K NIAH cases.
- The A100 latency result proves true RTX 4090 latency.
- Optional 0% NIAH is a valid failing point for HeteroKV under the current
  template.
- Diagnostic oracle runs can be reported as real method results.

## Required Language

Use phrasing like:

- "A100 under a 4090-like memory envelope."
- "Source-aware exact-copy/NIAH path."
- "SourceCopy-disabled WikiText-2 PPL."
- "Approximate Long-Context Cache."
- "Controlled semantic loss, not lossless KV recovery."

Avoid phrasing like:

- "lossless 128K full KV replacement."
- "4090 latency proven."
- "pure dot-product retrieval solves NIAH."
- "equivalent to full attention."
