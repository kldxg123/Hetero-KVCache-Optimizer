# Experiment Gaps Before Submission

This file lists the missing experiments and the exact reason each is needed.

## Gap 1: Real RTX 4090 Measurement

Current status:

- A100 under a 22 GiB PyTorch cap and 30 GiB fuse supports a 4090-like memory
  envelope claim.
- It does not prove native RTX 4090 latency or exact allocator behavior.

Needed:

- 128K HeteroKV survival on a real RTX 4090 24GB card.
- Record `nvidia-smi`, PyTorch reserved/allocated memory, active HBM tokens,
  DRAM bytes, NIAH accuracy, and decode latency.

Fallback if unavailable:

- Keep the claim as "A100 under a 4090-like memory envelope."

## Gap 2: Pure Retrieval Baseline

Current status:

- The strongest NIAH result uses source-aware prefiltering plus token-level
  retrieval.
- The PPL result disables SourceCopy and remains separate.

Needed:

- A clean pure dot-product-only retrieval table at 16K/32K/128K where feasible.
- Compare against source-aware prefilter on the same prompts.

Purpose:

- Prevent reviewers from saying the main mechanism is only source-copy prompt
  exploitation.

## Gap 3: More General PPL Evidence

Current status:

- One SourceCopy-disabled WikiText-2 setup gives +1.20% PPL delta.

Needed:

- More WikiText-2 slices or a second text corpus.
- If runtime permits, longer prefix contexts.

Purpose:

- Support semantic-loss robustness beyond NIAH.

## Gap 4: Memory Curves

Current status:

- Tables record peak memory and bounded active HBM evidence.
- Paper still needs curves.

Needed:

- Extract per-chunk active HBM tokens.
- Extract per-chunk DRAM entries/bytes.
- Extract per-run PyTorch allocated/reserved memory and monitor memory.

Purpose:

- Visually prove bounded HBM and growing DRAM-side storage.

## Gap 5: Optional 0% NIAH

Current status:

- FullKV fails current 0% NIAH.

Needed:

- Redesign the 0% prompt/template.
- First require FullKV to pass.
- Then evaluate HeteroKV.

Purpose:

- Avoid a misleading benchmark failure in the main table.
