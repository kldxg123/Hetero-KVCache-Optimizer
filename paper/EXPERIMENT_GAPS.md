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
- Earlier pure dot-product 128K attempts are preserved and failed at 0/4 to 1/4
  while staying below the 30 GiB fuse.
- A current clean 128K pure dot-product top8/qhist64 negative control has now
  been run with unique outputs: 0/4, mean decode 1005.04 ms/step, peak
  21.8242 GiB, no monitor kill.
- A pure dot-product scaling diagnostic has been run at 16K/32K/64K with
  required depths 25/50/75/90 and 2 trials each. It stayed under the 22 GiB
  cap but failed quality at 11/24 overall.

Needed:

- If the paper still wants to claim a pure retrieval baseline, compare
  source-aware prefilter and pure dot-product on exactly matched shorter
  prompts.
- Otherwise, treat this gap as resolved by claim narrowing: the promoted method
  is source-aware retrieval plus token-level scoring, not pure dot-product
  alone.

Purpose:

- Prevent reviewers from saying the main mechanism is only source-copy prompt
  exploitation.

## Gap 3: More General PPL Evidence

Current status:

- SourceCopy-disabled WikiText-2 setups give +1.20% at 14K, +1.66% at 16K,
  +0.45% at 16K offset32768, and +3.14% at 32K.
- SourceCopy-disabled IMDb 16K gives +1.09%.
- All are below the 5% semantic-loss budget.

Needed:

- Add a true 128K PPL-style diagnostic only if runtime and memory safety allow.
- Add more non-WikiText corpora only if they already exist locally or network
  access is reliable.

Purpose:

- Support semantic-loss robustness beyond NIAH.

## Gap 4: Memory Curves

Current status:

- Tables record peak memory and bounded active HBM evidence.
- Raw-log-derived memory curves have been generated for the promoted seed6004
  128K required-depth run.

Needed:

- Improve final paper styling.
- Add another seed's curve only if reviewers ask for non-cherry-picked visual
  evidence.

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
