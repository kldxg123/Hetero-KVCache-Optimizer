# Top-Conference Readiness Review

Review mode: fully automatic, evidence-first.

Current decision: not yet ready to claim top-conference quality.

The current evidence is strong enough to start Workflow3 paper drafting, but it
is not yet enough for a confident top-conference submission without additional
experiments and clearer methodological separation.

## Strengths

- The main 128K source-aware NIAH result is real, multi-seed, and memory-safe:
  24/24 across required depths under the 22 GiB PyTorch cap and 30 GiB
  own-process fuse.
- The promoted source-aware path meets the stated latency target in the A100
  memory-envelope setting: 98.12 ms/step versus 52.25 ms/step, or 1.88x.
- The PPL result is a real loss/PPL measurement, not a proxy: +1.20% versus
  FullKV on the SourceCopy-disabled 14K WikiText-2 setup.
- Failed and invalid runs are recorded, including output clobbering, wrapper
  failures, rejected sink1024, and the non-discriminative 0% NIAH case.
- The claim boundary is clear: approximate cache, not lossless full attention.

## Blocking Weaknesses For A Top-Tier Claim

### 1. Hardware Claim Is Still Indirect

The project goal is framed around 4090-24G survival. Current survival evidence
is A100 under a 4090-like memory envelope. That is scientifically useful, but a
reviewer can still ask for a real RTX 4090 run.

Required fix before a strong hardware claim:

- Run at least one real RTX 4090 24GB 128K survival test.
- If real 4090 is unavailable, present the A100-under-cap result as an
  engineering surrogate and explicitly weaken the hardware claim.

### 2. Source-Aware Path Must Be Separated From Pure KV Retrieval

The strongest NIAH result uses source-aware prefiltering. This is valid as a
task-aware approximate cache mechanism, but it cannot be sold as pure
Query-Key dot-product retrieval.

There is already a controlled same-case ablation where source-aware retrieval
with SourceCopy disabled reaches 3/4 and SourceCopy boost20 reaches 4/4 on
128K 25%/50% trials. This is useful evidence, but still not enough for a broad
pure-retrieval claim.

A current clean pure dot-product top8/qhist64 negative control was also run on
128K 25%/50% trials2: it stayed under the memory fuse but scored 0/4 and
generated `000000` for all rows. This strengthens the honesty of the ablation
story, but it also shows that the top-conference method claim must be framed
around source-aware approximate caching rather than pure QK retrieval alone.

Required fix before a strong method claim:

- Extend the clean pure dot-product-only table to shorter contexts if claiming
  a broader retrieval scaling story.
- Label source-aware retrieval as the main high-accuracy NIAH path.
- Explain when source-aware metadata is available and why it is not answer
  leakage.

### 3. PPL Evidence Is Still Narrow, But Improved

The PPL result is encouraging and now includes two SourceCopy-disabled suffix
setups: +1.20% at 14K and +1.66% at 16K. This is stronger than a single PPL
point, but it is still not a broad long-context language-quality claim.

Required fix before a strong quality claim:

- Add more WikiText-2 offsets or a second corpus.
- Increase context length if feasible.
- Keep SourceCopy/source-aware features disabled unless testing them explicitly.

### 4. 0% NIAH Needs A Redesign Or Removal

The current 0% NIAH case is non-discriminative because FullKV also fails it.
That is not a HeteroKV failure, but leaving it unresolved may distract
reviewers.

Required fix before submission:

- Redesign the 0% template so FullKV passes.
- Rerun HeteroKV on the redesigned 0% case.
- Or remove 0% from the main benchmark and keep it in the appendix as a
  template pathology.

### 5. Figures Need Raw-Curve Backing

The paper has result tables but still needs paper-grade plots.

Required fix before submission:

- Generate memory curves from artifact logs.
- Plot active HBM KV tokens and DRAM bytes over prefill.
- Plot latency ablations and PPL deltas.
- Include run IDs in captions.

## Reviewer-Risk Questions

Likely reviewer questions:

1. Does the source-aware prefilter leak the answer or use needle position?
2. Does PyTorch memory fraction accurately model a 4090?
3. How does the method behave without source-aware exact-copy support?
4. Is the PPL result robust beyond one 14K setup?
5. Why should NIAH accuracy imply broad long-context quality?
6. What are the exact DRAM bytes and transfer costs?
7. Is the 4-bit storage truly bit-packed or uint8-held int4 values?

## Required Next Experiments

Minimum next experiments before claiming submission-level readiness:

| Priority | Experiment | Purpose | Stop Condition |
| ---: | --- | --- | --- |
| 1 | Pure dot-product-only 128K NIAH table | Separate core retrieval from source-aware assistance | Stop if memory exceeds 30 GiB |
| 2 | Real 4090 24GB survival test | Close the hardware proof gap | Stop if unavailable; mark as limitation |
| 3 | Redesigned 0% NIAH with FullKV pass | Repair benchmark pathology | Stop if FullKV still fails |
| 4 | Additional SourceCopy-disabled PPL slices | Strengthen semantic-loss claim | Stop if PPL delta exceeds 5% |
| 5 | Memory curve extraction | Turn logs into paper figures | Stop if artifacts lack needed fields |

## Current Recommendation

Do not enter final paper-writing-only mode yet. Continue Workflow3 automatic
review by generating figures and appendix materials from existing artifacts,
then run only the missing experiments that do not violate the 30 GiB fuse.
