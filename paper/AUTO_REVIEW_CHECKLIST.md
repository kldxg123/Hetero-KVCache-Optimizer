# Workflow3 Automatic Review Checklist

Use this checklist before each paper or report revision.

## Evidence Integrity

- [ ] Every numeric claim points to an actual artifact path.
- [ ] Failed, invalid, and diagnostic runs are preserved.
- [ ] Oracle results are never mixed into real method tables.
- [ ] Source-aware NIAH results are labeled as source-aware.
- [ ] SourceCopy-disabled PPL remains separate from source-aware NIAH.
- [ ] A100-under-cap latency is not described as native RTX 4090 latency.

## Main Claim Checks

- [ ] The project is described as Approximate Long-Context Cache.
- [ ] The draft does not claim lossless full-KV recovery.
- [ ] The draft does not claim token-level logits equivalence.
- [ ] The draft states that active HBM KV is bounded by policy.
- [ ] The draft states that DRAM-side compressed storage may grow with context.

## Result Checks

- [ ] Main NIAH result is 24/24 on depths 25/50/75/90.
- [ ] Optional 99% result is 6/6 and separated from required depths.
- [ ] Optional 0% is marked non-discriminative under the current template.
- [ ] PPL table includes the validated +1.20%, +1.66%, +0.45%, and +3.14%
      SourceCopy-disabled WikiText-2 deltas.
- [ ] Latency result is 98.12 ms/step vs 52.25 ms/step, 1.88x.
- [ ] Memory result reports the 22 GiB PyTorch cap and 30 GiB fuse.

## Missing Before Submission

- [ ] Real RTX 4090 survival/latency rerun, if the paper claims 4090 latency.
- [ ] Redesigned 0% NIAH benchmark where FullKV passes.
- [ ] Second-corpus PPL validation if a broad language-quality claim is made.
- [ ] Clean figure scripts for memory curves and result plots.
- [ ] Appendix table for rejected ideas and invalid runs.
