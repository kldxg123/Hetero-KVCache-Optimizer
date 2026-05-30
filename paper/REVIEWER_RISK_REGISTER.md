# Reviewer Risk Register

This file lists likely reviewer concerns and the current answer strategy.

## R1: "This is not really a 4090 result."

Risk: high.

Current evidence:

- A100 constrained to a 4090-like memory envelope.
- 22 GiB PyTorch cap.
- 30 GiB own-process fuse.
- Promoted 128K run peaks at 21.82 GiB nvidia-smi process memory.

Response:

- Agree and state the hardware boundary plainly.
- Claim "A100 under a 4090-like memory envelope."
- Do not claim native RTX 4090 latency.
- Add a real RTX 4090 rerun if hardware becomes available.

## R2: "Source-aware retrieval is answer leakage."

Risk: high.

Current evidence:

- Source-aware retrieval uses source/query token overlap.
- It does not use target labels, answer spans, or needle ranges.
- Pure dot-product negative controls are reported separately.
- A pure dot-product scaling diagnostic without source-aware filtering reaches
  only 11/24 across 16K/32K/64K, so the paper does not hide this dependency.

Response:

- Define exactly what metadata is available to the method.
- Show the pure-dot negative controls.
- Show SourceCopy-disabled PPL separately.
- Avoid calling the method pure KV-only retrieval.

## R3: "NIAH is too narrow."

Risk: medium-high.

Current evidence:

- NIAH required-depth result is strong: 24/24.
- PPL has five SourceCopy-disabled suffix setups: +1.20%, +1.66%, +0.45% at
  token offset 32768, +3.14% at WikiText-2 32K, and +1.09% on IMDb 16K.

Response:

- Present NIAH as exact-recall stress evidence, not broad language-quality
  proof.
- Use PPL to show controlled semantic loss outside the exact-copy path.
- Avoid presenting suffix PPL as true 128K PPL.

## R4: "Why did 0% NIAH fail?"

Risk: medium.

Current evidence:

- FullKV also fails the current 0% template.
- FullKV outputs `000000` for the seed6004 0% rows.

Response:

- Mark current 0% as non-discriminative.
- Keep it out of the main table.
- Redesign only if 0% must be a main benchmark.

## R5: "The memory curve may be cherry-picked."

Risk: medium.

Current evidence:

- Multi-seed NIAH peak memory is consistently 22348 MB.
- The parsed memory curve comes from the promoted seed6004 required-depth log.
- It has 64 prefill chunks, flat active HBM after 8192 tokens, and growing DRAM
  compressed KV.

Response:

- Include artifact paths.
- Add another seed's memory curve if asked.
- State that the curve is representative and peak values match the multi-seed
  table.

## R6: "The PPL setup is still short."

Risk: medium.

Current evidence:

- WikiText-2 14K/16K/16K offset32768/32K and IMDb 16K SourceCopy-disabled PPL
  pass within 5%.

Response:

- Avoid claiming 128K PPL.
- State that PPL is a semantic-loss sanity test.
- Add longer-context or second-corpus PPL if runtime and dataset setup allow.

## R7: "Why not optimize with Triton?"

Risk: low-medium.

Current evidence:

- The promoted source-aware path already meets the 2x latency target in the
  A100 memory-envelope setting.
- Custom Triton/CUDA kernels were intentionally avoided for the default
  validated path.

Response:

- Present PyTorch fallback as the conservative validated path.
- Reserve Triton fused dequant attention as future work if real 4090 latency
  requires it.
