# Reproducibility Notes

This file records the validated Workflow3 reproduction surface. It is intended
for internal auditing before a paper or artifact release.

## Environment

- Remote project: `/home/app-ahr/Hetero-KVCache-Optimizer`.
- Branch: `codex/workflow2-128k-survival-20260528`.
- Model: `Qwen2.5-7B-Instruct` from the server-local model cache.
- Python for real experiments: `/home/app-ahr/miniconda3/bin/python`.
- Hardware used: A100 server.
- Memory envelope: 22 GiB PyTorch cap, 30 GiB own-process fuse.
- Claim wording: "A100 under a 4090-like memory envelope."

## Safety Procedure

Before any GPU run:

1. Check `nvidia-smi`.
2. Use only an idle target GPU.
3. Set `CUDA_VISIBLE_DEVICES` to the physical target GPU.
4. Keep the process-local fuse at 30 GiB.
5. Stop if another user appears on the same GPU or if the process approaches
   the fuse.

## Main 128K NIAH Reproduction Surface

Promoted configuration:

- Length: 131072 tokens.
- Depths: 25%, 50%, 75%, 90%.
- Trials: 2 per depth.
- Seeds: 6004, 4242, 7777.
- Sink tokens: 64.
- Tail budget: 8192.
- Chunk size: 2048.
- Retrieval: source-prefiltered token-level path.
- TTL: 24.
- Active layers: 22-27.
- Triton/CUDA custom kernel: disabled.

Expected aggregate:

- Required-depth NIAH: 24/24.
- Mean decode: about 98.12 ms/step.
- A100 wide-memory FullKV reference: about 52.25 ms/step.
- Ratio: about 1.88x.
- Monitor peak: about 22348 MB.
- Active HBM KV plateau: 8192 tokens after warmup.

## FullKV 22 GiB-Cap Negative Control

Validated setup:

- Length: 131072 tokens.
- Depth: 25%.
- Trial: 1.
- Mode: `full_kv_baseline`.
- Cap: 22 GiB.
- Fuse: 30 GiB.

Expected outcome:

- CUDA OOM before generation succeeds.
- PyTorch reports about 20.90 GiB process memory in use against the 22.00 GiB
  allowance.
- Max reserved in the row is about 20.62 GiB.
- The external monitor does not trigger the 30 GiB fuse.

This row should be reported as a survival negative control, not as NIAH
quality evidence.

## PPL Reproduction Surface

PPL is evaluated with SourceCopy disabled and should be reported separately
from exact-copy NIAH.

Validated setups:

| Setup | FullKV PPL | HeteroKV PPL | Relative delta |
| --- | ---: | ---: | ---: |
| 14K suffix evaluation | 2.9706 | 3.0063 | +1.20% |
| 16K suffix evaluation | 4.9896 | 5.0723 | +1.66% |
| 16K suffix evaluation at token offset 32768 | 6.2955 | 6.3237 | +0.45% |
| 32K suffix evaluation | 6.5289 | 6.7336 | +3.14% |
| IMDb 16K suffix evaluation | 15.2031 | 15.3683 | +1.09% |

Do not report this as 128K PPL.

## Negative Controls

Pure dot-product controls are part of the record:

- clean current top8/qhist64: 0/4, no memory fuse trigger.
- older top2/top8/qhist64 variants: 0/4 to 1/4, no memory fuse trigger.

These results are not discarded. They define the boundary between failed pure
QK retrieval and the promoted source-aware path.

## Known Non-Reproducibility Risks

- GitHub push from the remote server can fail when remote outbound HTTPS is
  unstable.
- HuggingFace metadata requests may fail if remote network is down; cached
  datasets and models were used for the successful PPL run.
- A100 latency is not RTX 4090 latency.
- PyTorch memory fraction is not hard hardware partitioning.

## Artifact Integrity Checklist

Before using a result in the paper:

1. Confirm the JSON artifact exists.
2. Confirm a monitor or log exists for memory-sensitive claims.
3. Confirm the result is included in `paper/RESULT_TABLES.md`.
4. Confirm the claim boundary is included in `paper/CLAIM_BOUNDARY.md`.
5. Confirm failed or contradictory runs are included in
   `paper/APPENDIX_FAILED_IDEAS.md` or `paper/REVIEWER_RISK_REGISTER.md`.
