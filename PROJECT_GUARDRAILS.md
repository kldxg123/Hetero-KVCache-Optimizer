# Hetero-KVCache-Optimizer Guardrails

## Project Position

Hetero-KVCache-Optimizer is an Approximate Long-Context Cache system.  It does
not attempt to reproduce full 128K causal attention exactly at token-logit
level.  The goal is to keep active GPU HBM KV bounded while reducing semantic
loss with Sink, Tail, Heavy-Hitter eviction, 4-bit DRAM KV storage, and
Query-aware token-level dot-product retrieval.

## Required Hardware Claim

The primary claim is: Qwen2.5-7B-Instruct with 128K context can survive under a
4090-like 24G memory envelope.  On the A100 server this must be tested with a
conservative 22 GiB PyTorch memory cap for the HeteroKV acceptance run.

Latency measured on A100 under a 24G memory envelope must not be described as a
strict RTX 4090 latency result.  A real 4090 latency claim requires a real 4090
rerun.

## Hard Restrictions

- Do not modify model weights.
- Do not modify the installed transformers package.
- Do not pad short KV back to the original 128K length.
- Do not keep full DRAM KV in HBM to fix shape mismatches.
- Do not use mean-K, pooled chunk embeddings, or cosine similarity as the main
  retrieval path.
- Do not add or enable custom Triton/CUDA kernels before correctness, NIAH, PPL,
  and memory stability are proven.
- Acceptance runs default to `enable_triton=False`.
- Use `CUDA_VISIBLE_DEVICES=1` for the target physical GPU; inside the process
  the device is `cuda:0`.

## Shared Server Safety

Before any heavy run, inspect `nvidia-smi`.  If the target GPU already has other
users' processes, skip the run and report `skipped due to shared-server safety`.
Do not kill, pause, or interfere with other users' jobs.  Baseline uncapped A100
runs are allowed only when the target GPU is idle.

## Academic Authenticity

All claims must be real, reproducible, and traceable to logged experiments.
Do not convert a diagnostic run, oracle run, or partial success into an
acceptance claim.  Every reported result must state:

- the exact mode/configuration used;
- whether it is oracle/diagnostic or the real dot-product retrieval path;
- the model, context length, GPU index, memory cap, and monitor fuse;
- the observed `torch.cuda.max_memory_allocated`,
  `torch.cuda.max_memory_reserved`, and `nvidia-smi` process memory;
- the exact NIAH/PPL/latency outcome, including failures.

Failed ideas and negative results must be kept in the reports so the project
does not repeat the same mistakes.  A mechanism is considered proven only when
the code path, logs, and outputs all agree: physical short KV, DRAM entries,
retrieval selections, attention/retrieval evidence, and final generated answer.

Do not overstate hardware conclusions.  A100 results under a 22 GiB cap support
the claim "A100 under a 4090-like memory envelope"; they do not prove native
RTX 4090 latency.  Real 4090 latency requires a real 4090 rerun.

If source-aware token metadata is enabled, report it explicitly.  A run using
`method_d_source_token_boost` is not a pure KV-only dot-product retrieval run;
it is a source-aware Approximate Long-Context Cache variant that reranks DRAM
chunks with lightweight query/source token overlap.  It must still be separated
from oracle retrieval because it does not use the target answer or needle range.

## Evidence Required

- 128K no OOM under the 22 GiB cap.
- Active HBM KV length stays bounded.
- DRAM compressed KV length and actual DRAM bytes are reported.
- `_prefill_update()` returns physically truncated K/V.
- Short-KV attention uses real logical positions and a matching mask.
- NIAH uses real tokenizer and real Qwen generate.
- WikiText-2 PPL uses real loss, not MSE proxy.
- Latency report includes prefill time, decode ms/token, retrieval scoring,
  dequant/transfer, and end-to-end generate time.
