# Final Proposal: Hetero-KVCache-Optimizer

## Positioning

Hetero-KVCache-Optimizer is an **Approximate Long-Context Cache** system.

It is not designed to losslessly reproduce native 128K full KV cache, and it is not judged by token-level logits equivalence with full causal attention. The project is judged by whether it allows a 7B model to survive and generate under a fixed HBM budget while keeping semantic loss and latency explainable.

## Final Goal

Demonstrate that **Qwen2.5-7B-Instruct** can run with **128K context** under a **4090-like 24G GPU memory envelope** by using:

- Sink tokens as global anchors.
- Tail tokens as local recent context.
- Heavy-Hitter tokens selected by accumulated attention.
- 4-bit DRAM KV storage for evicted tokens.
- Query-aware token-level dot-product retrieval for relevant evicted chunks.

## Hardware Claim

The remote server uses A100 GPUs. The main acceptance run must therefore simulate the 4090 memory envelope:

- HeteroKV acceptance run: 22 GiB PyTorch memory cap.
- Optional supplemental run: 24 GiB cap.
- Full-KV baseline may be uncapped only when the target GPU is idle.

Allowed conclusion:

> HeteroKV survives 128K context for Qwen2.5-7B-Instruct under an A100 run capped to a 4090-like 24G memory envelope.

Not allowed without real 4090 retest:

> HeteroKV latency is proven on RTX 4090.

## Core Design

### 1. Physical Short-KV Cache

During prefill and decode, active HBM KV is bounded to:

```text
Sink + Tail + Heavy-Hitter + currently retrieved short chunks
```

Evicted tokens move to DRAM-side quantized storage. The attention-facing K/V tensor must not be full 128K and must not be padded back to 128K.

### 2. Logical Position Preservation

Short KV is physically compact but logically sparse. Each retained key token keeps its original absolute position.

Attention rules:

- Query RoPE uses true absolute positions.
- Key RoPE uses true absolute positions before cache update.
- Causal masking uses query absolute positions and retained key absolute positions.
- Future tokens are masked by logical position, not by physical short-KV index.

### 3. Query-aware Dot-Product Retrieval

The main retrieval path scores candidate DRAM chunks by:

```text
Query x Key
```

Requirements:

- Use RoPE-applied `query_states`.
- Dequantize only candidate chunks or small batches.
- Compute token-level scores with `torch.matmul`.
- Use max token score as the default chunk score.
- Retrieve Top-K chunks.
- Release temporary dequantized tensors and scores after each batch.

### 4. Heavy-Hitter Eviction

The eviction policy must prove:

- Sink is always retained.
- Tail is always retained.
- High accumulated attention tokens compete for Heavy-Hitter space.
- Low-score Heavy-Hitter tokens are evicted to DRAM.
- The algorithm is not FIFO and not simple position trimming.

### 5. No Early Triton

The default acceptance path uses PyTorch operations and `enable_triton=False`.

Triton/CUDA fused dequant attention is only considered after:

- 128K survival passes.
- NIAH passes.
- WikiText-2 PPL passes.
- generate compatibility passes.
- latency is still above target.

## Success Criteria

Minimum success:

- 128K context does not OOM under 22 GiB cap.
- Active HBM KV length stays bounded.
- DRAM compressed KV grows with context.
- `generate()` works.
- NIAH multi-depth accuracy >= 95%.
- WikiText-2 PPL degradation <= 5% versus a valid full-KV baseline length.
- Latency is measured and broken down.

Excellent success:

- NIAH multi-depth accuracy reaches 100%.
- 128K memory curve is stable without abnormal retrieval spikes.
- Dot-product retrieval clearly outperforms Mean-K retrieval.
- Latency is <= 2x the valid baseline.

## Deliverable Shape

The final project demonstration should include:

- Short-KV implementation evidence.
- 4090-24G capped survival log.
- Active HBM KV curve.
- DRAM compressed KV curve and actual bytes.
- NIAH results.
- WikiText-2 PPL results.
- Ablation table.
- Heavy-Hitter logs.
- Dot-product retrieval logs.
- generate compatibility results.
- latency breakdown.
- clear note on any tests skipped because of shared-server safety.

