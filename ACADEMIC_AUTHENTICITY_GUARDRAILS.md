# Academic Authenticity Guardrails

These rules are permanent constraints for Hetero-KVCache-Optimizer work.

## Truthfulness

- Report only experiments that were actually run.
- Preserve raw configuration, logs, memory monitor output, generated answers,
  and failure cases.
- Do not infer success from a diagnostic proxy.
- Do not hide negative results or failed ideas.

## Claim Boundaries

- Oracle retrieval proves an upper bound or diagnoses a bottleneck; it is not
  the real dot-product retrieval result.
- A diagnostic attention bias run must be labeled as such until it is validated
  as part of the final method.
- A100 under a 22 GiB cap supports a 4090-like memory-envelope claim only; it
  does not prove native RTX 4090 latency.
- Full-KV baseline failures, skipped runs, and shared-server safety skips must
  be reported exactly.

## Acceptance Evidence

Acceptance requires agreement between:

- code path and static checks;
- physical short-KV length;
- DRAM compressed KV entries and actual bytes;
- retrieval selection logs;
- attention/retrieval evidence;
- real NIAH or PPL outputs;
- `torch.cuda` memory stats;
- `nvidia-smi` process memory;
- latency breakdown when latency is claimed.

Any future report must distinguish acceptance results from ablation,
diagnostic, oracle, and safety-only results.

## Source-Aware Metadata

If a result uses source-token metadata, say so plainly.  Do not describe
`method_d_source_token_boost` experiments as pure KV-only dot-product retrieval.
They are real non-oracle experiments only if they use source/query token overlap
without target labels, needle ranges, or answer leakage.
