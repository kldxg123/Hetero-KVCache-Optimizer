# Workflow3 Paper Workspace

This directory is the Workflow3 handoff area for turning the verified
Workflow2 evidence into a paper-grade claim set, draft, tables, and figure
plan.

## Current Paper Readiness

Status: ready to start paper drafting, with explicit claim boundaries.

The strongest current result is not a pure full-KV replacement claim. It is a
source-aware approximate long-context cache result:

- Model: Qwen2.5-7B-Instruct.
- Context: 128K NIAH.
- Hardware envelope: A100 server, constrained to a 4090-like memory budget.
- Memory policy: 22 GiB PyTorch cap plus 30 GiB own-process fuse.
- Main promoted path: source-prefiltered retrieval, TTL24, active layers 22-27.
- Required NIAH depths: 25%, 50%, 75%, 90%.
- Multi-seed result: 24/24 correct.
- Mean decode: 98.12 ms/step.
- FullKV wide-memory A100 reference: 52.25 ms/step.
- Latency ratio: 1.88x.
- Own-process monitor peak: 22348 MB.
- Safety: no 30 GiB fuse trigger.

## Non-Negotiable Positioning

Hetero-KVCache-Optimizer is an Approximate Long-Context Cache. It is not a
lossless reconstruction of native 128K full KV cache, and token-level logits
equivalence to native full attention is not the target.

The project target is fixed-HBM survival with controlled semantic loss:

- keep active HBM KV bounded;
- move evicted KV into compressed DRAM-side storage;
- retrieve useful historical KV through query-aware, token-level mechanisms;
- keep generation functional;
- make latency explainable and bounded for the validated path.

## Workflow3 Tasks

1. Convert verified Workflow2 results into paper tables.
2. Preserve failed ideas and diagnostic-only mechanisms.
3. Draft the paper with conservative language.
4. Identify missing paper-grade experiments before submission.
5. Keep A100-under-cap evidence separate from true RTX 4090 latency claims.

## Files

- `CLAIM_BOUNDARY.md`: allowed claims, disallowed claims, and caveats.
- `RESULT_TABLES.md`: paper-ready result tables with artifact references.
- `PAPER_DRAFT.md`: initial paper skeleton and draft text.
- `FIGURE_PLAN.md`: figures needed for a paper or thesis defense.
- `AUTO_REVIEW_CHECKLIST.md`: automatic review checklist before each revision.
