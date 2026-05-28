# ARIS-Style Workflow 2 Adaptation For HeteroKV

This file records the workflow rules adopted from the ARIS / Auto-Claude
research loop for Hetero-KVCache-Optimizer. It is a process guardrail, not an
experimental result.

## Workflow 2 Loop

Each round must follow this order:

1. Review current evidence and identify the highest-risk missing claim.
2. Propose one or more falsifiable ideas.
3. Rank ideas by expected scientific value, feasibility, and risk.
4. Implement only the smallest code or configuration change needed.
5. Run sanity tests before GPU-heavy runs.
6. Run the smallest real experiment that can falsify the idea.
7. Record both success and failure artifacts.
8. Update the tracker and decide whether to loop, stop, or ask for workflow3.

## Truth Rules

- A failed idea must stay in the report.
- Oracle or diagnostic runs must never be merged with main results.
- A run that only passes under relaxed memory must not support the 24G survival
  claim.
- A100 latency under a 22 GiB cap is not real RTX 4090 latency.
- FullKV baselines must state their attention backend.
- Eager FullKV OOM and SDPA FullKV OOM are separate evidence.
- If a baseline cannot complete, do not claim a latency ratio against it.

## Review State

Every round should update `review-stage/REVIEW_STATE.json` with:

- current_round
- accepted_claims
- blocked_claims
- active_hypothesis
- next_experiment
- latest_failure
- latest_success
- workflow3_ready

## HeteroKV-Specific Stop Criteria

Workflow2 may stop and ask the user about workflow3 only when:

- 128K HeteroKV required-depth NIAH is robust across seeds/trials.
- 128K HeteroKV survives under 22 GiB cap.
- FullKV under the same cap fails or is clearly worse in memory.
- PPL has at least one real full-vs-Hetero comparison and known failure modes
  are reported.
- Latency is measured with a fair baseline or explicitly marked unresolved.
- Mechanism logs prove physical truncation, DRAM storage, retrieval, and
  source-aware fusion behavior.

If any item remains unresolved, continue workflow2 unless only high-risk
experiments above the 30 GiB safety fuse remain.
