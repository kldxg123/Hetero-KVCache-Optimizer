# Appendix: Failed, Invalid, And Diagnostic Ideas

This appendix prevents repeated dead ends and preserves negative evidence.

## Invalid Runs

| Run | Failure Mode | Treatment |
| --- | --- | --- |
| Parallel seed4242 + seed7777 source-prefilter run before output-path fix | Shared child output path clobbered a result | Excluded; sequential reruns are valid |
| First direct seed7777 wrapper run | Missing `CUDA_VISIBLE_DEVICES`, exited before GPU use | Wrapper failure, not model failure |
| First stage2 generate smoke after wrapper changes | `run_stage2_smoke.py` lacked `--attn-implementation` | Fixed script, reran smoke |

## Rejected Method Ideas

| Idea | Evidence | Decision |
| --- | --- | --- |
| Increase sink tokens to 1024 for optional 0/99 | seed6004 produced 0/4 and broke 99% | Reject |
| Treat current 0% NIAH as HeteroKV failure | FullKV also failed both 0% rows | Mark non-discriminative |
| Report TTL-only SourceCopy path as final | Correct but too slow before source prefilter | Superseded |
| Report source-prefilter path as pure dot-product | Mechanism uses source-aware filtering | Disallowed |
| Claim 4090 latency from A100-under-cap | Hardware differs | Disallowed |

## Diagnostic-Only Evidence

Oracle and diagnostic runs are useful for locating bottlenecks, but they must
not be mixed into real method results.

Rules:

- Oracle retrieval is an upper bound or diagnostic, not a deployable result.
- Source-aware retrieval is real only when it uses source/query overlap without
  target labels, needle positions, or answer leakage.
- PPL claims must state whether source-aware features were enabled or disabled.

## Lessons

- Output paths must be unique before parallel experiments.
- Optional depths are only useful if FullKV passes them.
- Latency improvements should be reported as per-token decode time, not only
  lower task elapsed time from shorter generation.
- Source-aware exact-copy results and general-language PPL results must stay in
  separate tables.
