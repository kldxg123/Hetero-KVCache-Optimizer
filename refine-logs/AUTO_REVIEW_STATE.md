# Auto Review State

Mode: fully automatic Workflow2 review loop.

The loop should continue as:

1. Review current evidence and code paths.
2. Record failed ideas before trying new variants.
3. Make the smallest justified implementation or testing-method change.
4. Run syntax/unit/sanity checks first.
5. Run GPU experiments only when shared-server memory safety allows it.
6. Update `IDEA_REPORT.md` and `refine-logs/EXPERIMENT_RESULTS.md` after every material result.

Stop only when:

- Own-process GPU memory approaches or exceeds the 30 GiB fuse.
- The only remaining experiments are expected to exceed the safe memory envelope.
- SSH/GPU access is blocked and cannot be restored without user credential clarification.
- No credible optimization direction remains.
- Evidence is strong enough to ask the user whether to enter Workflow3.

Truthfulness rules:

- Keep oracle/diagnostic results separate from real retrieval results.
- Keep SourceCopy exact-string reranker results separate from pure Query-Key dot-product retrieval.
- Do not use invalid OOMs caused by test configuration as algorithm failures.
- Do not claim real 4090 latency from A100 memory-envelope experiments.
- Report optional 0% NIAH as a boundary weakness unless it is explicitly fixed and retested.

Current local prepared action:

- `remote_edit/scripts/run_ppl_eval.py` now defaults to `--attn-implementation sdpa` to avoid repeating the known eager-attention PPL OOM configuration.

Current blocker:

- Remote SSH access is not available from the local environment without clarifying the unlabeled fields in `服务器信息.txt`.
- Do not brute-force username/password combinations.
- After access is restored, first sync the local docs and prepared PPL default patch, then rerun py_compile/unit tests remotely before GPU experiments.
