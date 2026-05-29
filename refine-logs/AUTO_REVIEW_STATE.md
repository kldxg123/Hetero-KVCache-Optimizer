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

Update 2026-05-29:

- Chinese labels in `服务器信息.txt` are parsed with the full-width colon `\uff1a`.
- SSH authentication succeeded once and the local prepared files were synchronized to the remote repository.
- Remote `py_compile` and unit tests passed: `16 passed`.
- Remote commit was created: `4049ef6 Record auto review state and default PPL SDPA`.
- Push confirmation is still pending because the SSH port became unreachable immediately afterward.
- Current network check: TCP connection to `182.92.245.8:2222` failed.
- On recovery, first run `git log --oneline -3`, compare `HEAD` with `origin/codex/workflow2-128k-survival-20260528`, then push if needed. Do not rerun GPU tests before confirming repository state.

Update 2026-05-29 later:

- Remote branch was advanced and pushed through `4049ef6`.
- Driver fix commit was created remotely: `396379c Pass NIAH SourceCopy args through workflow driver`.
- Push for `396379c` is still pending due server-side GitHub HTTPS connectivity failure.
- Driver-based 128K ablation completed:
  - No SourceCopy: `3/4`, monitor peak `21.8242 GiB`.
  - SourceCopy boost20: `4/4`, monitor peak `21.8262 GiB`.
- Next automatic step after documenting this round:
  - Sync these report updates to remote.
  - Commit them after py_compile/tests.
  - Retry GitHub push when `github.com:443` is reachable.
  - Continue with broader SourceCopy multi-seed/multi-trial or latency breakdown, depending on GPU safety.

Update after Round 27:

- SourceCopy robustness run completed through the workflow driver:
  - Seed `4242`, 128K, depths 25/50/75/90, 2 trials each.
  - Result `8/8`, monitor peak `21.8262 GiB`, no monitor kill.
- Driver-based SourceCopy evidence is now:
  - Seed6004 25/50 trials2: `4/4`.
  - Seed4242 required-depth trials2: `8/8`.
- Next candidates:
  - Run seed7777 required-depth trials2 if GPU3 remains safe.
  - Or run latency breakdown if enough exactness evidence is considered sufficient.

Update after Round 28:

- SourceCopy robustness run completed through the workflow driver:
  - Seed `7777`, 128K, depths 25/50/75/90, 2 trials each.
  - Result `8/8`, monitor peak `21.8262 GiB`, no monitor kill.
- Driver-based SourceCopy evidence is now:
  - Seed6004 25/50 trials2: `4/4`.
  - Seed4242 required-depth trials2: `8/8`.
  - Seed7777 required-depth trials2: `8/8`.
- Combined monitored SourceCopy exactness rows: `20/20`, with the boundary that seed6004 is not yet a full required-depth 2-trial driver rerun.
- Next automatic priority:
  - Commit/sync this evidence.
  - Retry GitHub push when available.
  - Check GPU safety, then run latency breakdown/fair baseline refresh or a seed6004 full required-depth driver rerun.
