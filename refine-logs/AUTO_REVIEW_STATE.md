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

Update after Round 29:

- SourceCopy robustness run completed through the workflow driver:
  - Seed `6004`, 128K, depths 25/50/75/90, 2 trials each.
  - Result `8/8`, monitor peak `21.8262 GiB`, no monitor kill.
- Driver-based SourceCopy-assisted required-depth matrix is now complete:
  - Seed4242 required-depth trials2: `8/8`.
  - Seed7777 required-depth trials2: `8/8`.
  - Seed6004 required-depth trials2: `8/8`.
  - Aggregate: `24/24`.
- Claim boundary:
  - This is strong exact-copy NIAH evidence for the SourceCopy-assisted path.
  - It is still not the pure dot-product retrieval result.
  - Pure source-aware retrieval without SourceCopy remains documented as weaker on the same hard ablation.
- Next automatic priority:
  - Commit/sync/push this evidence.
  - Run latency breakdown and fair baseline refresh under the same 22 GiB memory envelope, if GPU safety permits.

Update after Round 30:

- Latency candidate update:
  - SourceCopy TTL12 completed required-depth trials2 on seeds `6004`, `4242`, and `7777`.
  - Aggregate result: `24/24`.
  - Mean decode improved from TTL6 `689.4 ms/step` to TTL12 `450.6 ms/step`.
  - Peak process memory stayed `21.8262 GiB`; torch reserved stayed `21.3262 GiB`.
- Fair baseline refresh:
  - FullKV 128K SDPA manual decode on idle GPU1, 75 GiB cap: `1/1`.
  - Decode `52.25 ms/step`, torch reserved `62.9629 GiB`.
- Claim boundary:
  - TTL12 is the current latency candidate.
  - The `<=2x` latency target is still not met; mean ratio is `8.62x` and median ratio is `7.53x` versus wide-memory FullKV SDPA.
  - Do not enter Workflow3 yet.
- Failed/weak ideas recorded:
  - SourceCopy + Triton scoring: correct but weak speed gain on seed6004 25/50 trials2.
  - Decode no-attention-mask: rejected after clean retry with cuBLAS/runtime errors.
- Next automatic priority:
  - Commit/sync/push Round 30 evidence.
  - Refresh PPL with the TTL12 candidate while keeping SourceCopy out of general-language PPL.

Update after Round 31:

- PPL runner audit fixed a parameter-recording gap:
  - Added `--method-d-reuse-ttl-tokens`.
  - Added `--method-d-reuse-source-threshold`.
  - Passed both into `build_fused_cache`.
  - Added `cache_config` to the PPL JSON.
- Safety:
  - First strict GPU1 PPL attempt stopped with `rc=8` after another user's process appeared.
  - Successful retry used GPU3 with `--allow-other-processes-if-memory-fits`, while preserving the 22 GiB PyTorch cap, 30 GiB own-process fuse, and 4 GiB reserve.
- TTL12 PPL refresh:
  - Artifact: `experiments/ppl_14k_prefix12288_tail4096_gate5_top1_nofusion_sdpa_ttl12_sourcecopy_disabled_allowcoexist_gpu3_20260529_auto.json`.
  - Full PPL: `2.9706`.
  - HeteroKV PPL: `3.0063`.
  - Relative delta: `+1.20%`.
  - Hetero max reserved: `19.2754 GiB`.
  - Own-process monitor peak: `20.248 GiB`.
- Claim boundary:
  - This is real WikiText-2 decode-suffix PPL, not a 128K PPL claim.
  - SourceCopy was disabled for PPL; exact-copy NIAH and general-language PPL remain separate.
  - Workflow3 is still blocked by latency ratio, not PPL.
- Next automatic priority:
  - Commit/sync/push Round 31 code and evidence.
  - Continue Workflow2 latency-oriented optimization or run a diagnostic generic-TTL PPL ablation only if it is needed for paper clarity.

Update after Round 32:

- TTL24 small latency ablation completed:
  - Artifact: `experiments/niah_128k_depth25_50_trials2_sourcecopy_ttl24_seed6004_driver_gpu3_20260529_auto.json`.
  - Result: `4/4` on seed6004, 128K, depths 25/50, 2 trials each.
  - Mean decode: `369.8 ms/step`.
  - Mean elapsed: `57.8s`.
  - Monitor peak: `21.8262 GiB`.
- Short-answer display ablation completed:
  - Artifact: `experiments/niah_128k_depth25_50_trials2_sourcecopy_ttl24_maxnew8_seed6004_driver_gpu3_20260529_auto.json`.
  - Result: `4/4`.
  - Mean elapsed: `53.1s`.
  - Mean decode: `483.8 ms/step`.
- Claim boundary:
  - TTL24 is a weak positive latency idea, not yet a promoted main configuration.
  - `max_new_tokens=8` is a fair demo setting for a 6-digit code answer, but it is not an algorithmic per-token speedup.
- GitHub push state:
  - Remote local commit `27e26bb` exists for Round 31.
  - Push is pending because `github.com:443` was unreachable on the retry.
- Next automatic priority:
  - Commit Round 32 docs locally on the remote.
  - Retry GitHub push when connectivity returns.
  - Choose between full TTL24 validation and a deeper structural latency idea; do not claim Workflow3 readiness yet.

Update after Round 33:

- Structural latency optimization implemented:
  - Source-overlap prefilter now narrows Method-D DRAM candidates before token-level dot-product scoring when SourceCopy/source-aware mode is enabled.
  - Fallback remains full DRAM candidate scoring if the source prefilter finds no chunk.
  - Mechanism logs expose `source_prefilter{selected}of{total}`.
- Code review outcome:
  - Initial implementation had an attribute-name bug; stage-1 tests caught it.
  - Fixed threshold field to `_method_d_reuse_source_threshold`.
  - Remote `tests/test_heterokv_stage1.py`: `16 passed`.
- Clean 128K source-prefilter evidence:
  - Seed6004 required-depth trials2: `8/8`, mean decode `166.5 ms/step`.
  - Seed4242 required-depth trials2, sequential rerun: `8/8`, mean decode `168.8 ms/step`.
  - Seed7777 required-depth trials2, sequential rerun: `8/8`, mean decode `169.1 ms/step`.
  - Aggregate: `24/24`, depth-wise `6/6` at 25/50/75/90.
  - Aggregate mean decode: `168.1 ms/step`.
  - Aggregate mean prefill: `48.47s`.
  - Source prefilter evidence: `(1, 60)` chunks in mechanism tail logs.
  - Seed7777 monitor peak: `22348 MB`, no 30 GiB fuse trigger.
- Invalid/failed runs:
  - Parallel seed4242/seed7777 prefilter run is invalid because fixed child output `experiments/niah_eval.json` caused clobbering.
  - First seed7777 direct wrapper run failed before GPU allocation because `CUDA_VISIBLE_DEVICES=3` was missing.
- Workflow harness fix:
  - `scripts/run_experiment.py` now derives unique NIAH child output from non-default tracker stems.
  - New explicit override: `--niah-output`.
  - Remote compile and stage-1 tests passed.
- Claim boundary:
  - Source-prefiltered TTL24 is the current best NIAH/source-cue path.
  - It is not the pure dot-product-only claim.
  - Workflow3 remains blocked by latency: `168.1 / 52.25 = 3.22x`, still above the original `<=2x` target.
- Next automatic priority:
  - Commit Round 33 code/docs locally on the remote.
  - Push if GitHub authentication/network works; if not, record auth failure and keep remote local commits.
  - Continue Workflow2 with either a latency-ratio optimization or prepare a strict paper-claim split where source-aware NIAH and SourceCopy-disabled PPL are presented separately.
