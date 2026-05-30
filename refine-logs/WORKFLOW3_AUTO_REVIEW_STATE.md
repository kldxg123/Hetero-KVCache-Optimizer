# Workflow3 Automatic Review State

Mode: fully automatic review.

Date: 2026-05-30.

## Active Operating Rules

- Continue autonomously through review, documentation, and lightweight checks.
- Do not start heavy GPU experiments during paper drafting unless the next
  missing claim requires them.
- Keep all claims tied to real artifacts.
- Keep source-aware NIAH, pure dot-product retrieval, PPL, and diagnostic/oracle
  evidence in separate categories.
- Preserve failed ideas and invalid runs.
- Stop only for high-risk operations, missing permissions, or experiments that
  may exceed the configured safety envelope.

## Current Workflow3 Status

Local paper workspace created:

- `paper/WORKFLOW3_README.md`
- `paper/CLAIM_BOUNDARY.md`
- `paper/RESULT_TABLES.md`
- `paper/PAPER_DRAFT.md`
- `paper/FIGURE_PLAN.md`
- `paper/AUTO_REVIEW_CHECKLIST.md`

Local review completed:

- ASCII check passed.
- No trailing whitespace found in `paper/*.md`.
- High-risk phrases are present only in explicit boundary or disallowed-claim
  sections.
- Long lines are limited to result-table artifact paths.
- Workflow3 summary artifacts were downloaded from the remote experiment
  directory into `outputs/workflow3_artifacts/`.
- Standard-library SVG figure generation was added in
  `paper/scripts/build_workflow3_figures.py`.
- Initial summary figures and `paper/data/workflow3_summary.json` were
  generated from real artifacts.
- Additional source-aware/SourceCopy and earlier pure-dotproduct failed
  artifacts were downloaded and added to the Workflow3 summary.
- Two additional figures were generated:
  `sourcecopy_ablation_accuracy.svg` and
  `pure_dotproduct_failed_accuracy.svg`.
- The result tables now explicitly preserve earlier pure-dotproduct failures,
  avoiding a success-only presentation.

Remote status:

- Remote project: `/home/app-ahr/Hetero-KVCache-Optimizer`.
- Branch observed before remote write block:
  `codex/workflow2-128k-survival-20260528`.
- Workflow3 paper workspace commit was pushed:
  `cc6a41c Start Workflow3 paper workspace`.

## Next Automatic Step After Permission Is Available

1. Upload generated figure scripts/data/SVGs to remote `paper/`.
2. Run remote `git diff --check` on Workflow3 paper files.
3. Commit only Workflow3 paper/state artifacts.
4. Continue with paper figure audit and missing-experiment decisions.

## Latest Local Review Snapshot

- Required-depth source-aware NIAH: `24/24`.
- Source-aware no-SourceCopy hard ablation: `3/4`.
- Source-aware + SourceCopy hard ablation: `4/4`.
- Earlier pure dot-product attempts: `0/4`, `1/4`, `0/4`, `0/4`.
- Current clean pure dot-product top8/qhist64 negative control:
  `0/4`, mean decode `1005.04 ms/step`, peak `21.8242 GiB`, no monitor kill.
- PPL SourceCopy-disabled: `+1.20%`.
- Current submission gate remains not passed; the strongest remaining blockers
  are real RTX 4090 validation, broader PPL, and a broader short-to-long pure
  retrieval scaling table if the paper wants to discuss pure QK retrieval beyond
  a negative 128K control.
