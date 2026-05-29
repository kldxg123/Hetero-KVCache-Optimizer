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

Remote status:

- Remote project: `/home/app-ahr/Hetero-KVCache-Optimizer`.
- Branch observed before remote write block:
  `codex/workflow2-128k-survival-20260528`.
- Remote `paper/` directory was empty before Workflow3 material generation.
- SSH/SFTP write attempts were blocked by permission approval timeouts.

## Next Automatic Step After Permission Is Available

1. Upload local `paper/*.md` files to remote `paper/`.
2. Run remote `git add -N paper/*.md && git diff --check -- paper`.
3. Commit only `paper/*.md` and relevant Workflow3 state docs.
4. Push the Workflow3 paper-start commit.
5. Continue with paper figure script generation and claim audit.
