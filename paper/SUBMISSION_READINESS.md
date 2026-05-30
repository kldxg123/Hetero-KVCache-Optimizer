# Submission Readiness Gate

Current gate result: not passed.

The project is ready for serious paper drafting, but not yet ready for a
top-conference submission claim.

## Passed Gates

| Gate | Status | Evidence |
| --- | --- | --- |
| 128K required-depth NIAH | Passed | 24/24 across depths 25/50/75/90 |
| A100 memory-envelope survival | Passed | 22 GiB PyTorch cap, 30 GiB fuse, 22348 MB peak |
| Source-aware latency target | Passed | 98.12 ms/step, 1.88x FullKV A100 reference |
| PPL degradation on tested setup | Passed | +1.20% on SourceCopy-disabled WikiText-2 |
| Generate API smoke | Passed | 2K/4K/8K HF `generate()` smoke |
| Negative-result preservation | Passed | Failed ideas and invalid runs recorded |

## Not-Yet-Passed Gates

| Gate | Status | Required Action |
| --- | --- | --- |
| True RTX 4090 latency | Not passed | Run on real RTX 4090 or weaken claim |
| Broad semantic quality | Partially passed | Two SourceCopy-disabled WikiText-2 suffix setups pass; add offsets or second corpus for a broad claim |
| Pure dot-product retrieval claim | Not passed | Current clean 128K pure-dot control is 0/4; keep claim source-aware or add shorter-context scaling table |
| 0% NIAH discriminativeness | Not passed | Redesign template and require FullKV pass |
| Paper-ready figures | Partially passed | Main result, latency, PPL, ablation, and memory curves rendered; still need final paper styling/captions |

## Automatic Decision

Continue Workflow3. Do not request user direction yet unless:

- a GPU experiment is required and estimated memory may exceed 30 GiB;
- a real RTX 4090 run becomes the only remaining path;
- results contradict the current claim boundary;
- all gates pass and paper writing can move to final submission packaging.
