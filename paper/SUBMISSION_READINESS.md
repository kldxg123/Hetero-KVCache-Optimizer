# Submission Readiness Gate

Current gate result: not passed.

The project is ready for serious paper drafting, but not yet ready for a
top-conference submission claim.

## Passed Gates

| Gate | Status | Evidence |
| --- | --- | --- |
| 128K required-depth NIAH | Passed | 24/24 across depths 25/50/75/90 |
| A100 memory-envelope survival | Passed | 22 GiB PyTorch cap, 30 GiB fuse, 22348 MB peak |
| FullKV 22 GiB-cap negative control | Passed | 128K FullKV fails with CUDA OOM under the same cap without 30 GiB fuse trigger |
| Source-aware latency target | Passed | 98.12 ms/step, 1.88x FullKV A100 reference |
| PPL degradation on tested setups | Passed | +1.20%, +1.66%, +0.45%, +3.14% on WikiText-2 and +1.09% on IMDb, all SourceCopy-disabled |
| Generate API smoke | Passed | 2K/4K/8K HF `generate()` smoke |
| Negative-result preservation | Passed | Failed ideas, invalid runs, and 16K/32K/64K pure-dot scaling diagnostic recorded |

## Not-Yet-Passed Gates

| Gate | Status | Required Action |
| --- | --- | --- |
| True RTX 4090 latency | Not passed | Run on real RTX 4090 or weaken claim |
| Broad semantic quality | Partially passed | SourceCopy-disabled suffix PPL passes on WikiText-2 through 32K and IMDb at 16K; still not true 128K PPL |
| Pure dot-product retrieval claim | Not passed by evidence | Clean 128K pure-dot control is 0/4 and 16K/32K/64K scaling is 11/24; keep the promoted claim source-aware |
| 0% NIAH discriminativeness | Not passed | Redesign template and require FullKV pass |
| Paper-ready figures | Partially passed | Main result, latency, PPL, ablation, and memory curves rendered; still need final paper styling/captions |

## Automatic Decision

Continue Workflow3. Do not request user direction yet unless:

- a GPU experiment is required and estimated memory may exceed 30 GiB;
- a real RTX 4090 run becomes the only remaining path;
- results contradict the current claim boundary;
- all gates pass and paper writing can move to final submission packaging.
