# HeteroKV Paper Draft

Working title:

**HeteroKV: Source-Aware Approximate Long-Context KV Cache for 128K Generation
under Fixed HBM Budgets**

## Abstract Draft

Long-context generation with full KV caching scales memory linearly with
sequence length, making 128K-token contexts difficult to support on commodity
24GB GPUs for 7B-scale instruction models. We study HeteroKV, an approximate
long-context cache that keeps only a bounded active KV set in GPU HBM and moves
evicted KV into compressed DRAM-side storage. The cache combines sink tokens,
tail tokens, heavy-hitter retention, compressed DRAM KV, and source-aware
token-level retrieval. HeteroKV is not designed to reproduce full attention
losslessly; instead, it targets stable memory use, functional generation, and
controlled semantic loss under a fixed HBM budget.

On Qwen2.5-7B-Instruct at 128K context length, our promoted source-aware NIAH
configuration achieves 24/24 accuracy across required depths 25%, 50%, 75%,
and 90% under an A100 run constrained to a 4090-like memory envelope. The same
configuration runs with a 22 GiB PyTorch cap and a 30 GiB own-process safety
fuse, reaches 98.12 ms/step mean decode latency, and is 1.88x the wide-memory
A100 FullKV reference for this setting. On a separate SourceCopy-disabled
WikiText-2 PPL setup, HeteroKV shows a +1.20% PPL delta relative to FullKV.
These results support HeteroKV as an approximate long-context survival
mechanism, while leaving true RTX 4090 latency and broader 128K PPL validation
as future hardware-specific work.

## 1. Introduction

Full KV caching is simple and accurate, but its memory footprint grows
linearly with context length. At 128K tokens, a 7B model can exceed the memory
budget of a 24GB GPU even before accounting for allocator fragmentation,
runtime context, and generation overhead. This creates a practical gap between
model context-window specifications and deployable long-context generation.

HeteroKV addresses this gap as an approximate long-context cache. Rather than
keeping the entire historical KV tensor in HBM, it keeps a bounded active set:
global sink tokens, local tail tokens, heavy-hitter tokens, and selected
retrieved tokens. Evicted tokens are stored in compressed DRAM-side form and
can be retrieved when the query indicates that historical information is needed.

The central thesis is modest but useful: for long-context tasks where semantic
recall matters more than token-level full-attention equivalence, a bounded-HBM
cache can preserve enough useful information to generate correctly while
remaining within a 24GB-class memory envelope.

## 2. Scope And Non-Goals

HeteroKV is not a lossless full-KV replacement. It does not target exact
token-level logits equivalence with native full attention. It also does not
claim native RTX 4090 latency from A100 measurements. The validated hardware
claim is an A100 run under a 4090-like memory envelope.

The current strongest NIAH path uses source-aware retrieval support. It should
be described as a source-aware exact-copy/NIAH path, not as pure dot-product
retrieval alone. General-language quality is evaluated separately through a
SourceCopy-disabled WikiText-2 PPL run.

## 3. Method

### 3.1 Bounded Active KV

During chunked prefill, the cache physically truncates the returned active KV
instead of returning full historical K/V. The active set is composed of:

- sink tokens;
- tail tokens;
- heavy-hitter tokens selected from accumulated attention evidence;
- required current active tokens.

Evicted KV is stored outside HBM in compressed DRAM-side records.

### 3.2 DRAM-Side Compressed KV

Evicted K/V entries are recorded in compressed form with explicit accounting
for entries and bytes. This supports a claim that active GPU KV is bounded
while CPU/DRAM storage can grow with context length.

### 3.3 Query-Aware Retrieval

The cache uses token-level retrieval evidence rather than chunk-mean features.
The promoted NIAH path adds a source-aware prefilter that narrows DRAM
candidates before token-level scoring. This reduces retrieval overhead while
remaining non-oracle: it does not use target labels, answer spans, or needle
ranges.

### 3.4 Late-Layer Retrieval

The best validated NIAH configuration restricts source-aware retrieval to
layers 22-27. This cuts retrieval events to 150 per row while preserving
accuracy across the required NIAH depths.

## 4. Experimental Setup

Model: Qwen2.5-7B-Instruct.

Primary context: 128K.

Primary hardware claim: A100 under a 4090-like memory envelope, configured with
a 22 GiB PyTorch cap and a 30 GiB own-process fuse.

Main NIAH depths: 25%, 50%, 75%, 90%.

Optional edge depths: 0%, 99%.

PPL dataset: WikiText-2, real loss/PPL calculation, SourceCopy disabled.

Baseline latency reference: FullKV SDPA manual decode on wide-memory A100.

## 5. Main Results

The promoted source-aware 128K NIAH path achieved 24/24 accuracy across three
seeds and the required depths. Mean decode latency was 98.12 ms/step, compared
with 52.25 ms/step for the wide-memory FullKV A100 reference, giving a 1.88x
ratio. Own-process monitor peak was 22348 MB and no 30 GiB fuse trigger
occurred.

On WikiText-2, the SourceCopy-disabled PPL run measured FullKV PPL 2.9706 and
HeteroKV PPL 3.0063, a +1.20% relative delta.

Generate smoke tests passed for 2K, 4K, and 8K contexts through the ordinary HF
`generate()` path.

## 6. Ablations And Diagnostics

Layer-range ablations showed that restricting retrieval to late layers is
important for meeting the latency target. The 22-27 layer range preserved
accuracy on the seed6004 25%/50% smoke and then passed full-depth multi-seed
validation.

The optional 99% depth passed 6/6. The optional 0% depth is currently
non-discriminative because the FullKV wide-memory baseline also failed it.

Rejected diagnostics and invalid runs are retained in `RESULT_TABLES.md` and
must not be removed from the record.

## 7. Limitations

The current results have several explicit limits:

- A100-under-cap latency is not native RTX 4090 latency.
- The strongest NIAH result is source-aware and should not be called pure
  dot-product retrieval.
- PPL evidence is SourceCopy-disabled and measured on a 14K setup, not 128K.
- The optional 0% NIAH template needs redesign before it can be used as a
  discriminative benchmark.
- The project is an approximate cache, not a lossless full-attention system.

## 8. Next Experiments Before Submission

The next paper-grade improvements are:

1. Rerun survival and latency on a real RTX 4090 24GB GPU.
2. Redesign optional 0% NIAH so FullKV passes it, then retest HeteroKV.
3. Add a larger PPL suite or longer-context PPL if runtime allows.
4. Add a clean pure-dot-product retrieval table, separate from source-aware
   retrieval.
5. Plot active HBM tokens, DRAM tokens/bytes, and process memory over prefill
   chunks for the promoted 128K run.
