# Skill: Hetero-KVCache-Research

## Context
You are optimizing KV Cache memory for Qwen2-VL. The goal is to sustain 45k+ tokens on 16GB VRAM using Transient Cache and In-place Shift.

## Commands
### `/stress-test`
- **Action**: Run `python tests/profiling_script.py` with incremental `--input_len`.
- **Target**: Find the OOM breaking point.
- **Success Criteria**: `input_len >= 45000` without "💥 Native OOM".

### `/auto-patch`
- **Action**: If `IndexError` or `ShapeMismatch` occurs in `src/memory/cache.py`, analyze the tensor slicing logic and apply a fix.
- **Rule**: Never change the core logic of `In-place Shift` unless explicitly asked.

### `/ablation-sink`
- **Action**: Loop through `--sink_size` values (0, 32, 64, 128) and record TTFT/TPOT.
- **Report**: Generate a Markdown table comparing performance vs. memory.

## Safety Gates
- If VRAM exceeds 15.8GB, immediately trigger `gc.collect()` and check for memory leaks in `HeteroTransientCache`.
- Ensure `PYTORCH_ALLOC_CONF=expandable_segments:True` is always set.