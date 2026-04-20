#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p results

echo "=========================================="
echo "  Hetero-KVCache-Optimizer Real SOTA Benchmarks"
echo "  Comparing against vLLM, SGLang, StreamingLLM, Native HF"
echo "=========================================="

# Auto-prepare sample.mp4
if [ ! -f "sample.mp4" ]; then
    echo "[Setup] Preparing sample.mp4 ..."
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"
    CANDIDATES=(
        "$REPO_ROOT/dummy_1min.mp4"
        "$REPO_ROOT/test_video.mp4"
        "$REPO_ROOT/dummy_2min.mp4"
        "$REPO_ROOT/dummy_4min.mp4"
    )
    for c in "${CANDIDATES[@]}"; do
        if [ -f "$c" ]; then
            ln -s "$c" sample.mp4
            echo "[Setup] Linked sample.mp4 -> $c"
            break
        fi
    done
fi

if [ ! -f "sample.mp4" ]; then
    echo "[Error] No available test video found. Please manually place sample.mp4"
    exit 1
fi

# Set model path (defaults to local model)
export MODEL_PATH="$(dirname "$SCRIPT_DIR")/models/Qwen2-VL-7B"
export PYTHONUNBUFFERED=1

# Define run function, continue on error
run_benchmark() {
    local script=$1
    local name=$2
    echo ""
    echo ">>> [$name] Starting ..."
    if python "$script"; then
        echo "<<< [$name] Completed successfully"
    else
        echo "<<< [$name] Failed (error captured, continuing with remaining tests)"
    fi
}

# Launch all baseline tests
run_benchmark "scripts/run_native_hf.py"    "Native HF (4-bit NF4)"
run_benchmark "scripts/run_streamingllm.py" "StreamingLLM (64 Sink + 4096 Local)"
run_benchmark "scripts/run_vllm.py"         "vLLM PagedAttention (16GB limit)"
run_benchmark "scripts/run_sglang.py"       "SGLang RadixAttention (16GB limit)"

# Aggregate final JSON report
echo ""
echo ">>> Aggregating final JSON report ..."
python3 - << 'PYEOF'
import os
import json
import glob

results_dir = "results"
report = {
    "title": "Hetero-KVCache-Optimizer Real SOTA Benchmark Report",
    "benchmarks": []
}

for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
    if path.endswith("final_report.json"):
        continue
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        report["benchmarks"].append(data)
    except Exception as e:
        report["benchmarks"].append({"file": os.path.basename(path), "load_error": str(e)})

final_path = os.path.join(results_dir, "final_report.json")
with open(final_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"Final report generated: {final_path}")
PYEOF

echo ""
echo "=========================================="
echo "  All tests completed"
echo "  Results directory: results/"
echo "  Aggregated report: results/final_report.json"
echo "=========================================="
