#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Hetero-KVCache-Optimizer Environment Setup"
echo "=========================================="

# 1. Install base dependencies
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt

# 2. Install flash-attn (handled separately to avoid build isolation issues)
echo "[2/4] Installing flash-attn..."
pip install flash-attn --no-build-isolation || echo "[Warning] flash-attn installation failed, some features may fall back to eager attention"

# 3. Install vLLM (auto-selected based on CUDA version)
echo "[3/4] Installing vLLM..."
pip install 'vllm>=0.6.1' || echo "[Warning] vLLM installation failed, run_vllm.py will be unavailable"

# 4. Prepare sample.mp4
echo "[4/4] Preparing test video sample.mp4..."
if [ ! -f "sample.mp4" ]; then
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
            echo "[Setup] Created symlink: sample.mp4 -> $c"
            break
        fi
    done
fi

if [ ! -f "sample.mp4" ]; then
    echo "[Error] No available test video found. Please manually place sample.mp4 in the current directory"
    exit 1
fi

# 5. HuggingFace login hint
echo ""
echo "[Hint] If you need to download models from HuggingFace, run huggingface-cli login"
echo "       This benchmark defaults to local model: ../models/Qwen2-VL-7B"
echo ""
echo "=========================================="
echo "  Environment setup complete"
echo "=========================================="
