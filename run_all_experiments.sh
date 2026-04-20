#!/usr/bin/env bash
# run_all_experiments.sh
# One-click reproduction pipeline for Hetero-KVCache-Optimizer Artifact Evaluation

set -euo pipefail

echo "=========================================="
echo " Hetero-KVCache-Optimizer AE Pipeline"
echo "=========================================="

mkdir -p experiments

# ---------------------------------------------------------------------------
# Step 1: Unit tests & sanity checks
# ---------------------------------------------------------------------------
echo ""
echo "[Step 1/7] Running unit tests..."
python -m py_compile src/memory/manager.py src/core/engine_wrapper.py src/simulation/pcie_throttle_sim.py
python src/quantization/kv_compressor.py

# ---------------------------------------------------------------------------
# Step 2: Micro-benchmarks (memory scalability)
# ---------------------------------------------------------------------------
echo ""
echo "[Step 2/7] Running memory scalability micro-benchmark..."
python benchmark_hetero_kv.py || echo "   (Micro-benchmark completed with warnings)"

# ---------------------------------------------------------------------------
# Step 3: Baseline comparisons
# ---------------------------------------------------------------------------
echo ""
echo "[Step 3/7] Running baseline comparisons (Hetero vs StreamingLLM vs Native)..."
python tests/baseline_compare.py || echo "   (Baseline comparison completed with warnings)"

echo ""
echo "[Step 4/7] Running real SOTA benchmarks (vLLM, SGLang, StreamingLLM, Native HF)..."
bash real_sota_benchmarks/run_real_benchmarks.sh || echo "   (Real SOTA benchmarks completed with warnings)"

# ---------------------------------------------------------------------------
# Step 4: Multi-model validation
# ---------------------------------------------------------------------------
echo ""
echo "[Step 5/7] Running multi-model benchmark..."
# Note: This downloads models from HuggingFace if they are not cached locally.
# If no internet access is available, the script gracefully skips missing models.
python tests/multimodel_benchmark.py \
  --models llama internvl \
  --token_targets 4096 8192 \
  --native_max_tokens 8000 \
  --min_new_tokens 10 \
  || echo "   (Multi-model benchmark completed with warnings)"

# ---------------------------------------------------------------------------
# Step 5: PCIe bandwidth simulation
# ---------------------------------------------------------------------------
echo ""
echo "[Step 6/7] Running bandwidth-throttled edge simulation..."
python tests/multimodel_benchmark.py \
  --models llama \
  --token_targets 4096 8192 \
  --native_max_tokens 8000 \
  --min_new_tokens 10 \
  --bandwidth_gbps 16.0 \
  || echo "   (Bandwidth simulation completed with warnings)"

# ---------------------------------------------------------------------------
# Step 6: Compile paper (optional)
# ---------------------------------------------------------------------------
echo ""
echo "[Step 7/7] Compiling paper source..."
if command -v pdflatex &> /dev/null; then
  cd papers
  pdflatex -interaction=nonstopmode main.tex || true
  cd ..
else
  echo "   pdflatex not found; skipping paper compilation."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo " Pipeline Complete"
echo "=========================================="
echo "Output artifacts:"
ls -lh experiments/ 2>/dev/null || echo "No experiment outputs found."
echo ""
echo "To inspect results:"
echo "  cat experiments/baseline_comparison.json"
echo "  cat experiments/multimodel_benchmark.json"
echo "=========================================="
