#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="results/benchmark_runs/logs"
mkdir -p "${LOG_DIR}"

MVTEC_DATA="/home/hun/Volume/DATA/MVTecAD"
VISA_DATA="/home/hun/Volume/DATA/VisA_pytorch/1cls"

echo "========================================================"
echo "Starting Full Benchmark Runs on Local RTX 4090"
echo "Start Time: $(date)"
echo "========================================================"

# 1. MVTec AD 1 - DINOv2 (seed 0)
echo "[1/4] Running MVTec AD 1 with DINOv2 (seed 0)..."
.venv/bin/python run.py \
  --config configs/benchmarks/mvtec_dinov2_s0.yaml \
  --data-root "${MVTEC_DATA}" \
  --output-dir results/benchmark_runs/mvtec/dinov2_s0 \
  > "${LOG_DIR}/mvtec_dinov2_s0.log" 2>&1
echo "[1/4] MVTec AD 1 with DINOv2 complete at $(date)"

# 2. MVTec AD 1 - DINOv3 (seed 0)
echo "[2/4] Running MVTec AD 1 with DINOv3 (seed 0)..."
.venv/bin/python run.py \
  --config configs/benchmarks/mvtec_dinov3_s0.yaml \
  --data-root "${MVTEC_DATA}" \
  --output-dir results/benchmark_runs/mvtec/dinov3_s0 \
  > "${LOG_DIR}/mvtec_dinov3_s0.log" 2>&1
echo "[2/4] MVTec AD 1 with DINOv3 complete at $(date)"

# 3. ViSA - DINOv2 (seed 0)
echo "[3/4] Running ViSA with DINOv2 (seed 0)..."
.venv/bin/python run.py \
  --config configs/benchmarks/visa_dinov2_s0.yaml \
  --data-root "${VISA_DATA}" \
  --output-dir results/benchmark_runs/visa/dinov2_s0 \
  > "${LOG_DIR}/visa_dinov2_s0.log" 2>&1
echo "[3/4] ViSA with DINOv2 complete at $(date)"

# 4. ViSA - DINOv3 (seed 0)
echo "[4/4] Running ViSA with DINOv3 (seed 0)..."
.venv/bin/python run.py \
  --config configs/benchmarks/visa_dinov3_s0.yaml \
  --data-root "${VISA_DATA}" \
  --output-dir results/benchmark_runs/visa/dinov3_s0 \
  > "${LOG_DIR}/visa_dinov3_s0.log" 2>&1
echo "[4/4] ViSA with DINOv3 complete at $(date)"

echo "========================================================"
echo "All Benchmark Runs Successfully Completed!"
echo "End Time: $(date)"
echo "========================================================"
