#!/usr/bin/env bash
set -euo pipefail

cd /workspace/.local-sync/current
export PYTHONPATH="/workspace/.local-sync/current/src"
export TORCH_HOME="/model-cache"
export FMAD_DINOV2_OFFLINE=1

LOG_DIR="/workspace/results_remote/baselines/logs"
mkdir -p "${LOG_DIR}"

BASE_DATA="/home/hunim/Volume/DATA/mvtec_ad_2"
OUT_BASE="/workspace/results_remote/baselines"

echo "================================================================="
echo "Starting 4-GPU Few-Shot Anomaly Detection Baselines Benchmark"
echo "Dataset: MVTec AD 2 (8 objects)"
echo "Hardware: 4x NVIDIA RTX A6000 GPUs on dsba3"
echo "Mapping:"
echo "  - GPU 0: SuperAD (8, 4, 2, 1-shot)"
echo "  - GPU 1: PatchCore (16, 8, 4, 2, 1-shot)"
echo "  - GPU 2: RD4AD (8, 4, 2, 1-shot)"
echo "  - GPU 3: RN-FMLK (8, 4, 2, 1-shot)"
echo "Start Time: $(date)"
echo "================================================================="

# -------------------------------------------------------------
# GPU 0: SuperAD sequence
# -------------------------------------------------------------
echo "[GPU 0] Dispatching SuperAD sequence (8 -> 4 -> 2 -> 1 shot)..."
(
  for k in 8 4 2 1; do
    echo "[GPU 0] Running SuperAD ${k}-shot at $(date)..."
    python3 run.py \
      --config configs/benchmarks/baselines/superad/ad2_${k}shot.yaml \
      --data-root "${BASE_DATA}" \
      --output-dir "${OUT_BASE}/superad/${k}shot" \
      --device cuda:0 \
      > "${LOG_DIR}/superad_${k}shot.log" 2>&1
    echo "[GPU 0] SuperAD ${k}-shot completed!"
  done
) &
PID0=$!

# -------------------------------------------------------------
# GPU 1: PatchCore sequence
# -------------------------------------------------------------
echo "[GPU 1] Dispatching PatchCore sequence (16 -> 8 -> 4 -> 2 -> 1 shot)..."
(
  for k in 16 8 4 2 1; do
    echo "[GPU 1] Running PatchCore ${k}-shot at $(date)..."
    python3 run.py \
      --config configs/benchmarks/baselines/patchcore/ad2_${k}shot.yaml \
      --data-root "${BASE_DATA}" \
      --output-dir "${OUT_BASE}/patchcore/${k}shot" \
      --device cuda:1 \
      > "${LOG_DIR}/patchcore_${k}shot.log" 2>&1
    echo "[GPU 1] PatchCore ${k}-shot completed!"
  done
) &
PID1=$!

# -------------------------------------------------------------
# GPU 2: RD4AD sequence
# -------------------------------------------------------------
echo "[GPU 2] Dispatching RD4AD sequence (8 -> 4 -> 2 -> 1 shot)..."
(
  for k in 8 4 2 1; do
    echo "[GPU 2] Running RD4AD ${k}-shot at $(date)..."
    python3 run.py \
      --config configs/benchmarks/baselines/rd4ad/ad2_${k}shot.yaml \
      --data-root "${BASE_DATA}" \
      --output-dir "${OUT_BASE}/rd4ad/${k}shot" \
      --device cuda:2 \
      > "${LOG_DIR}/rd4ad_${k}shot.log" 2>&1
    echo "[GPU 2] RD4AD ${k}-shot completed!"
  done
) &
PID2=$!

# -------------------------------------------------------------
# GPU 3: RN-FMLK sequence
# -------------------------------------------------------------
echo "[GPU 3] Dispatching RN-FMLK sequence (8 -> 4 -> 2 -> 1 shot)..."
(
  for k in 8 4 2 1; do
    echo "[GPU 3] Running RN-FMLK ${k}-shot at $(date)..."
    python3 run.py \
      --config configs/benchmarks/baselines/rn_fmlk/ad2_${k}shot.yaml \
      --data-root "${BASE_DATA}" \
      --output-dir "${OUT_BASE}/rn_fmlk/${k}shot" \
      --device cuda:3 \
      > "${LOG_DIR}/rn_fmlk_${k}shot.log" 2>&1
    echo "[GPU 3] RN-FMLK ${k}-shot completed!"
  done
) &
PID3=$!

echo "All baseline jobs dispatched on 4 GPUs!"
echo "PIDs: GPU 0=$PID0 (SuperAD), GPU 1=$PID1 (PatchCore), GPU 2=$PID2 (RD4AD), GPU 3=$PID3 (RN-FMLK)"
echo "Awaiting completion..."

wait $PID0
echo "[GPU 0] SuperAD all shots completed at $(date)"

wait $PID1
echo "[GPU 1] PatchCore all shots completed at $(date)"

wait $PID2
echo "[GPU 2] RD4AD all shots completed at $(date)"

wait $PID3
echo "[GPU 3] RN-FMLK all shots completed at $(date)"

echo "================================================================="
echo "All 4-GPU Baseline Benchmarks Completed Successfully!"
echo "End Time: $(date)"
echo "================================================================="
