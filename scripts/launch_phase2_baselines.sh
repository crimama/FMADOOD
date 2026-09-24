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
echo "Launching Phase 2 Baselines on Idle GPUs (GPU 0, 1, 3)"
echo "  - GPU 0: FastRecon (ICCV 2023) 16, 8, 4, 2, 1-shot"
echo "  - GPU 1: RegAD (ECCV 2022 Oral) 16, 8, 4, 2, 1-shot"
echo "  - GPU 3: SuperAD (DINOv3 Backbone) 16, 8, 4, 2, 1-shot"
echo "Start Time: $(date)"
echo "================================================================="

# GPU 0: FastRecon sequence
echo "[GPU 0] Dispatching FastRecon (16 -> 8 -> 4 -> 2 -> 1 shot)..."
(
  for k in 16 8 4 2 1; do
    echo "[GPU 0] Running FastRecon ${k}-shot at $(date)..."
    python3 run.py \
      --config configs/benchmarks/baselines/fastrecon/ad2_${k}shot.yaml \
      --data-root "${BASE_DATA}" \
      --output-dir "${OUT_BASE}/fastrecon/${k}shot" \
      --device cuda:0 \
      > "${LOG_DIR}/fastrecon_${k}shot.log" 2>&1
    echo "[GPU 0] FastRecon ${k}-shot completed!"
  done
) &
PID0=$!

# GPU 1: RegAD sequence
echo "[GPU 1] Dispatching RegAD (16 -> 8 -> 4 -> 2 -> 1 shot)..."
(
  for k in 16 8 4 2 1; do
    echo "[GPU 1] Running RegAD ${k}-shot at $(date)..."
    python3 run.py \
      --config configs/benchmarks/baselines/regad/ad2_${k}shot.yaml \
      --data-root "${BASE_DATA}" \
      --output-dir "${OUT_BASE}/regad/${k}shot" \
      --device cuda:1 \
      > "${LOG_DIR}/regad_${k}shot.log" 2>&1
    echo "[GPU 1] RegAD ${k}-shot completed!"
  done
) &
PID1=$!

# GPU 3: SuperAD DINOv3 sequence
echo "[GPU 3] Dispatching SuperAD DINOv3 (16 -> 8 -> 4 -> 2 -> 1 shot)..."
(
  for k in 16 8 4 2 1; do
    echo "[GPU 3] Running SuperAD DINOv3 ${k}-shot at $(date)..."
    python3 run.py \
      --config configs/benchmarks/baselines/superad_dinov3/ad2_${k}shot.yaml \
      --data-root "${BASE_DATA}" \
      --output-dir "${OUT_BASE}/superad_dinov3/${k}shot" \
      --device cuda:3 \
      > "${LOG_DIR}/superad_dinov3_${k}shot.log" 2>&1
    echo "[GPU 3] SuperAD DINOv3 ${k}-shot completed!"
  done
) &
PID3=$!

echo "Dispatched on GPU 0 (PID $PID0), GPU 1 (PID $PID1), GPU 3 (PID $PID3)!"
echo "Awaiting completion..."

wait $PID0
echo "[GPU 0] FastRecon all shots completed at $(date)"

wait $PID1
echo "[GPU 1] RegAD all shots completed at $(date)"

wait $PID3
echo "[GPU 3] SuperAD DINOv3 all shots completed at $(date)"

echo "================================================================="
echo "Phase 2 Baselines Completed Successfully!"
echo "End Time: $(date)"
echo "================================================================="
