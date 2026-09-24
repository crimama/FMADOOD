#!/usr/bin/env bash
set -euo pipefail

cd /workspace/.local-sync/current
export PYTHONPATH="/workspace/.local-sync/current/src"
export TORCH_HOME="/model-cache"
export FMAD_DINOV2_OFFLINE=1

LOG_DIR="/workspace/results_remote/dist_shift/logs"
mkdir -p "${LOG_DIR}"

BASE_DATA="/home/hunim/Volume/DATA/MVTecAD_corrupted"
OUT_BASE="/workspace/results_remote/dist_shift/mvtec"

echo "================================================================="
echo "Starting ICCV 2023 Distribution Shift Benchmark (MVTec AD 16-shot)"
echo "Backbone: DINOv3 (dinov3_vitl16) on 4x NVIDIA RTX A6000 GPUs"
echo "Start Time: $(date)"
echo "================================================================="

# GPU 0: Brightness
echo "[GPU 0] Launching DINOv3 Brightness Shift..."
python3 run.py \
  --config configs/benchmarks/dist_shift/mvtec_dinov3_brightness.yaml \
  --data-root "${BASE_DATA}/brightness" \
  --output-dir "${OUT_BASE}/dinov3_brightness" \
  --device cuda:0 \
  > "${LOG_DIR}/dinov3_brightness.log" 2>&1 &
PID0=$!

# GPU 1: Contrast
echo "[GPU 1] Launching DINOv3 Contrast Shift..."
python3 run.py \
  --config configs/benchmarks/dist_shift/mvtec_dinov3_contrast.yaml \
  --data-root "${BASE_DATA}/contrast" \
  --output-dir "${OUT_BASE}/dinov3_contrast" \
  --device cuda:1 \
  > "${LOG_DIR}/dinov3_contrast.log" 2>&1 &
PID1=$!

# GPU 2: Defocus Blur
echo "[GPU 2] Launching DINOv3 Defocus Blur Shift..."
python3 run.py \
  --config configs/benchmarks/dist_shift/mvtec_dinov3_defocus_blur.yaml \
  --data-root "${BASE_DATA}/defocus_blur" \
  --output-dir "${OUT_BASE}/dinov3_defocus_blur" \
  --device cuda:2 \
  > "${LOG_DIR}/dinov3_defocus_blur.log" 2>&1 &
PID2=$!

# GPU 3: Gaussian Noise
echo "[GPU 3] Launching DINOv3 Gaussian Noise Shift..."
python3 run.py \
  --config configs/benchmarks/dist_shift/mvtec_dinov3_gaussian_noise.yaml \
  --data-root "${BASE_DATA}/gaussian_noise" \
  --output-dir "${OUT_BASE}/dinov3_gaussian_noise" \
  --device cuda:3 \
  > "${LOG_DIR}/dinov3_gaussian_noise.log" 2>&1 &
PID3=$!

echo "All 4 DINOv3 distribution shift jobs launched!"
echo "PIDs: GPU 0=$PID0 (brightness), GPU 1=$PID1 (contrast), GPU 2=$PID2 (defocus_blur), GPU 3=$PID3 (gaussian_noise)"
echo "Waiting for all jobs to complete..."

wait $PID0
echo "[GPU 0] DINOv3 Brightness completed at $(date)"
wait $PID1
echo "[GPU 1] DINOv3 Contrast completed at $(date)"
wait $PID2
echo "[GPU 2] DINOv3 Defocus Blur completed at $(date)"
wait $PID3
echo "[GPU 3] DINOv3 Gaussian Noise completed at $(date)"

echo "================================================================="
echo "All DINOv3 Distribution Shift Benchmark Runs Completed Successfully!"
echo "End Time: $(date)"
echo "================================================================="
