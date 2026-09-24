#!/usr/bin/env bash
set -euo pipefail

cd /workspace/.local-sync/current
export PYTHONPATH="/workspace/.local-sync/current/src"
export TORCH_HOME="/model-cache"
export FMAD_DINOV2_OFFLINE=1

LOG_DIR="/workspace/results_remote/fewshot_scaling/logs"
mkdir -p "${LOG_DIR}"

DATA_ROOT="/home/hunim/Volume/DATA/mvtec_ad_2"
OUT_BASE="/workspace/results_remote/fewshot_scaling/ad2"

echo "================================================================="
echo "Starting Few-shot Scaling Benchmark on MVTec AD 2 (1, 2, 4, 8 shots)"
echo "Backbones: DINOv2 & DINOv3 on 4x NVIDIA RTX A6000 GPUs"
echo "Start Time: $(date)"
echo "================================================================="

# GPU 0: DINOv2 1-shot -> 2-shot
(
  echo "[GPU 0] Running DINOv2 1-shot..."
  python3 run.py \
    --config configs/benchmarks/fewshot/ad2_dinov2_1shot.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_BASE}/dinov2_1shot" \
    --device cuda:0 \
    > "${LOG_DIR}/dinov2_1shot.log" 2>&1
  echo "[GPU 0] DINOv2 1-shot complete, running 2-shot..."
  python3 run.py \
    --config configs/benchmarks/fewshot/ad2_dinov2_2shot.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_BASE}/dinov2_2shot" \
    --device cuda:0 \
    > "${LOG_DIR}/dinov2_2shot.log" 2>&1
  echo "[GPU 0] DINOv2 (1-shot, 2-shot) all complete!"
) &
PID0=$!

# GPU 1: DINOv2 4-shot -> 8-shot
(
  echo "[GPU 1] Running DINOv2 4-shot..."
  python3 run.py \
    --config configs/benchmarks/fewshot/ad2_dinov2_4shot.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_BASE}/dinov2_4shot" \
    --device cuda:1 \
    > "${LOG_DIR}/dinov2_4shot.log" 2>&1
  echo "[GPU 1] DINOv2 4-shot complete, running 8-shot..."
  python3 run.py \
    --config configs/benchmarks/fewshot/ad2_dinov2_8shot.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_BASE}/dinov2_8shot" \
    --device cuda:1 \
    > "${LOG_DIR}/dinov2_8shot.log" 2>&1
  echo "[GPU 1] DINOv2 (4-shot, 8-shot) all complete!"
) &
PID1=$!

# GPU 2: DINOv3 1-shot -> 2-shot
(
  echo "[GPU 2] Running DINOv3 1-shot..."
  python3 run.py \
    --config configs/benchmarks/fewshot/ad2_dinov3_1shot.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_BASE}/dinov3_1shot" \
    --device cuda:2 \
    > "${LOG_DIR}/dinov3_1shot.log" 2>&1
  echo "[GPU 2] DINOv3 1-shot complete, running 2-shot..."
  python3 run.py \
    --config configs/benchmarks/fewshot/ad2_dinov3_2shot.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_BASE}/dinov3_2shot" \
    --device cuda:2 \
    > "${LOG_DIR}/dinov3_2shot.log" 2>&1
  echo "[GPU 2] DINOv3 (1-shot, 2-shot) all complete!"
) &
PID2=$!

# GPU 3: DINOv3 4-shot -> 8-shot
(
  echo "[GPU 3] Running DINOv3 4-shot..."
  python3 run.py \
    --config configs/benchmarks/fewshot/ad2_dinov3_4shot.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_BASE}/dinov3_4shot" \
    --device cuda:3 \
    > "${LOG_DIR}/dinov3_4shot.log" 2>&1
  echo "[GPU 3] DINOv3 4-shot complete, running 8-shot..."
  python3 run.py \
    --config configs/benchmarks/fewshot/ad2_dinov3_8shot.yaml \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_BASE}/dinov3_8shot" \
    --device cuda:3 \
    > "${LOG_DIR}/dinov3_8shot.log" 2>&1
  echo "[GPU 3] DINOv3 (4-shot, 8-shot) all complete!"
) &
PID3=$!

echo "All 4 GPU pipelines launched!"
echo "PIDs: GPU 0=$PID0, GPU 1=$PID1, GPU 2=$PID2, GPU 3=$PID3"
echo "Waiting for all jobs to complete..."

wait $PID0
echo "[GPU 0] Done at $(date)"
wait $PID1
echo "[GPU 1] Done at $(date)"
wait $PID2
echo "[GPU 2] Done at $(date)"
wait $PID3
echo "[GPU 3] Done at $(date)"

echo "================================================================="
echo "All Few-shot Scaling Benchmark Runs Completed Successfully!"
echo "End Time: $(date)"
echo "================================================================="
