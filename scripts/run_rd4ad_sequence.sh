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

echo "[GPU 2] Dispatching RD4AD (DINOv2 teacher) sequence (8 -> 4 -> 2 -> 1 shot)..."
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

echo "[GPU 2] All RD4AD shots completed successfully at $(date)!"
