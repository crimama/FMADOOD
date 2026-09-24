#!/usr/bin/env bash
set -euo pipefail

cd /workspace/.local-sync/current
export PYTHONPATH="/workspace/.local-sync/current/src"
export TORCH_HOME="/model-cache"
export FMAD_DINOV2_OFFLINE=1

LOG_DIR="/workspace/results_remote/ablation/logs"
mkdir -p "${LOG_DIR}"

BASE_DATA="/home/hunim/Volume/DATA/mvtec_ad_2"
OUT_BASE="/workspace/results_remote/ablation"

echo "================================================================="
echo "Starting Zero-Parameter Geometric Alignment Ablation Study"
echo "Dataset: MVTec AD 2 (8 objects) | Backbone: DINOv2 16-shot"
echo "Hardware: 4x NVIDIA RTX A6000 GPUs on dsba3"
echo "Start Time: $(date)"
echo "================================================================="

# GPU 0: w/o CQ
echo "[GPU 0] Launching w/o CQ (prediction_postprocess: none)..."
python3 run.py \
  --config configs/benchmarks/ablation/ad2_dinov2_16shot_wo_cq.yaml \
  --data-root "${BASE_DATA}" \
  --output-dir "${OUT_BASE}/wo_cq" \
  --device cuda:0 \
  > "${LOG_DIR}/wo_cq.log" 2>&1 &
PID0=$!

# GPU 1: w/o RSG
echo "[GPU 1] Launching w/o RSG (layer_fusion: mean)..."
python3 run.py \
  --config configs/benchmarks/ablation/ad2_dinov2_16shot_wo_rsg.yaml \
  --data-root "${BASE_DATA}" \
  --output-dir "${OUT_BASE}/wo_rsg" \
  --device cuda:1 \
  > "${LOG_DIR}/wo_rsg.log" 2>&1 &
PID1=$!

# GPU 2: w/o Tailscale
echo "[GPU 2] Launching w/o Tailscale (score_normalization: mad)..."
python3 run.py \
  --config configs/benchmarks/ablation/ad2_dinov2_16shot_wo_tailscale.yaml \
  --data-root "${BASE_DATA}" \
  --output-dir "${OUT_BASE}/wo_tailscale" \
  --device cuda:2 \
  > "${LOG_DIR}/wo_tailscale.log" 2>&1 &
PID2=$!

# GPU 3: w/o Multi-resolution, followed by Vanilla Baseline
echo "[GPU 3] Launching w/o Multi-resolution & Vanilla Baseline sequence..."
(
  python3 run.py \
    --config configs/benchmarks/ablation/ad2_dinov2_16shot_wo_multires.yaml \
    --data-root "${BASE_DATA}" \
    --output-dir "${OUT_BASE}/wo_multires" \
    --device cuda:3 \
    > "${LOG_DIR}/wo_multires.log" 2>&1
  echo "[GPU 3] wo_multires finished. Launching vanilla baseline..."
  python3 run.py \
    --config configs/benchmarks/ablation/ad2_dinov2_16shot_vanilla.yaml \
    --data-root "${BASE_DATA}" \
    --output-dir "${OUT_BASE}/vanilla" \
    --device cuda:3 \
    > "${LOG_DIR}/vanilla.log" 2>&1
) &
PID3=$!

echo "All ablation jobs dispatched!"
echo "PIDs: GPU 0=$PID0 (wo_cq), GPU 1=$PID1 (wo_rsg), GPU 2=$PID2 (wo_tailscale), GPU 3=$PID3 (wo_multires + vanilla)"
echo "Waiting for all jobs to complete..."

wait $PID0
echo "[GPU 0] w/o CQ completed at $(date)"
wait $PID1
echo "[GPU 1] w/o RSG completed at $(date)"
wait $PID2
echo "[GPU 2] w/o Tailscale completed at $(date)"
wait $PID3
echo "[GPU 3] wo_multires and vanilla completed at $(date)"

echo "================================================================="
echo "All Ablation Runs Completed Successfully!"
echo "End Time: $(date)"
echo "================================================================="
