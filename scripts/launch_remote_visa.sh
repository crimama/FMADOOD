#!/usr/bin/env bash
set -euo pipefail

cd /workspace/.local-sync/current
export PYTHONPATH="/workspace/.local-sync/current/src"
export TORCH_HOME="/model-cache"
export FMAD_DINOV2_OFFLINE=1

mkdir -p /workspace/results_remote/benchmarks/logs

echo "Starting ViSA on dsba3 GPU 0 (dinov2) and GPU 1 (dinov3)..."

# Launch GPU 0 (dinov2) in background
python3 run.py \
  --config configs/benchmarks/visa_dinov2_s0.yaml \
  --data-root /home/hunim/Volume/DATA/VisA_pytorch/1cls \
  --output-dir /workspace/results_remote/benchmarks/visa/dinov2_s0 \
  --device cuda:0 \
  > /workspace/results_remote/benchmarks/logs/visa_dinov2_s0.log 2>&1 &
PID0=$!

# Launch GPU 1 (dinov3) in background
python3 run.py \
  --config configs/benchmarks/visa_dinov3_s0.yaml \
  --data-root /home/hunim/Volume/DATA/VisA_pytorch/1cls \
  --output-dir /workspace/results_remote/benchmarks/visa/dinov3_s0 \
  --device cuda:1 \
  > /workspace/results_remote/benchmarks/logs/visa_dinov3_s0.log 2>&1 &
PID1=$!

echo "ViSA runs launched! PID0=$PID0 (cuda:0, dinov2), PID1=$PID1 (cuda:1, dinov3)"
wait $PID0
wait $PID1
echo "Both ViSA runs complete on dsba3!"
