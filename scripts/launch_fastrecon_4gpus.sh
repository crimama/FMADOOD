#!/usr/bin/env bash
set -euo pipefail

cd /workspace/.local-sync/current
export PYTHONPATH="/workspace/.local-sync/current/src"
export TORCH_HOME="/model-cache"
export FMAD_DINOV2_OFFLINE=1
export PYTHONUNBUFFERED=1

LOG_DIR="/workspace/results_remote/baselines/logs"
mkdir -p "${LOG_DIR}"

BASE_DATA="/home/hunim/Volume/DATA/mvtec_ad_2"
OUT_BASE="/workspace/results_remote/baselines"

echo "================================================================="
echo "Launching FastRecon (ICCV 2023) across 4 GPUs on dsba3"
echo "  - GPU 0: FastRecon 16-shot"
echo "  - GPU 1: FastRecon 8-shot"
echo "  - GPU 2: FastRecon 4-shot -> 2-shot"
echo "  - GPU 3: FastRecon 1-shot"
echo "Start Time: $(date)"
echo "================================================================="

# Clean old incomplete 16shot directory if needed
rm -rf "${OUT_BASE}/fastrecon/16shot"

# GPU 0: 16-shot
(
  echo "[GPU 0] Running FastRecon 16-shot at $(date)..."
  python3 -u run.py \
    --config configs/benchmarks/baselines/fastrecon/ad2_16shot.yaml \
    --data-root "${BASE_DATA}" \
    --output-dir "${OUT_BASE}/fastrecon/16shot" \
    --device cuda:0 \
    > "${LOG_DIR}/fastrecon_16shot.log" 2>&1
  echo "[GPU 0] FastRecon 16-shot completed at $(date)!"
) &
PID0=$!

# GPU 1: 8-shot
(
  echo "[GPU 1] Running FastRecon 8-shot at $(date)..."
  python3 -u run.py \
    --config configs/benchmarks/baselines/fastrecon/ad2_8shot.yaml \
    --data-root "${BASE_DATA}" \
    --output-dir "${OUT_BASE}/fastrecon/8shot" \
    --device cuda:1 \
    > "${LOG_DIR}/fastrecon_8shot.log" 2>&1
  echo "[GPU 1] FastRecon 8-shot completed at $(date)!"
) &
PID1=$!

# GPU 2: 4-shot then 2-shot
(
  for k in 4 2; do
    echo "[GPU 2] Running FastRecon ${k}-shot at $(date)..."
    python3 -u run.py \
      --config configs/benchmarks/baselines/fastrecon/ad2_${k}shot.yaml \
      --data-root "${BASE_DATA}" \
      --output-dir "${OUT_BASE}/fastrecon/${k}shot" \
      --device cuda:2 \
      > "${LOG_DIR}/fastrecon_${k}shot.log" 2>&1
    echo "[GPU 2] FastRecon ${k}-shot completed at $(date)!"
  done
) &
PID2=$!

# GPU 3: 1-shot
(
  echo "[GPU 3] Running FastRecon 1-shot at $(date)..."
  python3 -u run.py \
    --config configs/benchmarks/baselines/fastrecon/ad2_1shot.yaml \
    --data-root "${BASE_DATA}" \
    --output-dir "${OUT_BASE}/fastrecon/1shot" \
    --device cuda:3 \
    > "${LOG_DIR}/fastrecon_1shot.log" 2>&1
  echo "[GPU 3] FastRecon 1-shot completed at $(date)!"
) &
PID3=$!

echo "Dispatched on GPU 0 (PID $PID0), GPU 1 (PID $PID1), GPU 2 (PID $PID2), GPU 3 (PID $PID3)!"
echo "Awaiting completion..."

wait $PID0
wait $PID1
wait $PID2
wait $PID3

echo "================================================================="
echo "All FastRecon baselines completed successfully at $(date)!"
echo "================================================================="
