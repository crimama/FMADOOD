#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="results/local_runs/private_test_mixed_20260917"
DATA_ROOT="/home/hun/Volume/DATA/mvtec_ad_2"
mkdir -p "${OUT_ROOT}/logs"

echo "=== Launching local lane on RTX 4090 for test_private_mixed ==="

# Local: GPU 0 (dinov3 seed 4 -> dinov2 seed 4)
echo "Starting local dinov3 seed 4 [mixed]..."
.venv/bin/python scripts/run_private_inference.py --config configs/methods/tail_bag_RSG3_dinov3.yaml --seed 4 --split test_private_mixed --data-root "${DATA_ROOT}" --output-dir "${OUT_ROOT}/dinov3/seed4" --device cuda:0 > "${OUT_ROOT}/logs/dinov3_s4.log" 2>&1
echo "Local dinov3 seed 4 complete. Launching dinov2 seed 4 [mixed]..."
.venv/bin/python scripts/run_private_inference.py --config configs/methods/tail_bag_RSG3.yaml --seed 4 --split test_private_mixed --data-root "${DATA_ROOT}" --output-dir "${OUT_ROOT}/dinov2/seed4" --device cuda:0 > "${OUT_ROOT}/logs/dinov2_s4.log" 2>&1
echo "Local runs (dinov3 s4, dinov2 s4) [mixed] complete."
