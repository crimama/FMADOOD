#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/workspace/.local-sync/current"
cd "${WORKDIR}"
export PYTHONPATH="${WORKDIR}/src"
export TORCH_HOME="/model-cache"
export FMAD_DINOV2_OFFLINE=1

OUT_ROOT="/workspace/results_remote/private_test_mixed_20260917"
DATA_ROOT="/home/hunim/Volume/DATA/mvtec_ad_2"
mkdir -p "${OUT_ROOT}/logs"

echo "=== Launching 4 GPU lanes on dsba3 for test_private_mixed (WORKDIR: ${WORKDIR}) ==="

# Lane 0: GPU 0 (dinov2 seed 0 -> dinov2 seed 1)
nohup bash -c "
  cd '${WORKDIR}' && export PYTHONPATH='${WORKDIR}/src' && export TORCH_HOME='/model-cache' && export FMAD_DINOV2_OFFLINE=1
  python3 scripts/run_private_inference.py --config configs/methods/tail_bag_RSG3.yaml --seed 0 --split test_private_mixed --data-root '${DATA_ROOT}' --output-dir '${OUT_ROOT}/dinov2/seed0' --device cuda:0 > '${OUT_ROOT}/logs/dinov2_s0.log' 2>&1
  python3 scripts/run_private_inference.py --config configs/methods/tail_bag_RSG3.yaml --seed 1 --split test_private_mixed --data-root '${DATA_ROOT}' --output-dir '${OUT_ROOT}/dinov2/seed1' --device cuda:0 > '${OUT_ROOT}/logs/dinov2_s1.log' 2>&1
" >/dev/null 2>&1 &
PID0=$!
echo "Lane 0 (GPU 0: dinov2 s0, s1 [mixed]) launched with PID ${PID0}"

# Lane 1: GPU 1 (dinov2 seed 2 -> dinov2 seed 3)
nohup bash -c "
  cd '${WORKDIR}' && export PYTHONPATH='${WORKDIR}/src' && export TORCH_HOME='/model-cache' && export FMAD_DINOV2_OFFLINE=1
  python3 scripts/run_private_inference.py --config configs/methods/tail_bag_RSG3.yaml --seed 2 --split test_private_mixed --data-root '${DATA_ROOT}' --output-dir '${OUT_ROOT}/dinov2/seed2' --device cuda:1 > '${OUT_ROOT}/logs/dinov2_s2.log' 2>&1
  python3 scripts/run_private_inference.py --config configs/methods/tail_bag_RSG3.yaml --seed 3 --split test_private_mixed --data-root '${DATA_ROOT}' --output-dir '${OUT_ROOT}/dinov2/seed3' --device cuda:1 > '${OUT_ROOT}/logs/dinov2_s3.log' 2>&1
" >/dev/null 2>&1 &
PID1=$!
echo "Lane 1 (GPU 1: dinov2 s2, s3 [mixed]) launched with PID ${PID1}"

# Lane 2: GPU 2 (dinov3 seed 0 -> dinov3 seed 1)
nohup bash -c "
  cd '${WORKDIR}' && export PYTHONPATH='${WORKDIR}/src' && export TORCH_HOME='/model-cache' && export FMAD_DINOV2_OFFLINE=1
  python3 scripts/run_private_inference.py --config configs/methods/tail_bag_RSG3_dinov3.yaml --seed 0 --split test_private_mixed --data-root '${DATA_ROOT}' --output-dir '${OUT_ROOT}/dinov3/seed0' --device cuda:2 > '${OUT_ROOT}/logs/dinov3_s0.log' 2>&1
  python3 scripts/run_private_inference.py --config configs/methods/tail_bag_RSG3_dinov3.yaml --seed 1 --split test_private_mixed --data-root '${DATA_ROOT}' --output-dir '${OUT_ROOT}/dinov3/seed1' --device cuda:2 > '${OUT_ROOT}/logs/dinov3_s1.log' 2>&1
" >/dev/null 2>&1 &
PID2=$!
echo "Lane 2 (GPU 2: dinov3 s0, s1 [mixed]) launched with PID ${PID2}"

# Lane 3: GPU 3 (dinov3 seed 2 -> dinov3 seed 3)
nohup bash -c "
  cd '${WORKDIR}' && export PYTHONPATH='${WORKDIR}/src' && export TORCH_HOME='/model-cache' && export FMAD_DINOV2_OFFLINE=1
  python3 scripts/run_private_inference.py --config configs/methods/tail_bag_RSG3_dinov3.yaml --seed 2 --split test_private_mixed --data-root '${DATA_ROOT}' --output-dir '${OUT_ROOT}/dinov3/seed2' --device cuda:3 > '${OUT_ROOT}/logs/dinov3_s2.log' 2>&1
  python3 scripts/run_private_inference.py --config configs/methods/tail_bag_RSG3_dinov3.yaml --seed 3 --split test_private_mixed --data-root '${DATA_ROOT}' --output-dir '${OUT_ROOT}/dinov3/seed3' --device cuda:3 > '${OUT_ROOT}/logs/dinov3_s3.log' 2>&1
" >/dev/null 2>&1 &
PID3=$!
echo "Lane 3 (GPU 3: dinov3 s2, s3 [mixed]) launched with PID ${PID3}"

echo "All 4 lanes successfully launched for test_private_mixed."
