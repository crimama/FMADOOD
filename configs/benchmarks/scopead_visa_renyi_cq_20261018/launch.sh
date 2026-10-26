#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${REMOTE_WORKDIR:?Set REMOTE_WORKDIR to immutable release}"
exp_release="$REMOTE_WORKDIR"
set -a
source configs/remote/dsba3.env
set +a
export REMOTE_WORKDIR="$exp_release"
export REMOTE_PYTHON_HOME=/workspace/.local-runtime/python311-cu124-036828932804
gpus=(GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62)
export REMOTE_GPU_DEVICES="$(IFS=,; echo "${gpus[*]}")"
cfg=configs/benchmarks/scopead_visa_renyi_cq_20261018
labels=(visa_scale_order_renyi_cq_s01 visa_scale_order_renyi_cq_s02 visa_scale_order_renyi_cq_s04 visa_scale_order_renyi_cq_s08 visa_scale_order_renyi_cq_s16)
configs=()
stages=()
for label in "${labels[@]}"; do
  configs+=("$cfg/${label}.yaml")
  stages+=("${label}")
done
objects=(candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum)
exec bash scripts/run_experiments.sh --backend remote --container hun_fsad_tta_012 \
  --configs "${configs[@]}" --labels "${stages[@]}" \
  --objects "${objects[@]}" \
  --devices "${gpus[@]}" --data-root /home/hunim/Volume/DATA/VisA_pytorch/1cls \
  --output-root /workspace/results_local_compatible/scopead_visa_renyi_cq_20261018 \
  --split visa_test_public_seen_development_renyi_cq_seed0 \
  --timeout 10800 --workers 1
