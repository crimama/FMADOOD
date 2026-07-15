#!/usr/bin/env bash
set -euo pipefail

FSAD_ROOT="${FSAD_ROOT:-/workspace}"
DATA_ROOT="${DATA_ROOT:-/home/woojun/dataset/mvtec_ad}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_NAME="${RUN_NAME:-flow_latentbank_mvtecad1_shot4_component01_20260713_v1}"
RUN_ROOT="${RESULTS_ROOT}/${RUN_NAME}"
OBJECTS="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper"

cd "${FSAD_ROOT}"
mkdir -p "${RUN_ROOT}/logs"
export FMAD_DINOV2_OFFLINE="${FMAD_DINOV2_OFFLINE:-1}"

common_args=(
  --data-root "${DATA_ROOT}"
  --project-root /workspace
  --fsad-root "${FSAD_ROOT}"
  --objects "${OBJECTS}"
  --shots 4
  --seed 0
  --device cuda
  --backbone-model dinov2_vitb14_reg
  --preprocess-recipe fmad_shorter_edge
  --image-size 448
  --crop-size 448
  --feature-layers 2,5,8,11
  --feature-fusion layer_norm_mean
  --support-selection first
  --support-selection-seed 0
  --support-transforms identity
  --support-brightness-range 1.0,1.0
  --flow-epochs 3
  --coupling-layers 2
  --hidden-multiplier 1
  --flow-lr 2e-4
  --flow-clamp 1.9
  --flow-transform-mode flow
  --tail-weight 0.3
  --tail-top-k-ratio 0.05
  --lambda-logdet 1e-3
  --density-quantile 0.90
  --expansion-budget 1.0
  --distance-weight 1.0
  --score-mode latent_distance
  --dvt-denoise-mode none
  --normality-mode fused
  --top-percent 0.01
  --query-chunk-size 512
  --calibration-sample-size 0
  --cleanup-maps
)

run_arm() {
  local gpu="$1"
  local arm="$2"
  shift 2
  local output="${RUN_ROOT}/${arm}"
  local log="${RUN_ROOT}/logs/${arm}.log"
  if [[ -f "${output}/run_manifest.json" ]]; then
    echo "[skip] ${arm}" | tee -a "${log}"
    return
  fi
  echo "[start] arm=${arm} gpu=${gpu}" | tee "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/run_flow_tte_mvtec_ad1.py \
    "${common_args[@]}" --output-root "${output}" "$@" 2>&1 | tee -a "${log}"
  echo "[complete] arm=${arm} gpu=${gpu}" | tee -a "${log}"
}

run_arm 0 density0 --density-weight 0.0 &
pid0=$!
run_arm 1 cls_soft_w10 \
  --density-weight 0.25 \
  --context-source cls \
  --flow-context-source none \
  --memory-context-source cls \
  --context-mode soft_penalty \
  --context-weight 10.0 &
pid1=$!

status=0
wait "${pid0}" || status=1
wait "${pid1}" || status=1
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
echo "[complete] ${RUN_NAME}"
